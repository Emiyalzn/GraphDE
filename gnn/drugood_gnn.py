import os.path as osp
import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Tanh, Linear, Softmax, Parameter
from torch_geometric.nn import global_max_pool
from .overloader import overload
from .models import create_model, LSM, CosineLSM
from utils.helper import rand_prop, consis_loss

class DrugNet(torch.nn.Module):

    def __init__(self, in_channels, num_classes=2, n_train_data=None, args=None):
        super(DrugNet, self).__init__()

        self.num_classes = num_classes
        self.prior_ratio = args.prior_ratio
        self.n_train_data = n_train_data
        # for grand
        self.dropnode = args.dropnode
        self.temp = args.tem
        self.K = args.sample
        self.order = args.order
        self.lam = args.lam
        self.grand = args.grand

        self.gnn_model = create_model(args.backbone, in_channels, args.channels, args.num_unit, args.dropout,
                                    args.dropedge, args.bn)
        self.structure_model = CosineLSM(in_channels, args.channels, args.dropout, args.neg_ratio, m=2)
        self.out_mlp = torch.nn.Sequential(
            Linear(args.channels, 2 * args.channels),
            ReLU(),
            Linear(2 * args.channels, num_classes)
        )
        if n_train_data:
            self.e_logits = Parameter(torch.ones([n_train_data, 2], dtype=torch.float))

        self.CELoss = nn.CrossEntropyLoss(reduction='none')
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=-1)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        
    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        if self.grand:
            x = rand_prop(x, edge_index, self.order, self.dropnode, self.training)
        graph_rep = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_pred(graph_rep)

    @overload
    def get_node_rep(self, x, edge_index, edge_attr, batch):
        node_rep, batch = self.gnn_model(x, edge_index, edge_attr, batch)
        return node_rep, batch

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_rep, batch = self.get_node_rep(x, edge_index, edge_attr, batch)
        graph_rep = global_max_pool(node_rep, batch)
        return graph_rep

    def get_pred(self, graph_rep):
        pred = self.out_mlp(graph_rep)
        return pred

    def infer_e_gx(self, x, edge_index, batch):
        """
        Infer the environment variable based
        on the structure estimation model (for testing OOD data)
        """
        graph_neglogprob = self.structure_model.get_reg_loss(x, edge_index, batch)
        graph_prob = torch.exp(-graph_neglogprob)
        e_inferred = graph_prob / (graph_prob + 0.5)
        return e_inferred

    def infer_e_gxy(self, x, edge_index, edge_attr, batch, y):
        """
        Infer the environment variable based
        on the structure estimation and classification model 
        (for training OOD data, i.e. the outliers)
        """
        graph_neglogprob = self.structure_model.get_reg_loss(x, edge_index, batch)
        graph_prob = torch.exp(-graph_neglogprob)

        y_pred = self.forward(x, edge_index, edge_attr, batch)
        y_neglogprob = self.CELoss(y_pred, y)
        y_prob = torch.exp(-y_neglogprob)

        e_in = graph_prob * y_prob
        e_out = (1 / self.num_classes) * 1 / 2
        e_inferred = e_in / (e_in + e_out)
        return e_inferred
    
    def get_kl_loss(self, e_logprob):
        e_prior = torch.tensor([[1 - self.prior_ratio, self.prior_ratio]], dtype=torch.float, \
            device=e_logprob.device).expand(e_logprob.size(0), -1)
        kl_loss = self.KLDivLoss(e_logprob, e_prior)
        return kl_loss
    
    def get_pred_loss(self, x, edge_index, edge_attr, batch, y):
        graph_rep = self.get_graph_rep(x, edge_index, edge_attr, batch)
        pred = self.get_pred(graph_rep)
        loss = torch.mean(self.CELoss(pred, y))
        return loss

    def get_grand_pred_loss(self, x, edge_index, edge_attr, batch, y):
        output_list = []
        for k in range(self.K):
            output_list.append(torch.log_softmax(self(x, edge_index, edge_attr, batch), dim=-1))
        loss_train = 0.
        for k in range(self.K):
            loss_train += F.nll_loss(output_list[k], y)
        loss_train = loss_train / self.K
        loss_consis = consis_loss(output_list, self.temp)
        return loss_train + self.lam * loss_consis

    def get_graphde_v_loss(self, x, edge_index, edge_attr, batch, idx, y):
        # get environment variable value
        e_prob = self.Softmax(self.e_logits)[idx, :]
        e_log_prob = e_prob.log()
        e_in, e_out = e_prob[:, 0], e_prob[:, 1]
        
        graph_rep = self.get_graph_rep(x, edge_index, edge_attr, batch)
        pred = self.get_pred(graph_rep)
        graph_reg_loss = self.structure_model.get_reg_loss(x, edge_index, batch)
        
        # calculate kl loss
        kl_loss = self.get_kl_loss(e_log_prob)
        
        # calculate in-distribution loss
        inlier_pred_loss = torch.mean(e_in * self.CELoss(pred, y))
        inlier_reg_loss = torch.mean(e_in * graph_reg_loss)
        
        # the outlier prob is assumed to be uniform
        uni_logprob_pred = torch.full((len(y),), 1 / self.num_classes, device=y.device).log()
        outlier_pred_loss = torch.mean(e_out * -uni_logprob_pred)
        uni_logprob_reg = torch.full((len(y),), 1 / 2, device=x.device).log()
        outlier_reg_loss = torch.mean(e_out * -uni_logprob_reg)
        
        inlier_loss = inlier_pred_loss + inlier_reg_loss
        outlier_loss = outlier_pred_loss + outlier_reg_loss
        
        return inlier_loss + outlier_loss + kl_loss

    def get_graphde_a_loss(self, x, edge_index, edge_attr, batch, y):
        graph_neglogprob = self.structure_model.get_reg_loss(x, edge_index, batch)
        graph_prob = torch.exp(-graph_neglogprob)

        y_pred = self.forward(x, edge_index, edge_attr, batch)
        y_neglogprob = self.CELoss(y_pred, y)
        y_prob = torch.exp(-y_neglogprob)

        e_in = graph_prob * y_prob
        e_out = (1 / self.num_classes) * 1 / 2
        e_inferred = e_in / (e_in + e_out)

        logprob = torch.unsqueeze(-graph_neglogprob - y_neglogprob, dim=1)
        log_uniform = torch.tensor([[e_out]], dtype=torch.float, device=logprob.device) \
            .log().expand(logprob.size(0), -1)
        divider = torch.logsumexp(torch.cat([logprob, log_uniform], dim=1), 1, True)
        e_in_log = logprob - divider
        e_out_log = log_uniform - divider
        e_inferred_logprob = torch.cat([e_in_log, e_out_log], dim=1)

        # calculate loss
        kl_loss = self.get_kl_loss(e_inferred_logprob)
        inlier_pred_loss = torch.mean(e_inferred * y_neglogprob)
        inlier_reg_loss = torch.mean(e_inferred * graph_neglogprob)

        # the outlier prob is assumed to be uniform
        uni_logprob_pred = torch.full((len(y),), 1 / self.num_classes, device=y.device).log()
        outlier_pred_loss = torch.mean((1 - e_inferred) * -uni_logprob_pred)
        uni_logprob_reg = torch.full((len(y),), 1 / 2, device=x.device).log()
        outlier_reg_loss = torch.mean((1 - e_inferred) * -uni_logprob_reg)

        inlier_loss = inlier_pred_loss + inlier_reg_loss
        outlier_loss = outlier_pred_loss + outlier_reg_loss

        return inlier_loss + outlier_loss + kl_loss