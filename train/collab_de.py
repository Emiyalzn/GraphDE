import torch
import argparse
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import os
import os.path as osp
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.measure import calc_acc, ood_detection
import torch.nn.functional as F
from copy import deepcopy
import torch_geometric.transforms as T
from datasets import TUDataset
from gnn import CollabNet

in_channels = 492
num_classes = 3
n_train_data, n_val_data, n_in_test_data, n_out_test_data = 1000, 300, 500, 500
dataset_dir = 'data/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training arguments for MNISTSP')
    parser.add_argument('--epoch', default=400, type=int, help='training epoch')
    parser.add_argument('--seed', nargs='?', default='[42,43,44,45,46]', help='random seed')
    parser.add_argument('--channels', default=64, type=int, help='width of network')
    # hyper
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--biased_ratio', default=0.1, type=float, help='prior outlier mixed ratio')
    parser.add_argument('--backbone', default='GAT', type=str, help='select backbone model')
    parser.add_argument('--prior_ratio', default=None, type=float, help='prior outlier ratio')
    parser.add_argument('--neg_ratio', default=1.0, type=float, help='edge negative sampling ratio')
    # basic
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_unit', default=2, type=int, help='gnn layers number')
    parser.add_argument('--net_lr', default=1e-2, type=float, help='learning rate for the predictor')
    parser.add_argument('--e_lr', default=1e-1, type=float, help='learning rate for the learnable e')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for training')
    parser.add_argument('--dropedge', default=0.0, type=float, help='dropedge for regularization')
    parser.add_argument('--bn', action='store_true', help='if using batchnorm')
    parser.add_argument('--graphde_v', action='store_true', help='if enable GraphDE-v')
    parser.add_argument('--graphde_a', action='store_true', help='if enable GraphDE-a')
    # grand
    parser.add_argument('--dropnode', default=0.2, type=float, help='dropnode rate for grand')
    parser.add_argument('--lam', default=1., type=float, help='consistency loss weight for grand')
    parser.add_argument('--tem', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--sample', default=4, type=int, help='sampling time of dropnode')
    parser.add_argument('--order', default=2, type=int, help='propagation step')
    parser.add_argument('--grand', action='store_true', help='if enable grand training')

    args = parser.parse_args()
    args.seed = eval(args.seed)
    if not args.prior_ratio:
        args.prior_ratio = args.biased_ratio
    if args.graphde_a:
        args.pretrain = max(100, args.pretrain)

    return args

if __name__ == '__main__':
    args = parse_arguments()

    dataset = TUDataset(dataset_dir, name='COLLAB')
    transform = T.OneHotDegree(491)
    dataset.transform = transform
    
    # select the in-distribution and out-of-distribution data
    iid_index, mixed_index, ood_index = [], [], []
    for idx, data in enumerate(dataset):
        if data.num_nodes >= 45 and data.num_nodes <= 80:
            iid_index.append(idx)
        elif data.num_nodes > 80 and data.num_nodes <= 100:
            mixed_index.append(idx)
        elif data.num_nodes > 100:
            ood_index.append(idx)
    iid_dataset = dataset.index_select(iid_index)
    mixed_dataset = dataset.index_select(mixed_index)
    ood_dataset = dataset.index_select(ood_index)
    print(f"iid: {len(iid_dataset)}, mixed: {len(mixed_dataset)}, ood: {len(ood_dataset)}")

    perm_idx = torch.randperm(len(iid_dataset), generator=torch.Generator().manual_seed(0))
    iid_dataset = iid_dataset[perm_idx]
    perm_idx = torch.randperm(len(mixed_dataset), generator=torch.Generator().manual_seed(0))
    mixed_dataset = mixed_dataset[perm_idx]
    perm_idx = torch.randperm(len(ood_dataset), generator=torch.Generator().manual_seed(0))
    ood_dataset = ood_dataset[perm_idx]

    train_dataset = iid_dataset[:n_train_data]
    val_dataset = iid_dataset[n_train_data:n_train_data+n_val_data]
    in_test_dataset = iid_dataset[-n_in_test_data:]
    out_test_dataset = ood_dataset[-n_out_test_data:]

    n_mixed_train = int(len(train_dataset) * args.biased_ratio)
    n_id_train = n_train_data - n_mixed_train
    train_dataset = ConcatDataset([train_dataset[:n_id_train], mixed_dataset[:n_mixed_train]])

    # we need to modify outliers' idx to track their gradients (for GraphDE-v)
    for i, data in enumerate(train_dataset):
        data.idx += (i-data.idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    id_test_loader = DataLoader(in_test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_test_loader = DataLoader(out_test_dataset, batch_size=args.batch_size, shuffle=False)

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = {'train_acc': [], 'val_acc': [], 'in_test_acc': [], 'out_test_acc': [],
                'base_auroc': [], 'base_aupr': [], 'base_fpr95': [],
                'vi_auroc': [], 'vi_aupr': [], 'vi_fpr95': []}
    experiment_name = f'collab.graphde-a_{args.graphde_a}.graphde-v_{args.graphde_v}.backbone_{args.backbone}.' \
                    f'ood-ratio_{args.biased_ratio}.prior-ratio{args.prior_ratio}.' \
                    f'dropedge_{args.dropedge}.grand_{args.grand}.dropnode_{args.dropnode}.' \
                    f'netlr_{args.net_lr}.dropout_{args.dropout}.batch_{args.batch_size}.channels_{args.channels}.' \
                    f'pretrain_{args.pretrain}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    args_print(args, logger)

    for seed in args.seed:
        set_seed(seed)
        # models and optimizers
        g = CollabNet(in_channels, num_classes, n_train_data=n_train_data, args=args).to(device)
        model_optimizer = torch.optim.Adam([
            {'params': g.gnn_model.parameters()},
            {'params': g.structure_model.parameters()},
            {'params': g.out_mlp.parameters()},
            {'params': g.e_logits}
        ], lr=args.net_lr)

        logger.info(
            f"#Train: {n_train_data} #In Test: {n_in_test_data} #Out Test: {n_out_test_data} #Val: {n_val_data}")
        cnt, last_val_acc, last_train_acc, last_in_test_acc, last_out_test_acc, last_state_dict = 0, 0, 0, 0, 0, None
        for epoch in range(args.epoch):
            all_loss, n_bw = 0, 0
            g.train()
            for graph in train_loader:
                n_bw += 1
                graph.to(device)
                N = graph.num_graphs

                if args.graphde_a:
                    loss = g.get_graphde_a_loss(x=graph.x, edge_index=graph.edge_index,
                                                edge_attr=graph.edge_attr, batch=graph.batch, y=graph.y)
                elif args.graphde_v:
                    loss = g.get_graphde_v_loss(x=graph.x, edge_index=graph.edge_index,
                                                edge_attr=graph.edge_attr, batch=graph.batch, 
                                                idx=graph.idx, y=graph.y)
                elif args.grand:
                    loss = g.get_grand_pred_loss(x=graph.x, edge_index=graph.edge_index,
                                                edge_attr=graph.edge_attr, batch=graph.batch, y=graph.y)
                else:
                    loss = g.get_pred_loss(x=graph.x, edge_index=graph.edge_index,
                                        edge_attr=graph.edge_attr, batch=graph.batch, y=graph.y)
                all_loss += loss
            all_loss /= n_bw

            model_optimizer.zero_grad()
            all_loss.backward()
            model_optimizer.step()

            g.eval()
            with torch.no_grad():
                train_acc = calc_acc(train_loader, g)
                val_acc = calc_acc(val_loader, g)
                in_test_acc = calc_acc(id_test_loader, g)
                out_test_acc = calc_acc(ood_test_loader, g)

                logger.info("Epoch [{:3d}/{:d}]  all_loss:{:.3f} "
                            "Train_ACC:{:.3f} Val_ACC:{:.3f} In_Test_ACC:{:.3f} Out_Test_ACC:{:.3f}".format(
                    epoch, args.epoch, all_loss, train_acc, val_acc, in_test_acc, out_test_acc))

                # activate early stopping
                if epoch >= args.pretrain:
                    if val_acc < last_val_acc:
                        cnt += 1
                    else:
                        cnt = 0
                        last_val_acc = val_acc
                        last_train_acc = train_acc
                        last_in_test_acc = in_test_acc
                        last_out_test_acc = out_test_acc
                        last_state_dict = g.state_dict()
                if cnt >= 30:
                    logger.info("Early Stopping, start OOD detection")
                    break

        with torch.no_grad():
            g.load_state_dict(last_state_dict)
            base_auroc, base_aupr, base_fpr, vi_auroc, vi_aupr, vi_fpr = ood_detection(id_test_loader, ood_test_loader, g,
                                                                                       exp_dir, seed)
            logger.info("Seed {:d} Base AUROC:{:.3f} Base AUPR:{:.3f} Base FPR95:{:.3f} "
                        "VI AUROC:{:.3f} VI AUPR:{:.3f} VI FPR95:{:.3f}".format(
                seed, 100 * base_auroc, 100 * base_aupr, 100 * base_fpr, 100 * vi_auroc, 100 * vi_aupr,
                      100 * vi_fpr))

        all_info['train_acc'].append(last_train_acc)
        all_info['val_acc'].append(last_val_acc)
        all_info['in_test_acc'].append(last_in_test_acc)
        all_info['out_test_acc'].append(last_out_test_acc)
        all_info['base_auroc'].append(base_auroc)
        all_info['base_aupr'].append(base_aupr)
        all_info['base_fpr95'].append(base_fpr)
        all_info['vi_auroc'].append(vi_auroc)
        all_info['vi_aupr'].append(vi_aupr)
        all_info['vi_fpr95'].append(vi_fpr)

        torch.save(last_state_dict, osp.join(exp_dir, 'predictor-%d.pt' % seed))
        logger.info("=" * 100)

    logger.info(
        "Train ACC:{:.4f}-+-{:.4f} Val ACC:{:.4f}-+-{:.4f} In Test ACC:{:.4f}-+-{:.4f} Out Test ACC:{:.4f}-+-{:.4f} "
        "Base AUROC:{:.4f}-+-{:.4f} Base AUPR:{:.4f}-+-{:.4f} Base FPR95:{:.4f}-+-{:.4f} "
        "VI AUROC:{:.4f}-+-{:.4f} VI AUPR:{:.4f}-+-{:.4f} VI FPR95:{:.4f}-+-{:.4f}".format(
            torch.tensor(all_info['train_acc']).mean(), torch.tensor(all_info['train_acc']).std(),
            torch.tensor(all_info['val_acc']).mean(), torch.tensor(all_info['val_acc']).std(),
            torch.tensor(all_info['in_test_acc']).mean(), torch.tensor(all_info['in_test_acc']).std(),
            torch.tensor(all_info['out_test_acc']).mean(), torch.tensor(all_info['out_test_acc']).std(),
            torch.tensor(all_info['base_auroc']).mean(), torch.tensor(all_info['base_auroc']).std(),
            torch.tensor(all_info['base_aupr']).mean(), torch.tensor(all_info['base_aupr']).std(),
            torch.tensor(all_info['base_fpr95']).mean(), torch.tensor(all_info['base_fpr95']).std(),
            torch.tensor(all_info['vi_auroc']).mean(), torch.tensor(all_info['vi_auroc']).std(),
            torch.tensor(all_info['vi_aupr']).mean(), torch.tensor(all_info['vi_aupr']).std(),
            torch.tensor(all_info['vi_fpr95']).mean(), torch.tensor(all_info['vi_fpr95']).std(),
        ))

