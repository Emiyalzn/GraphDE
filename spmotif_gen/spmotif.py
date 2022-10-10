from BA3_loc import *
from tqdm import tqdm
import os.path as osp
import os
import warnings
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SPMotif dataset")
    parser.add_argument("--mode", default='train', type=str, help='dataset split')
    parser.add_argument("--bias", default=0.9, type=float, help='base bias ratio')
    parser.add_argument("--type", choices=['small', 'big'], default='small', type=str, help='choose big or small base type')
    parser.add_argument("--num", default=3000, type=int, help='number of graphs per motif')
    args = parser.parse_args()
    return args

def get_house(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:
    Start with a tree and attach HOUSE-shaped subgraphs.
    """
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def get_cycle(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:
    Start with a tree and attach cycle-shaped (directed edges) subgraphs.
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:
    Start with a tree and attach crane-shaped subgraphs.
    """
    list_shapes = [["crane"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def gen_small(mode, bias, num):
    edge_index_list, label_list = [], []
    ground_truth_list, role_id_list, pos_list = [], [], []

    def graph_stats(base_num):
        if base_num == 1:
            base = 'tree'
            # base = 'ba'
            width_basis = np.random.choice(range(3))
            # width_basis = np.random.choice(range(6, 8))
        if base_num == 2:
            base = 'ladder'
            # base = 'ba'
            width_basis = np.random.choice(range(8, 12))
        if base_num == 3:
            base = 'wheel'
            # base = 'ba'
            width_basis = np.random.choice(range(15, 20))
        return base, width_basis

    e_mean, n_mean = [], []
    for _ in tqdm(range(num)):
        base_num = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])
        # base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, bias, (1 - bias) / 2])
        # base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])
        base, width_basis = graph_stats(base_num)

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                     width_basis=width_basis, feature_generator=None, m=3, draw=False)
        label_list.append(0)
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=np.int).T

        role_id_list.append(role_id)
        edge_index_list.append(edge_index)
        pos_list.append(np.array(list(nx.spring_layout(G).values())))
        ground_truth_list.append(find_gd(edge_index, role_id))

    print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

    e_mean, n_mean = [], []
    for _ in tqdm(range(num)):
        base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, bias, (1 - bias) / 2])
        # base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])
        # base_num = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])
        base, width_basis = graph_stats(base_num)

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                     width_basis=width_basis, feature_generator=None, m=3, draw=False)
        label_list.append(1)
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=np.int).T

        role_id_list.append(role_id)
        edge_index_list.append(edge_index)
        pos_list.append(np.array(list(nx.spring_layout(G).values())))
        ground_truth_list.append(find_gd(edge_index, role_id))

    print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

    e_mean, n_mean = [], []
    for _ in tqdm(range(num)):
        base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])
        # base_num = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])
        # base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, bias, (1 - bias) / 2])
        base, width_basis = graph_stats(base_num)

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                     width_basis=width_basis, feature_generator=None, m=3, draw=False)
        label_list.append(2)
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=np.int).T

        role_id_list.append(role_id)
        edge_index_list.append(edge_index)
        pos_list.append(np.array(list(nx.spring_layout(G).values())))
        ground_truth_list.append(find_gd(edge_index, role_id))

    print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))
    np.save(osp.join(data_dir, f'{mode}.npy'), (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))

def gen_big(mode, bias, num):
    # no bias for test dataset
    edge_index_list, label_list = [], []
    ground_truth_list, role_id_list, pos_list = [], [], []

    def graph_stats_large(base_num):
        if base_num == 1:
            base = 'tree'
            width_basis = np.random.choice(range(3, 6))
        if base_num == 2:
            base = 'ladder'
            width_basis = np.random.choice(range(30, 50))
        if base_num == 3:
            base = 'wheel'
            width_basis = np.random.choice(range(60, 80))
        return base, width_basis

    e_mean, n_mean = [], []
    for _ in tqdm(range(num)):
        base_num = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])
        base, width_basis = graph_stats_large(base_num)

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                     width_basis=width_basis, feature_generator=None, m=3, draw=False)
        label_list.append(0)
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=np.int).T

        role_id_list.append(role_id)
        edge_index_list.append(edge_index)
        pos_list.append(np.array(list(nx.spring_layout(G).values())))
        ground_truth_list.append(find_gd(edge_index, role_id))

    print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

    e_mean, n_mean = [], []
    for _ in tqdm(range(num)):
        base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, bias, (1 - bias) / 2])
        base, width_basis = graph_stats_large(base_num)

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                     width_basis=width_basis, feature_generator=None, m=3, draw=False)
        label_list.append(1)
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=np.int).T

        role_id_list.append(role_id)
        edge_index_list.append(edge_index)
        pos_list.append(np.array(list(nx.spring_layout(G).values())))
        ground_truth_list.append(find_gd(edge_index, role_id))

    print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

    e_mean, n_mean = [], []
    for _ in tqdm(range(num)):
        base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])
        base, width_basis = graph_stats_large(base_num)

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                     width_basis=width_basis, feature_generator=None, m=3, draw=False)
        label_list.append(2)
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))

        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=np.int).T

        role_id_list.append(role_id)
        edge_index_list.append(edge_index)
        pos_list.append(np.array(list(nx.spring_layout(G).values())))
        ground_truth_list.append(find_gd(edge_index, role_id))

    print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))
    np.save(osp.join(data_dir, f'{mode}.npy'), (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    data_dir = f'../data/SPMotif-{args.bias}/raw/'
    os.makedirs(data_dir, exist_ok=True)
    if args.type == 'small':
        gen_small(args.mode, args.bias, args.num)
    else:
        gen_big(args.mode, args.bias, args.num)
