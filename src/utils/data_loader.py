from os.path import join
import torch
import sklearn
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from functools import partial

from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class G_data(object):
    def __init__(self, num_class, feat_dim, g_list):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.g_list = g_list
        self.num_graphs = len(g_list)
        # self.sep_data()

    def sep_data(self, seed=0):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        labels = [g.label for g in self.g_list]
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx+1
        train_idx, test_idx = self.idx_list[fold_idx]
        self.train_gs = [self.g_list[i] for i in train_idx]
        self.test_gs = [self.g_list[i] for i in test_idx]

    def split_data(self, train_ratio=0.5, valid_ratio=0.25):
        n_train = int(self.num_graphs*train_ratio)
        n_valid = int((train_ratio+valid_ratio)*self.num_graphs)
        self.train_gs = [self.g_list[i] for i in range(0, n_train)]
        self.valid_gs = [self.g_list[i] for i in range(n_train, n_train+n_valid)]
        self.test_gs = [self.g_list[i] for i in range(n_train+n_valid, self.num_graphs)]


class FileLoader(object):
    def __init__(self, args):
        self.args = args

    def line_genor(self, lines):
        for line in lines:
            yield line

    def gen_graph(self, f, i, label_dict, feat_dict, deg_as_tag):
        row = next(f).strip().split()
        # print("row", row)
        n, label = [int(w) for w in row]
        # print("n & label", n, label)
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        g = nx.Graph()
        g.add_nodes_from(list(range(n)))
        node_tags = []
        for j in range(n):
            row = next(f).strip().split()
            tmp = int(row[1]) + 2
            row = [int(w) for w in row[:tmp]]
            if row[0] not in feat_dict:
                feat_dict[row[0]] = len(feat_dict)
            for k in range(2, len(row)):
                if j != row[k]:
                    g.add_edge(j, row[k])
            if len(row) > 2:
                node_tags.append(feat_dict[row[0]])
        g.label = label
        g.remove_nodes_from(list(nx.isolates(g)))
        if deg_as_tag:
            g.node_tags = list(dict(g.degree).values())
        else:
            g.node_tags = node_tags
        return g

    def process_g(self, label_dict, tag2index, tagset, g):
        g.label = label_dict[g.label]
        g.feas = torch.tensor([tag2index[tag] for tag in g.node_tags])
        g.feas = F.one_hot(g.feas, len(tagset))
        A = torch.FloatTensor(nx.to_numpy_matrix(g))
        g.A = A + torch.eye(g.number_of_nodes())
        return g

    def load_data(self):
        args = self.args
        print('loading data ...')
        g_list = []
        label_dict = {}
        feat_dict = {}

        with open('../data/%s/%s.txt' % (args.data, args.data), 'r') as f:
            lines = f.readlines()
        f = self.line_genor(lines)
        n_g = int(next(f).strip())
        for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
            g = self.gen_graph(f, i, label_dict, feat_dict, args.deg_as_tag)
            g_list.append(g)

        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}

        f_n = partial(self.process_g, label_dict, tag2index, tagset)
        new_g_list = []
        for g in tqdm(g_list, desc="Process graph", unit='graphs'):
            new_g_list.append(f_n(g))
        num_class = len(label_dict)
        feat_dim = len(tagset)

        print('# classes: %d' % num_class, '# maximum node tag: %d' % feat_dim)
        return G_data(num_class, feat_dim, new_g_list)


class FileLoaderNew(object):
    def __init__(self, args):
        self.args = args

    def gen_graph(self, adj, inf_features, label, cur_node_features):
        # g = nx.Graph()
        # g.add_nodes_from(list(range(len(cur_vids))))
        g = nx.from_numpy_array(adj)
        node_tags = np.concatenate((cur_node_features, inf_features), axis=1)  #todo
        g.label = label
        g.remove_nodes_from(list(nx.isolates(g)))
        g.node_tags = node_tags
        return g

    def process_g(self, g):
        g.feas = torch.FloatTensor(g.node_tags)
        A = torch.FloatTensor(nx.to_numpy_matrix(g))
        g.A = A + torch.eye(g.number_of_nodes())
        return g

    def load_data(self):
        args = self.args
        file_dir = join(settings.DATA_DIR, args.data)
        print('loading data ...')

        graphs = np.load(join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # wheather a user has been influenced
        # wheather he/she is the ego user
        influence_features = np.load(
                join(file_dir, "influence_feature.npy")).astype(np.float32)
        logger.info("influence features loaded!")

        labels = np.load(join(file_dir, "label.npy"))
        logger.info("labels loaded!")

        vertices = np.load(join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        vertex_features = np.load(join(file_dir, "vertex_feature.npy"))
        vertex_features = preprocessing.scale(vertex_features)
        # vertex_features = torch.FloatTensor(vertex_features)
        logger.info("global vertex features loaded!")

        n_g = len(graphs)

        g_list = []
        label_dict = {0: 0, 1: 1}

        for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
            cur_vids = vertices[i]
            cur_node_features = vertex_features[cur_vids]
            g = self.gen_graph(graphs[i], influence_features[i], labels[i], cur_node_features)
            g_list.append(g)

        new_g_list = []
        for g in tqdm(g_list, desc="Process graph", unit='graphs'):
            new_g_list.append(self.process_g(g))

        new_g_list = sklearn.utils.shuffle(new_g_list, random_state=args.seed)

        num_class = 2
        feat_dim = new_g_list[0].feas.shape[1]

        print('# classes: %d' % num_class, '# maximum node tag: %d' % feat_dim)
        return G_data(num_class, feat_dim, new_g_list)
