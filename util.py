import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    tag2index = None
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

    def add_noise(self, adj_noise, tag):
        node_id = len(self.g)
        self.g.add_node(node_id)

        # add neighbors
        self.neighbors.append([])
        connections = (x for x in range(0, len(adj_noise)) if adj_noise[x]>0)
        degree_list = []
        for n in connections:
            self.g.add_edge(node_id, n)
            self.neighbors[node_id].append(n)
            self.neighbors[n].append(node_id)

        edges = [list(pair) for pair in self.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        self.edge_mat = torch.LongTensor(edges).transpose(0,1)

        for i in range(len(self.g)):
            degree_list.append(len(self.neighbors[i]))
        self.max_neighbor = max(degree_list)
        self.node_tags.append(tag)

        if S2VGraph.tag2index:
            self.node_features = torch.zeros(len(self.node_tags), len(S2VGraph.tag2index))
            self.node_features[range(len(self.node_tags)), [S2VGraph.tag2index[tag] for tag in self.node_tags]] = 1
        else:
            print('Error. Fail to find Tag2index')

        return self

    def add_single_edge_noise(self, target_id, tag):
        node_id = len(self.g)
        self.g.add_node(node_id)

        # add neighbors
        self.neighbors.append([])
        degree_list = []
        
        self.g.add_edge(node_id, target_id)
        self.neighbors[node_id].append(target_id)
        self.neighbors[target_id].append(node_id)

        edges = [list(pair) for pair in self.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        self.edge_mat = torch.LongTensor(edges).transpose(0,1)

        
        self.node_tags.append(tag)

        if S2VGraph.tag2index:
            self.node_features = torch.zeros(len(self.node_tags), len(S2VGraph.tag2index))
            self.node_features[range(len(self.node_tags)), [S2VGraph.tag2index[tag] for tag in self.node_tags]] = 1
        else:
            print('Error. Fail to find Tag2index')

        return self




def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    tag_count = []

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                    tag_count.append(0)
                node_tags.append(feat_dict[row[0]])
                tag_count[feat_dict[row[0]]] += 1


                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    S2VGraph.tag2index = tag2index

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print('Tag Set:'+str(tagset))
    assert len(tag_count) == len(tagset)

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), tag_count

def write_data(args, graph_list, graph_list_test):
    with open('dataset/%s/%s.txt' % ('poison', args.dataset+args.write_data), 'w') as f:
        f.write(str(len(graph_list)))
        for graph in graph_list:
            f.write("\n %d %d" % (len(graph.g), graph.label))
            for i in range(len(graph.g)): 
                string_ints = [str(int) for int in graph.neighbors[i]]
                f.write("\n %d %d %s" % (graph.node_tags[i], len(graph.neighbors[i]),
                                            " ".join(string_ints)))

        f.write('\n')

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


