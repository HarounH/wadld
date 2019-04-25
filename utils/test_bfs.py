#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
from itertools import permutations

from scipy.sparse.csgraph import breadth_first_tree
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=np.inf)

pkl_file = "data/preprocessed_data/binarized.pkl"
test_file = "data/preprocessed_data/test.pkl"
out_file = "data/preprocessed_data/test_out.pkl"

import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
)

from data import (
    PackCollate,
    WaddleDataset,
)

# every map, log(|V|) permutations
def render(adj, graph_name):
    D = adj.data.numpy().T
    coords = D[1:3, :]

    coords = coords.astype(int)

    adj_mat = D[3:, :]
    connections = np.where(adj_mat>0)
    edges = [(s, d) for s, d in zip(connections[0], connections[1])]
    graph = nx.Graph()
    for node, coord in enumerate(coords.T):
        graph.add_node(node, pos=(coord[0], coord[1]))
    graph.add_edges_from(edges)
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_size=1, node_color='blue', font_size=8, font_weight='bold')

    print("saving " + graph_name)
    plt.savefig("pngs/debug/"+graph_name, format="PNG")

perm_data = {'E':[], 'V':[]}
iter_used = None

if True:
    D = np.zeros((5,5))
    D[0, 1] = 1
    D[0, 2] = 1
    D[1, 2] = 1
    D[0, 3] = 1
    D[1, 3] = 1
    D[0, 4] = 1
    D[2, 4] = 1
    coords = np.array([np.array([0, 0]), np.array([10, 10]), np.array([10, 0]),
        np.array([0, 5]), np.array([0, -5])])
    print(coords)
    test_data = {'E':[D], 'V':[coords]}

    pickle.dump(test_data, open(test_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    data = pickle.load(open(test_file, "rb"))

#data = pickle.load(open(pkl_file, "rb"))
for ct, (pi, coords) in enumerate(zip(data.get('E'), data.get('V'))):
    iter_used = ct
    if ct % 1000 == 0:
        print("Created {} map permutations".format(ct))
    num_perms = np.log(len(coords))

    M = pi.T + pi
    adj_mat = np.zeros(M.shape)
    adj_mat[np.where(M>0)] = 1
    dims = [i for i in range(adj_mat.shape[0])]
    perms = set()

    while len(perms) < num_perms:
        perms.add(np.random.permutation(dims).tostring())

    bfts = []    
    for perm in perms:
        perm = np.frombuffer(perm, dtype=np.int)
        p_adj_mat = adj_mat[list(perm)][:, list(perm)]
        bft_mat = np.zeros(p_adj_mat.shape)
        for idx in dims:
            if np.sum(bft_mat[:, idx])== 0:
                bft_mat += breadth_first_tree(p_adj_mat, idx, directed=False).toarray()
        bfts.append((bft_mat, perm))

    n = [0]
    q = [0]

    for bft in bfts:
        bft_mat = bft[0]
        bft_perm = bft[1]
        while len(n) > 0:
            v = n.pop(0)
            children = np.nonzero(bft_mat.astype(int).T[:, v])[0]
            
            if not children.tolist() and len(q) < bft_mat.shape[0] and not n:
                q_set = set(q)
                children = [[idx for idx in range(bft_mat.shape[0]) if idx not in q_set][0]]
                
            n.extend(children)
            q.extend(children)
        print(adj_mat.T[perm[q]][:, perm[q]].T)
        d = np.triu(adj_mat.T[bft_perm[q]][:, bft_perm[q]].T)
        np.fill_diagonal(d, 0)
        perm_data['E'].append(d)
        perm_data['V'].append(coords[bft_perm[q]])
        print(d)
        print(coords[bft_perm[q]])
    break

dataset = WaddleDataset(test_file, min_number_of_nodes=1, standardize_positions=False)
#dataset = WaddleDataset(pkl_file, standardize_positions=False)

loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=PackCollate())

for i, (adj, lengths) in enumerate(loader):
    if i == iter_used: 
        render(adj, "orig")
        break


pickle.dump(perm_data, open(out_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

dataset = WaddleDataset(out_file, min_number_of_nodes=1, standardize_positions=False)

loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=PackCollate())

for i, (adj, lengths) in enumerate(loader):
    render(adj, str(i))

