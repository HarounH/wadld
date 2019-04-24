#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
from itertools import permutations

from scipy.sparse.csgraph import breadth_first_tree
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=np.inf)

pkl_file = "data/preprocessed_data/binarized.pkl"
out_file = "data/preprocessed_data/permuted.pkl"
data = pickle.load(open(pkl_file, "rb"))

# every map, log(|V|) permutations

perm_data = {'E':[], 'V':[]}
for ct, (pi, coords) in enumerate(zip(data.get('E'), data.get('V'))):
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
        d = np.triu(adj_mat.T[perm[q]][:, perm[q]].T)
        np.fill_diagonal(d, 0)
        perm_data['E'].append(d)
        perm_data['V'].append(coords[bft_perm[q]])
 
pickle.dump(perm_data, open(out_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
