
# Check out what the map data we're training on looks like
import numpy as np
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
)
from utils.data import (
    PackCollate,
    WaddleDataset,
)

data="data/preprocessed_data/permuted.pkl"

dataset = WaddleDataset(data)

indices = np.random.permutation(list(range(len(dataset))))
loader = DataLoader(
        dataset,
        batch_size=1,
        #sampler=SubsetRandomSampler(indices[split_point:]),
        collate_fn=PackCollate())

import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

for i, (adj, lengths) in enumerate(loader):
    D = adj.data.numpy()

    coords = D[1:3, :]*10000
    coords = coords.astype(int)

    adj_mat = D[3:, :]
    connections = np.where(adj_mat>0)
    edges = [(s, d) for s, d in zip(connections[0], connections[1])]

    graph = nx.Graph()
    for node, coord in enumerate(coords.T):
        graph.add_node(node, pos=(coord[0], coord[1]))
    graph.add_edges_from(edges)
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_size=50, node_color='blue', font_size=8, font_weight='bold')

    graph_name = "pngs/training-data/"+str(len(coords.T))+"nodes____"+str(len(edges))+"edges.png"
    print("saving " + graph_name)
    plt.savefig(graph_name, format="PNG")

