import os
import argparse
import torch
import numpy as np
from torch.nn.utils import rnn
from torch.distributions import Bernoulli
from modules import graph_rnn
from utils.data import WaddleDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dataset = WaddleDataset("data/preprocessed_data/permute.pkl")
num_feat = dataset.discrete_feature_dim + dataset.continuous_feature_dim \
        + dataset.max_vertex_num
# Why doesn't this work?
#G_t = torch.tensor((), dtype=torch.float, device=device)
#G_t.new_zeros((num_feat, num_feat))


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run', default=-1, help='Run ID to use')
args = parser.parse_args()
if args.run == -1:
    run = 11
else:
    run = args.run

checkpoint_path = "wadld/outputs/multi_run/run"+ str(run) + "/last.checkpoint"
checkpoint = torch.load(checkpoint_path)

if 'args' in checkpoint.keys():
    rnn_hidden_size = checkpoint['args']['hidden_size']
    rnn_num_layers = checkpoint['args']['num_layers']
else:
    rnn_hidden_size = 64
    rnn_num_layers = 3

model = graph_rnn.GraphRNN(
    discrete_feature_dim=dataset.discrete_feature_dim,
    continuous_feature_dim=dataset.continuous_feature_dim,
    max_vertex_num=dataset.max_vertex_num,
    rnn_hidden_size=rnn_hidden_size,
    rnn_num_layers=rnn_num_layers,
)
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()
max_vertices = 50

G_t_mat = np.zeros((1, num_feat))
G_t = torch.tensor(G_t_mat, dtype=torch.float, device=device)
G_t = rnn.PackedSequence(G_t, torch.tensor([1], dtype=torch.long))

hidden = torch.rand((1, 1, 32), dtype=torch.float, device = device)
cell = torch.rand((1, 1, 32), dtype=torch.float, device = device)

graph = [G_t.data.cpu().numpy()]

temperature = 1.0
adj_offset = dataset.discrete_feature_dim + dataset.continuous_feature_dim

def predict_edges(G_t, adjacencies, prev_i):
    edge_weights = adjacencies.data.squeeze().sigmoid()
    print(edge_weights)
    edge_preds = []
    m = Bernoulli(edge_weights)
    preds = m.sample()
    G_t[:, :prev_i+1] = torch.narrow(preds, 0, 0, prev_i+1)
    return G_t


with torch.no_grad():
    for i in range(max_vertices):
        discrete_features, continuous_features, adjacencies, (hidden, cell) = \
                model(G_t, (hidden, cell))
        G_t = torch.zeros((1, num_feat), device=device)
        G_t = predict_edges(G_t, adjacencies, i)
        #G_t[:, 1:3] = continuous_features.data.squeeze().sigmoid()
        #G_t[:, 1:3] = continuous_features.data.squeeze().tanh()
        G_t[:, 1:3] = continuous_features.data.squeeze()
        graph.append(G_t.cpu().numpy())
        eos_prob = Bernoulli(discrete_features.data.squeeze().sigmoid())
        eos = eos_prob.sample()

        if eos == 1:
            break

        G_t = rnn.PackedSequence(G_t, torch.tensor([1], dtype=torch.long))

D = np.concatenate(graph, axis=0).T

coords = D[1:3, :]*10000
coords = coords.astype(int)

import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if False:
    D = np.zeros((6,3))
    D[3, 1] = 1
    D[3, 2] = 1
    D[4, 2] = 1

    coords = [np.array([0, 0]), np.array([10, 10]), np.array([10, 0])]

adj_mat = D[3:, :]
connections = np.where(adj_mat>0)
edges = [(s, d) for s, d in zip(connections[0], connections[1])]

graph = nx.Graph()
for node, coord in enumerate(coords.T):
    graph.add_node(node, pos=(coord[0], coord[1]))

graph.add_edges_from(edges)
pos = nx.get_node_attributes(graph, 'pos')
nx.draw(graph, pos, node_size=50, node_color='blue', font_size=8, font_weight='bold')
graph_name = "pngs/generated_run{}/".format(run) + str(len(coords.T)) + "nodes____" + str(len(edges)) + "edges.png"
os.makedirs(os.path.dirname(graph_name), exist_ok=True)
plt.savefig(graph_name, format="PNG")
