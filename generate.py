import torch
import numpy as np
from modules import graph_rnn
from utils.data import WaddleDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dataset = WaddleDataset("data/preprocessed_data/permuted.pkl")
num_feat = dataset.discrete_feature_dim + dataset.continuous_feature_dim \
        + dataset.max_vertex_num 
# Why doesn't this work?
#G_t = torch.tensor((), dtype=torch.float, device=device)
#G_t.new_zeros((num_feat, num_feat))

model = graph_rnn.GraphRNN(
    discrete_feature_dim=dataset.discrete_feature_dim,
    continuous_feature_dim=dataset.continuous_feature_dim,
    max_vertex_num=dataset.max_vertex_num,
)

model.cuda()
checkpoint_path = "wadld/outputs/multi_run/run2/last.checkpoint"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()
max_vertices = 50

G_t_mat = np.zeros((1, 1, num_feat))
G_t = torch.tensor(G_t_mat, dtype=torch.float, device=device)
G_t = G_t
hidden = torch.rand((1, 1, 32), dtype=torch.float, device = device)
cell = torch.rand((1, 1, 32), dtype=torch.float, device = device)

with torch.no_grad():
    for i in range(max_vertices):
        discrete_features, continuous_features, adjacencies = \
                model(G_t, (hidden, cell))
        hidden, cell = model.hidden
        

