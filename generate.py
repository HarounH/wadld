import torch
from modules import graph_rnn
from utils.data import WaddleDataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dataset = WaddleDataset("data/preprocessed_data/permuted.pkl")
num_feat = dataset.discrete_feature_dim + dataset.continuous_feature_dim \
        + dataset.max_vertex_num 

model = graph_rnn.GraphRNN(
    discrete_feature_dim=dataset.discrete_feature_dim,
    continuous_feature_dim=dataset.continuous_feature_dim,
    max_vertex_num=dataset.max_vertex_num,
)

checkpoint_path = "wadld/outputs/multi_run/run2/last.checkpoint"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()
max_vertices = 50

hidden = torch.randint(32, (1, 1), dtype=torch.long).to(device)

with torch.no_grad():
    for i in range(max_vertices):
        discrete_features, continuous_features, adjacencies = model(G_t, hidden)
        hidden = model.hidden
        

