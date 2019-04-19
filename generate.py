import torch
from modules import graph_rnn
from utils.data import WaddleDataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dataset = WaddleDataset("data/preprocessed_data/permuted.pkl")

model = graph_rnn.GraphRNN(
    discrete_feature_dim=dataset.discrete_feature_dim,
    continuous_feature_dim=dataset.continuous_feature_dim,
    max_vertex_num=dataset.max_vertex_num,
)

checkpoint_path = "wadld/outputs/multi_run/last.checkpoint"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


