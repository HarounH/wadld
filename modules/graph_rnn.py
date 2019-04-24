from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import (
    weight_norm,
    spectral_norm,
)
from torch.nn.utils import (
    rnn,
)

class GraphRNN(nn.Module):
    def __init__(
            self,
            rnn_constructor=nn.LSTM,
            discrete_feature_dim=1,
            continuous_feature_dim=2,
            max_vertex_num=400,
            rnn_hidden_size=32,
            rnn_num_layers=1,
            batch_first=True,
            rnn_dropout=0,
            ):
        super().__init__()
        self.batch_first = batch_first
        self.discrete_feature_dim = discrete_feature_dim
        self.continuous_feature_dim = continuous_feature_dim
        self.max_vertex_num = max_vertex_num
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout

        self.theta_net = rnn_constructor(
            discrete_feature_dim + continuous_feature_dim + max_vertex_num,
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=batch_first,
            dropout=rnn_dropout,
        )

        self.discrete_feature_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.Tanh(),
            nn.Linear(rnn_hidden_size, discrete_feature_dim),
        )

        self.continuous_feature_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.Tanh(),
            nn.Linear(rnn_hidden_size, continuous_feature_dim),
        )

        self.adjacency_feature_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.Tanh(),
            nn.Linear(rnn_hidden_size, max_vertex_num),
        )
   

    def forward(self, G_t, ret_hid=False, *args):
        h, hidden = self.theta_net.forward(G_t, *args)
        discrete_features = rnn.PackedSequence(
            self.discrete_feature_net(h.data),
            h.batch_sizes
        )
        continuous_features = rnn.PackedSequence(
            self.continuous_feature_net(h.data),
            h.batch_sizes
        )
        adjacencies = rnn.PackedSequence(
            self.adjacency_feature_net(h.data),
            h.batch_sizes
        )
        if ret_hid:
            return discrete_features, continuous_features, adjacencies, hidden
        return discrete_features, continuous_features, adjacencies

