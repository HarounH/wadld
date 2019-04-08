import torch
from torch import nn
import torch.functional as F


bce_logits_fn = nn.BCEWithLogitsLoss(reduction='none')
mse_loss_fn = nn.MSELoss(reduction='none')
def skeletal_losses(prediction, target, dataset):
    eos_hat, continuous_hat, adj_hat = prediction
    eos = target.data[:, 0].view(-1, 1)
    eos_loss = bce_logits_fn(
        eos_hat.data,
        target.data[:, :dataset.discrete_feature_dim],
    )
    adj_loss = (1 - eos) * bce_logits_fn(
        adj_hat.data,
        target.data[:, (dataset.continuous_feature_dim + dataset.discrete_feature_dim):],
    )
    pos_loss = (1 - eos) * mse_loss_fn(
        continuous_hat.data,
        target.data[:, dataset.discrete_feature_dim:dataset.continuous_feature_dim],
    )
    return eos_loss.mean(), pos_loss.mean(), adj_loss.mean()
