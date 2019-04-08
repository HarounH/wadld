import torch
from torch import nn
import torch.functional as F


def skeletal_losses(prediction, target, dataset):
    eos_hat, continuous_hat, adj_hat = prediction
    eos = target[:, 0].view(-1, 1)
    eos_loss = F.binary_cross_entropy_with_logits(
        discrete_hat,
        target[:, :dataset.discrete_feature_dim],
        reduction='none',
    )
    adj_loss = eos * F.binary_cross_entropy_with_logits(
        adj_loss,
        target[:, (dataset.continuous_feature_dim + dataset.discrete_feature_dim):],
        reduction='none',
    )
    pos_loss = eos * F.mse_loss(
        discrete_hat,
        target[:, dataset.discrete_feature_dim:dataset.continuous_feature_dim],
        reduction='none',
    )
    return eos_loss, pos_loss, adj_loss
