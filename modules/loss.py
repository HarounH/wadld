import torch
from torch import nn
import torch.nn.functional as F


bce_logits_fn = nn.BCEWithLogitsLoss(reduction='none')
mse_loss_fn = nn.MSELoss(reduction='none')



def skeletal_losses(prediction, target, dataset, mask, adj_weights=None, only_edges=False):
    if only_edges:
        eos_hat, adj_hat = prediction
    else:
        eos_hat, continuous_hat, adj_hat = prediction
    eos = target.data[:, 0].view(-1, 1)
    eos_loss = F.binary_cross_entropy_with_logits(
        eos_hat.data,
        target.data[:, :dataset.discrete_feature_dim],
    )

    adj_loss = F.binary_cross_entropy_with_logits(
        adj_hat.data[mask.byte().data],
        target.data[:, (dataset.continuous_feature_dim + dataset.discrete_feature_dim):][mask.byte().data],
        pos_weight=adj_weights,
    )
    if not only_edges:
        pos_loss = (1 - eos) * mse_loss_fn(
            continuous_hat.data,
            target.data[:, dataset.discrete_feature_dim:dataset.continuous_feature_dim],
        )
        return eos_loss.mean(), adj_loss.mean(), pos_loss.mean()
    else:
        return eos_loss.mean(), adj_loss.mean()
