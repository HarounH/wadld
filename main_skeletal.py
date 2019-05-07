'''
main.py for graph generation model
Deals with skeletal structure of maps, i.e., vertex positions and adjacency.
'''


import os
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import argparse
from collections import defaultdict
import torch
from torch import nn
import torch.functional as F
from torch import optim
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
)
from utils.data import (
    PackCollate,
    WaddleDataset,
)
from modules import (
    graph_rnn,
)
from modules.loss import (
    skeletal_losses,
)
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('data_path', help='Path to dataset of ordered vertices and adjacencies')
    parser.add_argument('-t', '--test_frac', default=0.3, help='Fraction of data to use for testing')
    parser.add_argument('-d', '--debug', action='store_true', help='If set, then use debugging mode - dataset consists of a small number of points')

    # Output
    parser.add_argument("--base_output", dest="base_output", default="wadld/outputs/multi_run/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument('--eval_every', dest='eval_every', default=1, help='How often to evaluate model')
    parser.add_argument('--save_every', dest='save_every', default=1, help='How often to save model')

    # Hyperparameters
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa

    # Hardware
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available or not')
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa
    parser.add_argument("--num_layers", dest="num_layers", default=1, type=int, metavar='<int>', help='Number of layers in LSTM model.')
    parser.add_argument("--hidden_size", dest="hidden_size", default=32, type=int, metavar='<int>', help='Number of nodes per layer LSTM.')

    args = parser.parse_args()

    args.num_workers = multiprocessing.cpu_count() // 3
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.dataparallel = args.dataparallel and args.cuda
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.debug:
        args.run_code = "debug"
    os.makedirs(args.base_output, exist_ok=True)
    if len(args.run_code) == 0:
        # Generate a run code by counting number of directories in oututs
        run_count = len(os.listdir(args.base_output))
        args.run_code = 'run{}'.format(run_count)
    args.base_output = os.path.join(args.base_output, args.run_code)
    os.makedirs(args.base_output, exist_ok=True)
    print("Using run_code: {}".format(args.run_code))
    return args


def test(args, dataset, model, loader, prefix='', verbose=True):
    metrics = defaultdict(list)
    replace_field_by_mean = ['eos_loss', 'adj_loss', 'pos_loss', 'loss']
    with torch.no_grad():
        for bidx, (G_t, G_tp1, mask) in enumerate(loader):
            G_t = G_t.to(args.device)
            G_tp1 = G_tp1.to(args.device)

            discrete_hat, continuous_hat, adj_hat = G_tp1_hat = model(G_t)
            eos_loss, adj_loss, pos_loss = skeletal_losses(G_tp1_hat, G_tp1, dataset, mask)
            loss = eos_loss + adj_loss + pos_loss

            metrics['eos_loss'].append(eos_loss.item())
            metrics['adj_loss'].append(adj_loss.item())
            metrics['pos_loss'].append(pos_loss.item())
            metrics['loss'].append(loss.item())

        for k in replace_field_by_mean:
            metrics[k] = np.mean(metrics[k])

        # Print!
        if verbose:
            start_string = '#### {} evaluation ####'.format(prefix)
            print(start_string)
            for k, v in metrics.items():
                print('#### {} = {:.3f}'.format(k, v))
            print(''.join(['#' for _ in range(len(start_string))]))
    return metrics


def train_skeletal_model(args, dataset, train_loader, test_loader):
    output_dir = os.path.join(args.base_output)
    model = graph_rnn.GraphRNN(
        discrete_feature_dim=dataset.discrete_feature_dim,
        continuous_feature_dim=dataset.continuous_feature_dim,
        max_vertex_num=dataset.max_vertex_num,
        rnn_hidden_size=args.hidden_size,
        rnn_num_layers=args.num_layers
    )

    model = model.to(args.device)
    if args.dataparallel:
        raise NotImplementedError('Check if nn.DataParallel works with RNN')

    params = list(model.parameters())
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    metrics = defaultdict(list)
    for epoch_idx in range(args.epochs):
        print('Starting epoch {}'.format(epoch_idx))
        epoch_metrics = defaultdict(list)
        tic = time.time()
        for bidx, (G_t, G_tp1, mask) in enumerate(train_loader):
            G_t = G_t.to(args.device)
            G_tp1 = G_tp1.to(args.device)
            discrete_hat, continuous_hat, adj_hat = G_tp1_hat = model(G_t)
            eos_loss, adj_loss, pos_loss = skeletal_losses(G_tp1_hat, G_tp1, dataset, mask)
            loss = eos_loss + adj_loss + pos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_metrics['eos_loss'].append(eos_loss.item())
            epoch_metrics['adj_loss'].append(adj_loss.item())
            epoch_metrics['pos_loss'].append(pos_loss.item())
            epoch_metrics['loss'].append(loss.item())
        metrics['eos_loss'].append(np.mean(epoch_metrics['eos_loss']))
        metrics['adj_loss'].append(np.mean(epoch_metrics['adj_loss']))
        metrics['pos_loss'].append(np.mean(epoch_metrics['pos_loss']))
        metrics['loss'].append(np.mean(epoch_metrics['loss']))
        print('[{:.2f}s] Epoch {}: losses={:.3f} eos, {:.3f} adj, {:.3f} pos = {:.3f} total'.format(
            time.time() - tic,
            epoch_idx,
            metrics['eos_loss'][epoch_idx],
            metrics['adj_loss'][epoch_idx],
            metrics['pos_loss'][epoch_idx],
            metrics['loss'][epoch_idx],
            ))

        # Eval and save if necessary.
        if utils.periodic_integer_delta(epoch_idx, args.eval_every):
            test_metrics = test(args, dataset, model, test_loader, prefix='Test Dataset, Epoch {}'.format(epoch_idx))
            for k, v in test_metrics.items():
                metrics['test_{}_epoch{}'.format(k, epoch_idx)] = v

        if utils.periodic_integer_delta(epoch_idx, args.save_every):
            checkpoint_path = os.path.join(output_dir, "last.checkpoint")
            print('Saving model to {}'.format(checkpoint_path))
            chk = utils.make_checkpoint(model, optimizer, epoch_idx)
            chk['args'] = vars(args)
            torch.save(chk, checkpoint_path)
    return model, metrics


def get_dataloaders(args, dataset, batch_size, test_frac):
    indices = np.random.permutation(list(range(len(dataset))))
    split_point = int(args.test_frac * len(indices))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[split_point:]),
        collate_fn=PackCollate(),
        num_workers=multiprocessing.cpu_count() // 4)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[:split_point]),
        collate_fn=PackCollate(),
        num_workers=multiprocessing.cpu_count() // 4)
    return train_loader, test_loader


def run_skeletal(args):
    tic = time.time()
    dataset = WaddleDataset(args.data_path, return_mask=True)
    print('[{:.2f}] Created dataset'.format(time.time() - tic))
    if args.debug:
        dataset.n = 40
    train_loader, test_loader = get_dataloaders(args, dataset, args.batch_size, args.test_frac)  # noqa
    model, metrics = train_skeletal_model(args, dataset, train_loader, test_loader)
    return model, metrics

if __name__ == '__main__':
    args = get_args()
    run_skeletal(args)
