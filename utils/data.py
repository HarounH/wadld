import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import pickle
np.set_printoptions(threshold=np.inf)


class WaddleDataset(Dataset):
    '''Waddle dataset.'''

    def __init__(self, pkl_file):
        '''
        @param: pkl_file: the path to the binarized waddle data.
        '''
        self.data = pickle.load(open(pkl_file, "rb"))
        # what's the max node?
        self.max_vertex_num = mx = max([len(x) for x in  self.data['E']])

        self.discrete_feature_dim = 1  # End of Sequence flag

        self.continuous_feature_dim = self.data['V'][0].shape[1]  # X and Y

        self.preprocessed = []

        for idx in range(len(self.data['E'])):
            d = len(self.data['E'][idx])

            padded = np.zeros((mx, d))
            padded[:d, :] = self.data['E'][idx]

            eos = np.zeros(d)
            eos[-1] = 1

            # stack coordinates on top
            # stack a vector [0, ..., 1] on top of that
            seq = np.vstack((eos, self.data['V'][idx].T, padded))
            self.preprocessed.append(seq)
        self.n = len(self.preprocessed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.preprocessed[idx], self.preprocessed[idx]


class PackCollate:
    ''' Waddle dataloader.'''

    def __call__(self, batch):
        '''For a given batch we pack the sequences of adjacencies and get there lengths. '''
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        adj_seqs = [torch.from_numpy(x[1]) for x in sorted_batch]
        lengths = torch.LongTensor([len(x) for x in adj_seqs])
        adj_seqs_packed = pack_sequence(adj_seqs)

        return adj_seqs_packed, lengths


if __name__ == "__main__":
    ds = WaddleDataset("data/preprocessed_data/binarized.pkl")
    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=PackCollate())
    for adj, lengths in dl:
        print("{}\t{}".format(type(adj), lengths))
        break
