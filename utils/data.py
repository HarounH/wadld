import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import pickle
# np.set_printoptions(threshold=np.inf)


class WaddleDataset(Dataset):
    '''Waddle dataset.'''

    def __init__(self, pkl_file, min_number_of_nodes=10, standardize_positions=True, return_mask=False):
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

            if d < min_number_of_nodes:
                continue

            padded = np.zeros((mx, d))
            padded[:d, :] = self.data['E'][idx]

            eos = np.zeros(d)

            # stack coordinates on top
            # stack a vector [0, ..., 1] on top of that
            points = self.data['V'][idx]
            if standardize_positions:
                points = (points - points.mean(0)) / (points.std(0) + 1e-8)
            seq = np.vstack((eos, points.T, padded))  # (3 + mx, d)
            self.preprocessed.append(seq.astype(np.float32))
        self.n = len(self.preprocessed)

        assert self.n > 0, "No datapoints."

        # Use shape[0] because preprocessed arrays are transposed.
        self.end_token = np.zeros((1, self.preprocessed[0].shape[0]), dtype=np.float32)
        self.end_token[0, 0] = 1.0
        self.return_mask = return_mask

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        inp = self.preprocessed[idx].T  # (d, 3 + mx)
        out = np.concatenate([
            inp[1:, :],  # d - 1, 3 + mx
            self.end_token  # 1, 3 + mx
        ], axis=0)  # d, 3 + mx
        if self.return_mask:
            mask = np.tril(
                np.ones(
                    (inp.shape[0],  # d
                    inp.shape[1] - (self.discrete_feature_dim + self.continuous_feature_dim))  # mx
                    )
                )
            return inp, out, mask
        return inp, out


class PackCollate:
    ''' Waddle dataloader.'''

    def __call__(self, batch):
        '''For a given batch we pack the sequences of adjacencies and get there lengths. '''
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        rets = []
        for i in range(len(sorted_batch[0])):
            adj_seqs = [torch.from_numpy(x[i]) for x in sorted_batch]
            lengths = torch.LongTensor([len(x) for x in adj_seqs])
            adj_seqs_packed = pack_sequence(adj_seqs)
            rets.append(adj_seqs_packed)
        return rets


if __name__ == "__main__":
    ds = WaddleDataset("data/preprocessed_data/binarized.pkl")
    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=PackCollate())
    for adj, lengths in dl:
        print("{}\t{}".format(type(adj), lengths))
        break
