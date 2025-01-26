import torch
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def normal_std(x):
#      return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, horizon, window):
        self.h = horizon
        self.P = window
        text = open(file_name)
        self.rawdat = np.loadtxt(text, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape# n,m denote the time-point,variables' number
        self.scale = np.ones(self.m)
        self.scale_min = np.ones(self.m)
        self._normalized()
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        self.scale_min = torch.from_numpy(self.scale_min).float()
        # tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m) #？？？

        self.scale = self.scale.to(device)
        self.scale_min = self.scale_min.to(device)


    def _normalized(self):

        for i in range(self.m):
            self.scale[i] = np.max(self.rawdat[:, i])
            self.scale_min[i] = np.min(self.rawdat[:, i])
            self.dat[:, i] = (self.rawdat[:, i] - self.scale_min[i]) / (self.scale[i] - self.scale_min[i])

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)#h,P
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield X, Y
            start_idx += batch_size