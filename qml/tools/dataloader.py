# -*- coding: utf-8 -*-

import numpy as np


class DataLoader:

    def __init__(self, xs, ys, num_batch: int, shuffle=True):
        assert len(xs) == len(ys)
        self._xs = np.asarray(xs)
        self._ys = np.asarray(ys)
        self.size = len(xs)
        self._num_batch = num_batch
        self._shuffle = shuffle
        self._idx = None
        self.iter_count = None

    def __iter__(self):
        self.iter_count = 0
        idx = np.arange(self.size)
        if self._shuffle:
            idx = np.random.permutation(idx)
        self._idx = [
            idx[i * self._num_batch:(i + 1) * self._num_batch]
            for i in range(int(np.ceil(self.size / self._num_batch)))
        ]
        return self

    def __next__(self):
        if self.iter_count >= len(self._idx):
            raise StopIteration
        idx = self._idx[self.iter_count]
        bxs = self._xs[idx]
        bys = self._ys[idx]
        self.iter_count += 1
        return bxs, bys

    def __len__(self):
        return self.size
