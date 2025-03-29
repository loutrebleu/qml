# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray

from .random import XRandomGenerator
from .dataset import Dataset



class DLIter:

    def __init__(self, xs: NDArray, ys: NDArray, indices: NDArray):
        self._xs = xs
        self._ys = ys
        self._indices = indices
        self._iter_counter = 0
    
    def __next__(self):
        if self._iter_counter >= len(self._indices):
            raise StopIteration()
        idx = self._indices[self._iter_counter]
        bxs = self._xs[idx]
        bys = self._ys[idx]
        self._iter_counter += 1
        return bxs, bys
    
    def __len__(self):
        return len(self._indices)


class DataLoader:

    def __init__(self, xs: NDArray, ys: NDArray, batch_size: int, shuffle: bool = True, seed: int = None):
        assert len(xs) == len(ys)
        assert batch_size > 0
        self._xs = xs
        self._ys = ys
        self.size = len(xs)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self.rng = XRandomGenerator(seed)
    
    def __iter__(self):
        idx = np.arange(self.size)
        if self._shuffle:
            idx = self.rng.permutation(idx)
        idx = [
            idx[i * self._batch_size:(i + 1) * self._batch_size]
            for i in range(int(np.ceil(self.size / self._batch_size)))
        ]
        return DLIter(self._xs, self._ys, idx)
    
    @classmethod
    def from_dataset(
            cls,
            dataset: Dataset,
            batch_size:int,
            shuffle: bool = False,
            seed: int = None
    ) -> "DataLoader":
        return cls(
            dataset.xs, dataset.ys,
            batch_size, shuffle=shuffle, seed=seed
        )
