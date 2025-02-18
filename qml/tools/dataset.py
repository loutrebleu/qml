# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from typing import Callable


class Dataset:

    def __init__(self, xs, ys):
        self._xs: NDArray = np.asarray(xs)
        self._ys: NDArray = np.asarray(ys)

    def __len__(self):
        return len(self._xs)
    
    @property
    def size(self):
        return len(self)
    
    @property
    def xs(self):
        return self._xs.copy()
    
    @property
    def ys(self):
        return self._ys.copy()
    
    @property
    def data(self):
        return self.xs, self.ys


def generate_dataset(num_point: int, func: Callable, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng.integers)
    
    xs = rng.uniform(-1, 1, num_point)
    ys = func(xs)
    return Dataset(xs, ys)
