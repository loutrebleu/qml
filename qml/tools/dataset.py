# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray
from typing import Callable

from qml.tools.random import XRandomGenerator
from qml.tools.typing import Vector



def _generate_dataset(num_data: int, func: Callable, seed: int = None) -> "Dataset":
    rng = XRandomGenerator(seed)
    
    xs = rng.uniform(-1, 1, num_data)
    ys = func(xs)
    return Dataset(xs, ys)



class Dataset:

    def __init__(self, xs: list[Vector] | NDArray, ys: list[Vector] | NDArray):
        self._xs: NDArray = np.asarray(xs)
        self._ys: NDArray = np.asarray(ys)
    
    def __len__(self):
        return len(self._xs)
    
    @property
    def size(self) -> int:
        return len(self)
    
    @property
    def xs(self) -> NDArray:
        return self._xs.copy()
    
    @property
    def ys(self) -> NDArray:
        return self._ys.copy()
    
    @property
    def data(self) -> tuple[NDArray, NDArray]:
        return self.xs, self.ys
    
    @staticmethod
    def generate_dataset(num_data: int, func: Callable, seed: int = None) -> "Dataset":
        return _generate_dataset(num_data, func, seed=seed)
    
    @property
    def dim_input(self) -> int:
        return self._xs.shape[1]
    
    @property
    def dim_output(self) -> int:
        return self._ys.shape[1]
