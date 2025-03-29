# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray

from ...model.model import Model
from ...tools.dataset import Dataset
from ...tools.dataloader import DataLoader
from ...tools.random import XRandomGenerator


class EvalResult:

    def __init__(self, xs: NDArray, ys: NDArray) -> None:
        self._xs = np.asarray(xs)
        self._ys = np.asarray(ys)
    
    @property
    def xs(self):
        return self._xs.copy()

    @property
    def ys(self):
        return self._ys.copy()


class Evaluator:

    def __init__(
            self,
            dataset: Dataset,
            model: Model = None,
            batch_size: int = None,
            shots: int = 50,
            seed: int = None,
            raise_iteration_error: bool = None,
    ):
        self._rng = XRandomGenerator(seed)
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size = batch_size if batch_size is not None else len(dataset)
        self.shuffle = False
        self.shots = shots

        self._loader = DataLoader.from_dataset(
            dataset, batch_size, shuffle=self.shuffle, seed=self._rng.new_seed()
        )
        self._loader_iter = None
        if raise_iteration_error is None and (batch_size is None or len(dataset) == batch_size):
            raise_iteration_error = False
        self._raise_iteration_error = raise_iteration_error
    
    def __call__(
            self,
            params: NDArray = None,
            model: Model = None
    ) -> EvalResult:
        if model is None:
            model = self.model
            
        if self._loader_iter is None:
            self._loader_iter = iter(self._loader)
        
        try:
            xs, ys = next(self._loader_iter)
        except StopIteration:
            if self._raise_iteration_error:
                raise StopIteration()
            self._loader_iter = iter(self._loader)
            xs, ys = next(self._loader_iter)
        
        return self.evaluate(
            model, params, xs, ys, shots=self.shots
        )
    
    @classmethod
    def evaluate(
        cls,
        model: Model,
        params: NDArray,
        xs: NDArray,
        ys: NDArray,
        shots: int = None,
    ):
        pass
        