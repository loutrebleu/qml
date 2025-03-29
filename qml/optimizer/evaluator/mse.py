# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray

from .base import Evaluator, EvalResult
from .error import ErrorEvaluator
from ...model.model import Model
from ...tools.dataset import Dataset


class MSEEvalResult(EvalResult):

    def __init__(self, loss: float, xs: NDArray, ys: NDArray):
        super().__init__(xs, ys)
        self._loss = np.asarray(loss)
    
    @property
    def loss(self):
        return self._loss.copy()


class MSEEvaluator(Evaluator):

    def __init__(
            self,
            dataset: Dataset,
            model: Model = None,
            batch_size: int = None,
            shots: int = 50,
            seed: int = None,
            raise_iteration_error: bool = None,
    ):
        super().__init__(dataset, model, batch_size, shots, seed, raise_iteration_error)
    
    @classmethod
    def evaluate(
            cls,
            model: Model,
            params: NDArray,
            xs: NDArray,
            ys: NDArray,
            shots: int = None,
    ):
        res = ErrorEvaluator.evaluate(model, params, xs, ys, shots)
        loss = np.square(res.es).mean()
        return MSEEvalResult(loss, xs, ys)
