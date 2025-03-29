# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray

from .base import Evaluator, EvalResult
from ...model.model import Model
from ...tools.dataset import Dataset



class ErrorEvalResult(EvalResult):

    def __init__(self, errors: NDArray, xs: NDArray, ys: NDArray):
        super().__init__(xs, ys)
        self._es = np.asarray(errors)
    
    @property
    def es(self):
        return self._es.copy()
    
    @property
    def errors(self):
        return self.es


class ErrorEvaluator(Evaluator):

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
        ps = np.asarray([
            model.forward(x, params=params, shots=shots)
            for x in xs
        ])
        es = ys - ps
        res = ErrorEvalResult(es, xs, ys)
        return res

