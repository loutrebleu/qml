# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray

from .base import Evaluator
from .error import ErrorEvaluator, ErrorEvalResult
from ...model.model import Model
from ...tools.dataset import Dataset


class GradientEvalResult(ErrorEvalResult):

    def __init__(self, grads: NDArray, errors: NDArray, xs: NDArray, ys: NDArray) -> None:
        super().__init__(errors, xs, ys)
        self._grads = grads

    @property
    def gradients(self):
        return self._grads.copy()
    
    @property
    def grads(self):
        return self.gradients

    @property
    def loss(self):
        return np.square(self.es).mean()



class GradientEvaluator(Evaluator):

    demi_pi = np.pi / 2

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
        grads = np.asarray([
            cls.calc_gradient_idx_(model, params, x, shots=shots)
            for x in xs
        ])
        eres = ErrorEvaluator.evaluate(model, params, xs, ys, shots=shots)
        return GradientEvalResult(grads, eres.errors, xs, ys)
    

    @classmethod
    def calc_gradient_idx_(
            cls,
            model: Model,
            params: NDArray,
            x: NDArray,
            shots: int = None,
    ):
        if params is None:
            params = model.trainable_parameters
        trainable_params = params.copy()
        tp_shapes = [len(tp) for tp in trainable_params]
        tp_shapes.insert(0, 0)
        tp_shape_idxs = np.cumsum(tp_shapes)

        trainable_params = np.hstack(trainable_params)

        def deflatten(flattened):
            return [
                flattened[idx_de:idx_to]
                for idx_de, idx_to
                in zip(tp_shape_idxs[:-1], tp_shape_idxs[1:])
            ]
        
        def calc_gradient_idx(idx):
            shifted_pos = trainable_params.copy()
            shifted_neg = trainable_params.copy()
            shifted_pos[idx] = trainable_params[idx] + cls.demi_pi
            shifted_neg[idx] = trainable_params[idx] - cls.demi_pi

            predict_pos = model.forward(
                x,
                params=deflatten(shifted_pos),
                shots=shots,
            )
            predict_neg = model.forward(
                x,
                params=deflatten(shifted_neg),
                shots=shots,
            )
            grad = (predict_pos - predict_neg) / 2
            return grad
        
        grads = np.asarray([
            calc_gradient_idx(idx)
            for idx in range(len(trainable_params))
        ])
        return grads
