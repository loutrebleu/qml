# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from ..model.model import Model
from .evaluator import Evaluator


class ErrorPattern:

    def __init__(self, loss, pattern, predics, xs, ys):
        self.loss = loss
        self.pattern = pattern
        self.ps = predics
        self.xs = xs
        self.ys = ys
        self.es = pattern
    
    def draw_pattern(self, show_predict=False, show_y=False, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.scatter(self.xs, self.pattern, label="error")
        if show_predict:
            ax.scatter(self.xs, self.ps, label="predict")
        if show_y:
            ax.scatter(self.xs, self.ys, label="y")
        ax.legend()

        return fig, ax
    


class ErrorPatternEvaluator(Evaluator):
    
    def __init__(
            self,
            xs: NDArray,
            ys: NDArray,
            model: Model = None,
            shots: int = 50
    ):
        super().__init__(xs, ys, model, shots)

    def __call__(
            self,
            params: NDArray = None,
            model: Model = None,
    ) -> ErrorPattern:
        if model is None:
            model = self._model
        return self.evaluate(
            model, params, self._xs, self._ys, shots=self.shots
        )
    
    def evaluate(
            cls,
            model: Model,
            params: NDArray,
            xs: NDArray,
            ys: NDArray,
            shots: int = 50,
    ) -> ErrorPattern:
        predicts = np.asarray([
            model.forward(x, params=params, shots=shots)
            for x in xs
        ])
        errors = predicts - ys
        loss = np.square(errors).mean()
        return ErrorPattern(loss, errors, predicts, xs, ys)
