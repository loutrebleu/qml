# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray

from ..model.model import Model


class Evaluator:

    def __init__(
            self,
            xs: NDArray,
            ys: NDArray,
            model: Model = None,
            shots: int = 50
    ):
        self._xs = xs
        self._ys = ys
        self._model = model
        self.shots = shots

    def __call__(self, params: NDArray = None, model: Model = None) -> float:
        if model is None:
            model = self._model
        return Evaluator.evaluate(
            model, params, self._xs, self._ys, shots=self.shots
        )

    @staticmethod
    def evaluate(
            model: Model,
            params: NDArray,
            xs: NDArray,
            ys: NDArray,
            shots: int = 50
    ) -> float:
        predicts = np.asarray([
            model.forward(x, params=params, shots=shots)
            for x in xs
        ])
        errors = ys - predicts
        loss = np.square(errors).mean()
        return loss
