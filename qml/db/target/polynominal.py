# -*- coding: utf-8 -*-


import numpy as np

from typing import Callable

from .base import TargetFunctionGenerator


class PolynominalTargetFunctionGenerator(TargetFunctionGenerator):

    def __init__(self, max_order: int, seed: int = None):
        super().__init__(seed)
        self._max_order = max_order
    
    def base_func(self):
        coefs = self.rng.normal(0, 1, size=(self._max_order, 1))
        def bf(xs):
            powered_xs = np.vstack([
                np.power(xs, n + 1)
                for n in range(self._max_order)
            ])
            ys = (powered_xs * coefs).sum(axis=0)
            return ys
        return bf, dict(coefs=coefs.flatten())
