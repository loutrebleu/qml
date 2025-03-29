# -*- coding: utf-8 -*-


import numpy as np

from typing import Callable

from .base import TargetFunctionGenerator


class TrigonometricTargetFuncGenerator(TargetFunctionGenerator):

    def __init__(self, max_frequency: int, seed: int = None):
        super().__init__(seed)
        self._max_frequency = max_frequency
    
    def base_func(self):
        fmax = self._max_frequency
        flags_select = self.rng.choice([0, 1], size=[2 * (fmax + 1), 1], replace=True)
        coefficients = self.rng.normal(0., 1., size=[2 * (fmax + 1), 1])
        coefficients = flags_select * coefficients

        def bf(xs):
            stacked_xs = np.vstack([xs for _ in range(fmax+1)])
            sin_cos = np.vstack([np.sin(stacked_xs), np.cos(stacked_xs)])
            ys = (sin_cos * coefficients).sum(axis=0)
            return ys
        
        return bf, dict(coefs=coefficients)
