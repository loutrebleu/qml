# -*- coding: utf-8 -*-


import numpy as np

from typing import Callable

from ...tools.random import XRandomGenerator


class TargetFunctionGenerator:

    NUM_POINT_TO_NORMALIZER = 500

    def __init__(self, seed: int = None):
        self.rng = XRandomGenerator(seed)
    
    @classmethod
    def wrap_with_normalizer(cls, func: Callable) -> Callable:
        xs = np.linspace(-1, 1, cls.NUM_POINT_TO_NORMALIZER)
        ys = np.asarray(func(xs))
        ymax = ys.max()
        ymin = ys.min()
        dys = ymax - ymin
        
        def normalizer(xs):
            base = func(xs)
            ys = (base - ymin) / dys * 2 - 1
            return ys
        
        return normalizer

    def generate(self, require_info: bool = False):
        bf, finfo = self.base_func()
        wrapped_func = self.wrap_with_normalizer(bf)
        if not require_info:
            return wrapped_func
        return wrapped_func, finfo
    
    def base_func(self):
        raise NotImplementedError()
