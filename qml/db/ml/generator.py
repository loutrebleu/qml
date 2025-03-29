# -*- coding: utf-8 -*-

from .. import target as xtarget
from ...tools.random import XRandomGenerator
from ...tools.dataset import Dataset


class MLDatasetGenerator:

    def __init__(self, function_generator: xtarget.TargetFunctionGenerator, seed: int = None):
        self.rng = XRandomGenerator(seed)
        self.fgen = function_generator
    
    def generate(self, size: int):
        func = self.fgen.generate()
        xs = self.rng.uniform(-1, 1, size)
        ys = func(xs)
        return Dataset(xs, ys)
