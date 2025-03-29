# -*- coding: utf-8 -*-


import numpy as np


class XRandomGenerator:

    SEED_RANGE = int(1e+8)

    def __init__(self, seed: int = None):
        if seed is None:
            seed = np.random.randint(0, self.SEED_RANGE)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        for key in dir(self._rng):
            if "__" in key:
                continue
            self.__setattr__(key, self._rng.__getattribute__(key))
    
    def new_seed(self):
        return self._rng.integers(0, self.SEED_RANGE)
