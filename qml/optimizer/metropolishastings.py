# -*- coding: utf-8 -*-

import numpy as np
from . import evaluator as xeval
from .localsearch import LocalSearchOptimizer
from ..tools.dataset import Dataset


class MetropolisHastingsOptimizer(LocalSearchOptimizer):

    def __init__(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset = None,
            test_interval: int = None,
            evaluator: xeval.Evaluator = None,
            shots: int = 50,
            div_candidate: float = 0.3,
            temperature: float = 1.0,
    ):
        super().__init__(
            train_dataset,
            test_dataset,
            test_interval=test_interval,
            evaluator=evaluator,
            shots=shots,
            div_candidate=div_candidate,
        )
        assert temperature > 0, f"temperature must be positive float, but {temperature} is given."
        self.temp = temperature
        self.beta = 1 / temperature
    
    def update_candicate(self, step, xc, rc, xp, rp):
        ec = self.metrics(rc)
        ep = self.metrics(rp)
        if ec >= ep:
            return xp, rp

        acceptance_ratio = self.potential(ep - ec)
        if self.rng.random() <= acceptance_ratio:
            return xp, rp

        print("")
        return xc, rc
    
    def potential(self, rc: xeval.EvalResult | float):
        e = rc if isinstance(rc, float) else self.metrics(rc)
        return np.exp(-1. * e * self.beta)
