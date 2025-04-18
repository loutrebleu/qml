# -*- coding: utf-8 -*-

from . import evaluator as xeval
from .base import Optimizer
from ..tools.dataset import Dataset


class LocalSearchOptimizer(Optimizer):
    
    def __init__(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset = None,
            test_interval: int = None,
            evaluator: xeval.Evaluator = None,
            shots: int = 50,
            div_candidate: float = 0.3,
    ):
        super().__init__(
            train_dataset,
            test_dataset,
            test_interval=test_interval,
            evaluator=evaluator,
            shots=shots,
        )
        self.sigma = div_candidate
    
    def propose_candidate(self, xc):
        xp = self.rng.normal(xc, self.sigma, size=xc.shape)
        return xp
    
    def update_candicate(self, step, xc, rc, xp, rp):
        ec = self.metrics(rc)
        ep = self.metrics(rp)
        if ep <= ec:
            return xp, rp
        return xc, rc
    
    def metrics(self, res):
        return res.loss
