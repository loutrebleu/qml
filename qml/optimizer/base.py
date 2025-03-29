# -*- coding: utf-8 -*-

import numpy as np

from IPython.display import clear_output

from . import evaluator as xeval
from ..model.model import Model
from ..tools.random import XRandomGenerator
from ..tools.dataset import Dataset
from ..tools.logger import Logger


class Optimizer:

    MODE_PREFIX = dict(
        info="[info]",
    )
    EVALUATOR_CLASS = xeval.MSEEvaluator

    def __init__(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset = None,
            test_interval: int = None,
            show_interval: int = 10,
            evaluator: xeval.Evaluator = None,
            shots: int = 50,
            seed: int = None
    ):
        self.rng = XRandomGenerator(seed)
        if test_dataset is None:
            test_dataset = train_dataset
        if evaluator is None:
            evaluator = self.EVALUATOR_CLASS

        self.train_dataset = train_dataset
        self.train_eval = None

        self.test_dataset = test_dataset
        self.test_eval = None
        self.test_interval = test_interval

        self.evaluator_class: xeval.Evaluator = evaluator
        self.shots = shots
        self.verbose = False
        self.logger: Logger = None
        self.show_interval = show_interval

        self.last_r = None

        self.xb = None
        self.eb = None
    
    def initialize(self, model, batch_size):
        train_batch_size = len(self.train_dataset) if batch_size is None else batch_size
        test_batch_size = len(self.test_dataset) if batch_size is None else batch_size
        self.train_eval = self.evaluator_class(self.train_dataset, model, train_batch_size)
        self.test_eval = self.evaluator_class(self.test_dataset, model, test_batch_size)
        self.logger = Logger()
    
    def initial_candidate(self, model: Model):
        xc = np.asarray([
            unit for unit in model.trainable_parameters
        ]).astype(float)
        return xc

    def evaluate(self, params, test: bool = False) -> float:
        eval_func = self.train_eval if not test else self.test_eval
        return eval_func(params)
    
    def metrics(self, res) -> float:
        raise NotImplementedError()
    
    def log(self, step, xc, rc, xp=None, rp=None, test=False):
        if test:
            self.log_test(step, xc, rc)
            return
        # updata best solution
        if self.xb is None:
            self.xb = xc
            self.rb = rc
        elif self.metrics(rc) <= self.metrics(self.rb):
            self.xb = xc
            self.rb = rc
        if xp is None:
            xp = xc
            rp = rc
        self.logger.store(
            step, xp, xc, self.xb, self.metrics(rp), self.metrics(rc), self.metrics(self.rb)
        )
    
    def log_test(self, step, xt, rt):
        self.logger.store_test(
            step, xt, self.metrics(rt)
        )

    def optimize(
            self,
            model: Model,
            num_iter: int,
            batch_size: int = None,
            shots: int = None,
            test_interval: int = None,
            verbose: bool = False,
    ):
        if shots is not None:
            self.shots = shots
        test_interval = num_iter+1 if test_interval is None else test_interval

        self.initialize(model, batch_size)
        self.verbose = verbose

        xc = self.initial_candidate(model)
        rc = self.evaluate(xc)
        self.last_res = rc

        # initial logging
        self.log(0, xc, rc)
        if test_interval is not None:
            rt = self.evaluate(xc, test=True)
            self.log_test(0, xc, rt)

        for step in range(1, num_iter+1):
            xc, rc = self.optimize_once(step, xc, rc)

            if step % test_interval == 0:
                self.test_once(step)

            if self.verbose and step % self.show_interval == 0:
                clear_output()
                self.logger.draw()

            self.last_res = rc
        
        return self.logger

    def optimize_once(self, step, xc, rc):
        xp = self.propose_candidate(xc)
        rp = self.evaluate(xp)

        xc, rc = self.update_candicate(step, xc, rc, xp, rp)
        self.log(step, xc, rc, xp, rp)

        return xc, rc
    
    def test_once(self, step, xc=None):
        xt = xc if xc is not None else self.xb
        rt = self.evaluate(xt, test=True)
        self.log_test(
            step, xt, rt
        )

    def propose_candidate(self, xc):
        raise NotImplementedError()

    def update_candicate(self, step, xc, rc, xp, rp):
        raise NotImplementedError()
    
    @classmethod
    def print(cls, values, mode="info", verbose=True):
        if verbose:
            values = str(values)
            if mode is not None:
                values = cls.MODE_PREFIX[mode] + " " + values
            print(values)
    
    @classmethod
    def info(cls, values, verbose=True):
        cls.print(values, mode="info", verbose=verbose)


