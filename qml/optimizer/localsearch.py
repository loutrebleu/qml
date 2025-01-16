# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from IPython.display import clear_output

from ..model.model import Model
from ..tools.logger import Logger
from ..tools.evaluator import Evaluator


class LocalSearchOptimizer:
    MAX_THETA = np.pi * 2

    def __init__(
            self,
            train_data: list[NDArray],
            test_data: list[NDArray] = None,
            shots: int = 50,
            test_interval: int = 5,
            random_search: bool = False,
            single_param_update: bool = False,
            variance: float = 0.1,
    ):
        self._train_data = train_data
        if test_data is None:
            test_data = train_data
        self._test_data = test_data
        self._shots = shots
        self._test_interval = test_interval
        self.random_search = random_search
        self.single_param_update = single_param_update
        self._variance = variance
        self._logger = None

    def prepare_evaluator(self, data, model, shots):
        evaluator = Evaluator(*data, model=model, shots=shots)
        params = model.trainable_parameters
        nums_params = [len(param) for param in params]
        nums_params.insert(0, 0)

        def eval_func(x):
            xs = [
                x[idx_de:idx_to]
                for idx_de, idx_to in zip(nums_params[:-1], nums_params[1:])
            ]
            return evaluator(xs)

        return eval_func

    def optimize(
            self,
            model: Model,
            num_iter: int,
            shots: int = None,
            test_interval: int = None,
            verbose: bool = False,
    ):
        train_eval = self.prepare_evaluator(self._train_data, model, shots=shots)
        test_eval = self.prepare_evaluator(self._test_data, model, shots=shots)

        xc = np.hstack(model.trainable_parameters)
        ec = train_eval(xc)

        xb = xc
        eb = ec

        logger = Logger()
        logger.store(0, xc, xc, xb, ec, ec, eb)
        logger.store_test(0, xc, test_eval(xc))

        for i in range(num_iter):
            step = i + 1
            xp = self.propose_candidate(xc)
            ep = train_eval(xp)

            if ep <= ec:
                xc = xp
                ec = ep
            if ep <= eb:
                xb = xp
                eb = ep

            logger.store(step, xp, xc, xb, ep, ec, eb)

            if step % self._test_interval == 0:
                logger.store_test(step, xc, test_eval(xc))

            if verbose:
                clear_output()
                print(f"Step:{step:3d} | Energy current:{ec:5.3f}  best:{eb:5.3f}")
                logger.draw()

        return logger

    def propose_candidate(self, xc):
        if self.single_param_update:
            target = np.random.randint(0, len(xc))

        if self.random_search:
            if self.single_param_update:
                candidate = xc.copy()
                candidate[target] = np.random.uniform() * self.MAX_THETA
                return candidate
            return np.random.uniform(size=xc.size) * self.MAX_THETA

        if self.single_param_update:
            candidate = xc.copy()
            candidate[target] += np.random.normal(0, self._variance)
            return candidate % self.MAX_THETA
        candidate = xc + np.random.normal(0, self._variance, size=xc.shape)
        return candidate % self.MAX_THETA

