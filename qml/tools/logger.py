# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple


Solution = namedtuple('Solution', ['x', 'energy'])


class Logger:

    def __init__(self):
        self._step_total = []
        self._step_test = []
        # variables
        self._variable_proposed = []
        self._variable_current = []
        self._variable_best = []
        self._variable_test = []
        # energies
        self._energy_proposed = []
        self._energy_current = []
        self._energy_best = []
        self._energy_test = []

    def store(self, step, xp, xc, xb, ep, ec, eb):
        self._step_total.append(step)
        self._variable_proposed.append(xp)
        self._variable_current.append(xc)
        self._variable_best.append(xb)
        self._energy_proposed.append(ep)
        self._energy_current.append(ec)
        self._energy_best.append(eb)

    def store_test(self, step, xt, et):
        self._step_test.append(step)
        self._variable_test.append(xt)
        self._energy_test.append(et)

    @property
    def result_energies(self):
        return pd.DataFrame(dict(
            step=self._step_total,
            proposed=self._energy_proposed,
            current=self._energy_current,
            best=self._energy_best,
        ))

    @property
    def result_variable(self):
        xb = np.vstack(self._variable_best)
        return pd.DataFrame(
            dict(step=self._step_total) | {
                i: xb[..., i]
                for i in range(xb.shape[1])
            }
        )

    @property
    def result_test(self):
        return pd.DataFrame(dict(
            step=self._step_test,
            energy=self._energy_test,
        ))

    def draw(self, fig_name: str = None):
        fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
        self.result_energies.plot(x="step", y=["proposed", "current", "best"], ax=axes[0])
        self.result_test.plot(x="step", y="energy", ax=axes[0])
        self.result_variable.plot(x="step", ax=axes[1])
        plt.show()
        if fig_name is not None:
            fig.savefig(fig_name)

    @property
    def first(self):
        return Solution(self._variable_best[-1], self._energy_best[-1])
