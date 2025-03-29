# -*- coding: utf-8 -*-

import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from .unit import Unit


class Model:

    def __init__(
            self,
            num_qubit: int,
            num_output: int,
            input_units: Unit,
            fixed_units: list[Unit],
            trainable_units: list[Unit] = None,
            shots: int = 100,
            sim=None
    ):
        if not hasattr(fixed_units, "__len__"):
            fixed_units = [fixed_units]
        if trainable_units is not None and not hasattr(trainable_units, "__len__"):
            trainable_units = [trainable_units]
        self.nq = num_qubit
        self.nc = num_output
        self._input_units = input_units
        self._fixed_units = fixed_units
        self._trainable_units = trainable_units
        self._shots = shots
        self._sim = sim if sim is not None else AerSimulator()

    def forward(self, x, params=None, shots=None) -> float:
        if params is None:
            params = [unit.values for unit in self._trainable_units]
        if shots is None:
            shots = self._shots
        feed_dict = self._input_units.feed_dict(x)
        for unit in self._fixed_units:
            feed_dict |= unit.feed_dict()
        for unit, param in zip(self._trainable_units, params):
            feed_dict |= unit.feed_dict(param)

        qc = self._apply()
        bc = qc.assign_parameters(feed_dict)
        job = transpile(bc, self._sim)
        res = self._sim.run(job, shots=shots).result().get_counts()
        pre = res.get("0", 0) - res.get("1", 0)
        return pre / shots

    def _apply(self):
        qc = QuantumCircuit(self.nq, self.nc)

        self._input_units.apply_to_qc(qc)
        [
            fixed_unit.apply_to_qc(qc)
            for fixed_unit in self._fixed_units
        ]
        [
            trainable_unit.apply_to_qc(qc)
            for trainable_unit in self._trainable_units
        ]

        qc.measure(0, 0)
        return qc

    @property
    def input_units(self):
        return self._input_units

    @property
    def fixed_units(self):
        return self._fixed_units

    @property
    def trainable_units(self):
        return self._trainable_units

    @trainable_units.setter
    def trainable_units(self, units):
        self._trainable_units.append(units)

    def fix_trainable_unit(self):
        [
            self._fixed_units.append(trainable_unit)
            for trainable_unit in self._trainable_units
        ]
        self._trainable_units = []

    @property
    def shots(self):
        return self._shots

    @shots.setter
    def shots(self, value):
        self._shots = value

    @property
    def trainable_parameters(self):
        return [
            unit.parameters for unit in self._trainable_units
        ]

    def update_parameters(self, new_parameters):
        for unit, param in zip(self.trainable_units, new_parameters):
            unit.values = param
        return self

    def draw(self, ax=None):
        qc = self._apply()
        if ax is not None:
            qc.draw("mpl", ax=ax)
            return
        qc.draw("mpl")

    def _draw(self, inplace=False):
        qc = self._apply()
        if inplace:
            qc.draw("mpl")
        img = qc.draw("mpl")
        return img

    def draw_qc_image(self):
        buf = io.BytesIO()
        fig_ = self._draw()
        fig_.savefig(buf, format='png', dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
