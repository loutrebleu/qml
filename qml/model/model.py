# -*- coding: utf-8 -*-

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .unit import Unit
from ..tools.typing import Vector


class Model:

    def __init__(
            self,
            num_qubits: int,
            dim_output: int,
            input_units: Unit | list[Unit],
            fixed_units: Unit | list[Unit] = None,
            trainable_units: Unit | list[Unit] = None,
            shots: int = 100,
            sim=None,
    ):        
        self.nq = num_qubits
        self.nc = dim_output

        self._input_units = self._to_list(input_units)
        self._fixed_units = self._to_list(fixed_units)
        self._trainable_units = self._to_list(trainable_units)

        self._shots = shots
        self._sim = sim if sim is not None else AerSimulator()
    
    @staticmethod
    def _to_list(units: list[Unit] | Unit):
        if units is None:
            return []
        if hasattr(units, "__len__"):
            return units
        return [units]
    
    def forward(self, x: Vector, params=None, shots: int =  None) -> float:
        if params is None:
            params = [unit.values for unit in self._trainable_units]
        if shots is None:
            shots = self._shots
        feed_dict = dict()
        for unit in self._input_units:
            feed_dict |= unit.feed_dict(x)
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
        qc =QuantumCircuit(self.nq, self.nc)
        [
            unit.apply_to_qc(qc)
            for unit in self._input_units
        ]
        [
            unit.apply_to_qc(qc)
            for unit in self._fixed_units
        ]
        [
            unit.apply_to_qc(qc)
            for unit in self._trainable_units
        ]

        [
            qc.measure(i, i)
            for i in range(self.nc)
        ]
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
    
    def fix_trainable_units(self):
        [
            self._fixed_units.append(unit)
            for unit in self._trainable_units
        ]
        self._trainable_units = []
    
    @property
    def shots(self):
        return self._shots
    
    @shots.setter
    def shots(self, value):
        assert value > 0, f"shots is positive integer, but {value} is given."
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
    
    def deflatten(self, flattened_params):
        return self._deflatten(self, flattened_params)

    @staticmethod
    def _deflatten(model, flattened_params):
        trainable_params = model.trainable_parameters
        tp_shapes = [len(tp) for tp in trainable_params]
        tp_shapes.insert(0, 0)
        tp_shape_idxs = np.cumsum(tp_shapes)
        return [
            flattened_params[idx_de:idx_to]
            for idx_de, idx_to
            in zip(tp_shape_idxs[:-1], tp_shape_idxs[1:])
        ]
    
    def draw(self, ax=None):
        qc = self._apply()
        if ax is not None:
            qc.draw("mpl", ax=ax)
            return
        qc.draw("mpl")
    