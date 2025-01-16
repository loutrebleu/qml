# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from collections import namedtuple

from .gate import GateInfo, get_gateset, Gate


class Unit:
    VALUE_MAX = 2 * np.pi

    def __init__(
            self,
            name: str,
            gates: list[Gate],
            params: list[Parameter],
            values: list[float] | NDArray,
    ):
        self._name = name
        self._gates = gates
        self._params = params
        self._values = np.asarray(values) % self.VALUE_MAX

    def feed_dict(self, values=None) -> dict[str, float]:
        if values is None:
            values = self.values
        if not hasattr(values, "__len__"):
            values = [values]
        assert len(values) == len(
            self.parameters), f"Length of values {len(values)} must be equal to number of parameters {len(self.values)}"

        feed_dict = dict()
        for param, value in zip(self._params, values):
            feed_dict |= {
                param.name: value
            }
        return feed_dict

    def apply_to_qc(self, qc: QuantumCircuit) -> QuantumCircuit:
        for gate in self._gates:
            gate.apply_to_qc(qc)
        return qc

    @classmethod
    def generate_random_unit(
            cls,
            name: str,
            num_qubit: int,
            num_gate: int,
            gateset: dict[str, GateInfo] = None,
    ):
        if gateset is None:
            gateset = get_gateset(num_qubit)

        # select gate at random
        gate_names_on_set = list(gateset.keys())
        gate_names = np.random.choice(gate_names_on_set, size=num_gate, replace=True)

        # select qubits to apply gates
        qubits = np.random.randint(0, num_qubit, size=num_gate)

        return cls.new_with_gate_names_and_qubits(name, gate_names, qubits, gateset)

    @classmethod
    def new_with_gate_names_and_qubits(
            cls,
            name: str,
            gate_names: list[str],
            qubits: list[int],
            gateset: dict[str, GateInfo]
    ):
        gate_infos = [gateset[gate_name] for gate_name in gate_names]

        # build instance of gates and parameters
        gates = []
        params = []
        for gate_info, qubit in zip(gate_infos, qubits):
            if not gate_info.trainable:
                gates.append(Gate.new_with_info(gate_info, qubit))
                continue

            pname = f"param_{len(params)}"
            if name is not None:
                pname = name + "_" + pname
            param = Parameter(pname)
            params.append(param)
            gates.append(Gate.new_with_info(gate_info, qubit, param))

        # initialize parameter values
        values = np.zeros_like(params)

        return cls(name, gates, params, values)

    @property
    def values(self):
        return self._values.copy()

    @values.setter
    def values(self, values):
        assert len(values) == len(
            self.values), f"Length of values {len(values)} must be equal to number of parameters {len(self.values)}"
        values = np.asarray(values)
        values = values % self.VALUE_MAX
        self._values = values

    @property
    def parameters(self):
        return self.values

    @parameters.setter
    def parameters(self, values):
        self.values = values

    @property
    def gates(self):
        return [gate for gate in self._gates]

    @property
    def parameter_instances(self):
        return self._params

    @property
    def num_param(self):
        return len(self._params)


class EmbedUnit(Unit):

    def __init__(
            self,
            name: str,
            gates: list[Gate],
            params: list[Parameter],
            values: list[float] | NDArray,
    ):
        super().__init__(name, gates, params, values)
        self.pre_process = lambda x: x

    def feed_dict(self, values=None) -> dict[str, float]:
        if values is None:
            values = self.values
        values = self.pre_process(values)
        return super().feed_dict(values=values)

    @staticmethod
    def generate_ry_arcsin_embed_unit(
            name: str,
            num_qubit: int,
            dim_input: int,
            gateset: dict[str, GateInfo] = None,
    ):
        if gateset is None:
            gateset = get_gateset(num_qubit)
        gates = ["ry" for _ in range(num_qubit)]
        qubits = [i for i in range(num_qubit)]

        unit = EmbedUnit.new_with_gate_names_and_qubits(
            name, gates, qubits, gateset
        )
        unit.pre_process = lambda x: [
            np.arcsin(x) for _ in range(num_qubit)
        ]
        return unit


PresetUnitInfo = namedtuple('PresetUnitInfo', ["gate_names", "qubits"])


class EntangleUnit(Unit):
    NEIGHBOR_INFOS = {
        2: PresetUnitInfo(
            ["cz"], [0]
        ),
        3: PresetUnitInfo(
            ["cz", "cz", "cz"],
            [0, 1, 2]
        ),
    }

    def __init__(
            self,
            name: str,
            gates: list[Gate],
            params: list[Parameter],
            values: list[float] | NDArray,
    ):
        super().__init__(name, gates, params, values)

    @staticmethod
    def new_neighbor_cz(name: str, num_qubit: int, gateset: dict[str, GateInfo] = None):
        if gateset is None:
            gateset = get_gateset(num_qubit)
        infos = EntangleUnit.NEIGHBOR_INFOS[num_qubit]
        return EntangleUnit(name, EntangleUnit._new_with_infos(infos, gateset), [], [])

    @staticmethod
    def _new_with_infos(infos, gateset):
        gate_infos: list[GateInfo] = [
            gateset[gname]
            for gname in infos.gate_names
        ]
        gates = [
            Gate.new_with_info(info, qubit)
            for info, qubit in zip(gate_infos, infos.qubits)
        ]
        return gates
