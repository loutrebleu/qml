# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum

from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate as QkGate
from qiskit.circuit.library import RXGate, RYGate, RZGate, CZGate
from qiskit.circuit import Parameter, Instruction

from qml.tools.random import XRandomGenerator


def single_qubit_qargs(qc: QuantumCircuit, qubit_idx: int) -> list:
    return [qc.qubits[qubit_idx]]


def generate_double_qubit_args(spread: int = 1) -> Callable:
    def double_qubit_args(qc: QuantumCircuit, qubit_idx: int) -> list:
        return [
            qc.qubits[qubit_idx],
            qc.qubits[(qubit_idx+spread) % qc.num_qubits],
        ]
    return double_qubit_args


class GateInfo(Enum):

    RX = (RXGate, True, False, single_qubit_qargs)
    RY = (RYGate, True, False, single_qubit_qargs)
    RZ = (RZGate, True, False, single_qubit_qargs)
    CZ1 = (CZGate, False, True, generate_double_qubit_args(1))

    def __init__(
            self,
            gate_class: QkGate,
            trainable: bool,
            multibit: bool,
            qargs: Callable,
    ):
        self.gate_class = gate_class
        self.trainable = trainable
        self.multibit = multibit
        self.qargs = qargs
    
    @classmethod
    def get(cls, obj):
        if isinstance(obj, Gateset):
            return obj
        if not isinstance(obj, str):
            raise ValueError(f"Gateset.get() requires Gateset or string, but {type(obj)} is given.")
        name = obj.upper()
        for g in cls:
            if name == g.name:
                return g
        raise ValueError(f"Incorrect obj is given; {obj}")
    

class Gateset:

    GATE_LIST = dict()

    def __init__(self, num_qubits: int, seed: int = None):
        self._rng = XRandomGenerator(seed)
        self._nq = num_qubits

        for key, gate in self.GATE_LIST.items():
            self.__setattr__(key, gate)

        # accessors
            self.keys = self.GATE_LIST.keys
            self.items = self.GATE_LIST.items
            self.values = self.GATE_LIST.values
    
    @staticmethod
    def set_num_qubits(num_qubits: int, seed: int = None) -> "Gateset":
        if num_qubits == 2:
            return Gateset2Qubits(seed=seed)
        if num_qubits == 3:
            return Gateset3Qubits(seed=seed)
        raise ValueError(f'num_qubits {num_qubits} is not supported.')
    
    def __iter__(self):
        return iter(self.GATE_LIST.values())
    
    def get(self, obj):
        if isinstance(obj, GateInfo):
            return obj
        obj = obj.upper()
        if obj not in self.GATE_LIST:
            raise ValueError("Undefined gate is attempted to retrieve.")
        return self.GATE_LIST[obj]
    
    def get_at_random(self):
        return self._rng.choice([g for g in self])
    
    @property
    def size(self):
        return len(self.GATE_LIST)


class Gateset2Qubits(Gateset):

    NUM_QUBITS = 2

    GATE_LIST = dict(
        RX=GateInfo.RX,
        RY=GateInfo.RY,
        RZ=GateInfo.RZ,
        CZ=GateInfo.CZ1,
    )

    def __init__(self, seed: int = None):
        super().__init__(self.NUM_QUBITS, seed=seed)


class Gateset3Qubits(Gateset):

    NUM_QUBITS = 3

    GATE_LIST = dict(
        RX=GateInfo.RX,
        RY=GateInfo.RY,
        RZ=GateInfo.RZ,
        CZ=GateInfo.CZ1,
    )

    def __init__(self, seed: int = None):
        super().__init__(self.NUM_QUBITS, seed=seed)


class Gate:

    def __init__(
            self,
            gate: Instruction,
            trainable: bool,
            multibit: bool,
            qubit: int,
            qargs_func: Callable,
    ):
        self._gate = gate
        self._trainable = trainable
        self._multibit = multibit
        self._qubit = qubit
        self._qargs_func = qargs_func

    def apply_to_qc(self, qc: QuantumCircuit):
        qargs = self._qargs_func(qc, self._qubit)
        qc.append(self._gate, qargs)

    @staticmethod
    def new_with_info(info: GateInfo, qubit: int, parameter: Parameter = None):
        gate = info.gate_class() if parameter is None else info.gate_class(parameter)
        return Gate(
            gate, info.trainable, info.multibit, qubit, info.qargs
        )

    @property
    def gate(self):
        return self._gate

    @property
    def trainable(self):
        return self._trainable

    @property
    def multi_qubit(self):
        return self._multi_qubit

    @property
    def qubit(self):
        return self._qubit
