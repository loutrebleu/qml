# -*- coding: utf-8 -*-

from collections import namedtuple
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Parameter
from qiskit.circuit.library import RXGate, RYGate, RZGate, CZGate
from typing import Callable


GateInfo = namedtuple(
    'GateInfo',
    ['gate_class', 'trainable', "multi_bit", "qargs"]
)


def single_qubit_qargs(qc: QuantumCircuit, qubit_idx: int) -> list:
    return [qc.qubits[qubit_idx]]


def generate_double_qubit_args(spread: int = 1) -> Callable:
    def double_qubit_args(qc: QuantumCircuit, qubit_idx: int) -> list:
        return [
            qc.qubits[qubit_idx],
            qc.qubits[(qubit_idx+spread) % qc.num_qubits],
        ]
    return double_qubit_args


GATESET_FOR_2_QUBIT = dict(
    rx=GateInfo(RXGate, True, False, single_qubit_qargs),
    ry=GateInfo(RYGate, True, False, single_qubit_qargs),
    rz=GateInfo(RZGate, True, False, single_qubit_qargs),
    cz=GateInfo(CZGate, False, True, generate_double_qubit_args(1)),
)


GATESET_FOR_3_QUBIT = dict(
    rx=GateInfo(RXGate, True, False, single_qubit_qargs),
    ry=GateInfo(RYGate, True, False, single_qubit_qargs),
    rz=GateInfo(RZGate, True, False, single_qubit_qargs),
    cz=GateInfo(CZGate, False, True, generate_double_qubit_args(1)),
)


def get_gateset(num_qubit: int) -> dict:
    if num_qubit == 2:
        return GATESET_FOR_2_QUBIT
    if num_qubit == 3:
        return GATESET_FOR_3_QUBIT
    raise ValueError(f'num_qubit {num_qubit} is not supported')


class Gate:

    def __init__(
            self,
            gate: Instruction,
            trainable: bool,
            multi_qubit: bool,
            qubit: int,
            qargs_func: Callable
    ):
        self._gate = gate
        self._trainable = trainable
        self._multi_qubit = multi_qubit
        self._qubit = qubit
        self._qargs_func = qargs_func

    def apply_to_qc(self, qc: QuantumCircuit):
        qargs = self._qargs_func(qc, self._qubit)
        qc.append(self._gate, qargs)

    @staticmethod
    def new_with_info(info: GateInfo, qubit: int, parameter: Parameter = None):
        gate = info.gate_class() if parameter is None else info.gate_class(parameter)
        return Gate(
            gate, info.trainable, info.multi_bit, qubit, info.qargs
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
