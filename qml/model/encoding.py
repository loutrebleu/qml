# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum

from types import MethodType

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from .gate import Gateset, Gate, GateInfo
from .unit import Unit
from ..tools.typing import Vector, IntVector


class EncodingUnit(Unit):
    
    def __init__(
            self,
            name: str,
            gates: list[Gate],
            params: list[Parameter],
            values: Vector,
    ):
        super().__init__(name, gates, params, values)
    
    @staticmethod
    def pre_process(x: Vector) -> Vector:
        raise NotImplementedError()
    
    def feed_dict(self, x):
        values = self.pre_process(x)
        
        feed_dict = dict()
        for param, value in zip(self._params, values):
            feed_dict |= {
                param.name: value
            }
        return feed_dict


class EncodingUnitFactory:

    unit_class = EncodingUnit

    @classmethod
    def create(
            cls,
            dim_input: int,
            num_qubits: int,
            name: str = None,
            repeat: bool = False,
    ) -> EncodingUnit:
        patterns = cls._create_patterns(dim_input, num_qubits, repeat)

        if name is None:
            name = "encode"
        
        gates = []
        params = []

        for info, qubit in zip(*patterns):
            if not info.trainable:
                gates.append(Gate.new_with_info(info, qubit))
                continue

            pname = f"param_{len(params)}"
            pname = "_".join([name, pname])
            param = Parameter(pname)
            params.append(param)

            gates.append(Gate.new_with_info(info, qubit, param))
        
        values = np.zeros_like(params)

        unit = cls.unit_class(name, gates, params, values)
        num_repeat = num_qubits // cls.required_qubits(dim_input)
        if repeat and num_repeat > 1:
            unit.pre_process = cls.repeat_pre_process_func(num_repeat)
        
        return unit

    
    @classmethod
    def _create_patterns(
            cls,
            dim_input: int,
            num_qubits: int,
            repeat: bool = None,
    ) -> tuple[list[GateInfo], IntVector]:
        num_repeat = num_qubits // cls.required_qubits(dim_input)

        if not repeat or num_repeat == 1:
            return cls._create_single_pattern(dim_input, num_qubits)
        
        infos = []
        qubits = []
        
        for ridx in range(num_repeat):
            infos_, qubits_ = cls._create_single_pattern(dim_input, num_qubits)
            qubits_ = np.asarray(qubits_) + ridx * cls.required_qubits(dim_input)
            infos.append(infos_)
            qubits.append(qubits_)
        
        infos = sum(infos, [])
        qubits = np.hstack(qubits)

        return infos, qubits
        
    @classmethod
    def _create_single_pattern(cls, dim_input: int, num_qubits: int) -> tuple[list[GateInfo], IntVector]:
        raise NotImplementedError()

    @classmethod
    def required_qubits(cls, dim_input: int) -> int:
        raise NotImplementedError()
    
    @classmethod
    def repeat_pre_process_func(cls, num_qubits):
        return lambda x: np.hstack([
            cls.unit_class.pre_process(x)
            for _ in range(num_qubits)
        ])


# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#
#  Algorithm Implementation
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
#  Angle Encoding
#
class AngleEncodingUnit(EncodingUnit):

    @staticmethod
    def pre_process(x: Vector) -> Vector:
        return x


class AngleEncodingUnitFactory(EncodingUnitFactory):
    
    unit_class = AngleEncodingUnit

    @classmethod
    def _create_single_pattern(cls, dim_input: int, num_qubits: int) -> tuple[list[GateInfo], IntVector]:
        gset = Gateset.set_num_qubits(num_qubits)
        infos = [gset.RY for _ in range(cls.required_qubits(dim_input))]
        qubits = [i for i in range(cls.required_qubits(dim_input))]
        return infos, qubits

    @classmethod
    def required_qubits(cls, dim_input: int) -> int:
        return dim_input


# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#
#  Facctory Manager
#
class EncodingUnitManager:

    @staticmethod
    def create_create_func(factory_class):
        def func(
                cls,
                dim_input: int,
                num_qubits: int,
                name: str = None,
                repeat: bool = False,
        ) -> EncodingUnit:
            return factory_class.create(dim_input, num_qubits, name=name, repeat=repeat)
        return func


EncodingUnitManager.AngleEncoding = MethodType(
    EncodingUnitManager.create_create_func(AngleEncodingUnitFactory),
    EncodingUnitManager
)
