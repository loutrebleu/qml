# -*- coding: utf-8 -*-

import xtools as xt
import yaml

from qml.optimizer import evaluator as xeval

class Config:

    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self._dict = yaml.safe_load(f)
            self.cf = cf = xt.Config(self._dict)
        
        self.nq = cf.circuit.num_qubits
        self.ng = cf.circuit.num_gates
        self.nl = cf.circuit.num_layers
        self.shots = cf.circuit.shots

        self.nx = cf.qml.db.dim_input
        self.ny = cf.qml.db.dim_output

        self.qml = cf.qml
        self.dpo = cf.dpo
        
        self.ocg = cf.ocg
        self.wavelet = xeval.Haar()
