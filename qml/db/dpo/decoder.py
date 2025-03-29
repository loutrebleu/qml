# -*- coding: utf-8 -*-

import numpy as np
import json
from collections import namedtuple

from ...model.gate import Gateset


DPOData = namedtuple("DPOData", ["wseries", "gindices", "qubits", "losses"])

class DPODataDecoder:

    KEY_WSERIES = "wseries"
    KEY_UNITS = "units"
    KEY_UNITS_GATES = "gates"
    KEY_UNITS_QUBITS = "qubits"
    KEY_LOSSES = "losses"

    @classmethod
    def from_json(cls, djson: str, gateset: Gateset, dim_wavelet: int = 4):
        gdict = {gate_name: idx for idx, gate_name in enumerate(gateset.keys())}
        ddict = json.loads(djson)

        # wseries
        wseries = np.asarray(ddict[cls.KEY_WSERIES])
        len_wseries = 2 ** dim_wavelet - 1
        dwseries = wseries[:len_wseries]

        # units
        udicts = ddict[cls.KEY_UNITS]
        # units/gate_indices
        dginfices = [
            [
                gdict[ugate.upper()]
                for ugate in udict[cls.KEY_UNITS_GATES]
            ] for udict in udicts
        ]
        dqubits = [
            udict[cls.KEY_UNITS_QUBITS]
            for udict in udicts
        ]

        # losses
        dlosses = np.asarray(ddict[cls.KEY_LOSSES])

        return DPOData(dwseries, dginfices, dqubits, dlosses)
    
    @classmethod
    def divide_best_and_others(cls, data: DPOData):
        bgindices = data.gindices
        bqubits = data.qubits
        blosses = data.losses

        bbest_indices = np.argmin(blosses, axis=1)

        # collect best candidates
        best_gindices = [
            [gindices[best_index]]
            for gindices, best_index in zip(bgindices, bbest_indices)
        ]

        best_qubits = [
            [qubits[best_index]]
            for qubits, best_index in zip(bqubits, bbest_indices)
        ]

        best_losses = np.min(blosses, axis=1, keepdims=True)

        best_data = DPOData(data.wseries, best_gindices, best_qubits, best_losses)

        # collect others
        others_ginfices = [
            [gindex for idx, gindex in enumerate(gindices) if idx != best_index]
            for gindices, best_index in zip(bgindices, bbest_indices)
        ]

        others_qubits = [
            [qubit for idx, qubit in enumerate(qubits) if idx != best_index]
            for qubits, best_index in zip(bqubits, bbest_indices)
        ]

        others_losses = [
            [loss.item() for idx, loss in enumerate(losses) if idx != best_index]
            for losses, best_index in zip(blosses, bbest_indices)
        ]

        others_data = DPOData(data.wseries, others_ginfices, others_qubits, others_losses)
        
        return best_data, others_data


