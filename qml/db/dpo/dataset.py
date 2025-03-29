# -*- coding: utf-8 -*-

from ...model.gate import Gateset
from .decoder import DPODataDecoder

class DPODataset:

    def __init__(self, db_filename: str, num_qubits: int, dim_wavelet: int = 4):
        self.db_filename = db_filename
        self.num_qubits = num_qubits
        self.dim_wavelet = dim_wavelet

        self.gateset = Gateset.set_num_qubits(num_qubits)

        self.db = self.load_db_file()

    def __getitem__(self, index):
        djson = self.db[index]
        return DPODataDecoder.from_json(djson, self.gateset, self.dim_wavelet)
    
    def load_db_file(self):
        with open(self.db_filename) as fp:
            djsons = fp.readlines()        
        return djsons

    @property
    def size(self):
        return len(self.db)
