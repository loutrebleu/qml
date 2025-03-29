# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .decoder import DPOData, DPODataDecoder
from .dataset import DPODataset
from ...model.gate import Gateset
from ...tools.random import XRandomGenerator


class DPODataBatchDivided:

    def __init__(self, batch_data: DPOData, nq: int, ngc: int):
        self.data = batch_data
        self.nq = nq
        self.ngc = ngc
    
    @staticmethod
    def as_onehot(indices, num_classes):
        onehot = nn.functional.one_hot(indices, num_classes)
        return onehot

    @property
    def np_gindices(self):
        return np.asarray(self.data.gindices).astype(int)
    
    @property
    def gindices(self):
        return torch.from_numpy(self.np_gindices).float()
    
    @property
    def onehot_gindices(self):
        return self.as_onehot(self.gindices.long(), self.ngc)
    
    @property
    def np_qubits(self):
        return np.asarray(self.data.qubits).astype(int)
    
    @property
    def qubits(self):
        return torch.from_numpy(self.np_qubits).float()
    
    @property
    def onehot_qubits(self):
        return self.as_onehot(self.qubits.long(), self.nq)
    
    @property
    def np_losses(self):
        return np.asarray(self.data.losses).astype(float)
    
    @property
    def losses(self):
        return torch.from_numpy(self.np_losses).float()


class DPODataBatch:

    def __init__(self, batch_data: DPOData, num_qubits: int, num_gate_classes: int = None):
        if num_gate_classes is None:
            num_gate_classes = Gateset.set_num_qubits(num_qubits).size
        self.num_qubits = num_qubits
        self.num_gate_classes = num_gate_classes
        self.data = data = batch_data
        self.best_data = None
        self.others_data = None
        self._wseries = np.vstack(data.wseries)

        self.encode(data)
    
    def encode(self, data):
        best_data, others_data = DPODataDecoder.divide_best_and_others(data)
        self.best_data = DPODataBatchDivided(best_data, self.num_qubits, self.num_gate_classes)
        self.others_data = DPODataBatchDivided(others_data, self.num_qubits, self.num_gate_classes)
    
    @property
    def size(self):
        pass

    @property
    def wserieses(self):
        return torch.from_numpy(self._wseries).float()
    
    @property
    def np_wserieses(self):
        return self._wseries.copy()
    
    @property
    def best(self):
        return self.best_data
    
    @property
    def others(self):
        return self.others_data


class DPODataLoaderIter:

    def __init__(
            self,
            dataset: DPODataset,
            batched_indices: list[list[int]],
            num_qubits: int,
            num_gate_classes: int,
    ):
        self.db = dataset
        self.indices = batched_indices

        self.nq = num_qubits
        self.ngc = num_gate_classes

        self.indices_iter = iter(batched_indices)
    
    def __next__(self):
        idxs = next(self.indices_iter)
        bdata = [self.db[idx] for idx in idxs]
        bdata = DPOData(
            [data.wseries for data in bdata],
            [data.gindices for  data in bdata],
            [data.qubits for  data in bdata],
            [data.losses for  data in bdata],
        )
        return DPODataBatch(bdata, self.nq, self.ngc)

class DPODataLoader:
    
    def __init__(self, dataset: DPODataset, num_qubits: int, batch_size: int, max_wavelet_dim: int = 4, seed: int = None):
        self.rng = XRandomGenerator(seed)

        self.dataset = dataset
        self.num_qubits = num_qubits
        self.num_gate_classes = Gateset.set_num_qubits(num_qubits).size
        self.max_wavelet_dim = max_wavelet_dim
        self.batch_size = batch_size
    
    @property
    def size(self):
        return int(np.ceil(self.dataset.size / self.batch_size))
    
    def __iter__(self):
        indices = np.arange(self.dataset.size).astype(int)
        indices = self.rng.permutation(indices)
        batched_indices = indices.reshape((self.size, self.batch_size))
        return DPODataLoaderIter(self.dataset, batched_indices, self.num_qubits, self.num_gate_classes)
