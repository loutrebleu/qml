# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from ..model.unit import UnitManager
from ..optimizer import dpo as xdpo
from .random import XRandomGenerator


class CandidateSampler:

    def __init__(self, policy: xdpo.Policy, seed: int = None):
        self.rng = XRandomGenerator(seed)
        self.policy = policy

        self.nq = policy.nq
        self.ng = policy.ng
        self.gset = policy.gset
        self.dim_gindices =  policy.dim_gindices

        self.uman = UnitManager(self.nq, self.ng, self.rng.new_seed())

    
    def divide_gate_and_qubit(self, logits):
        logits_gate = logits[..., :self.dim_gindices]
        logits_qbit = logits[..., self.dim_gindices:]
        return logits_gate, logits_qbit
    
    def as_logps(self, logits):
        logits_gate, logits_qbit = self.divide_gate_and_qubit(logits)
        logps_gate = nn.functional.log_softmax(logits_gate, dim=-1)
        logps_qbit = nn.functional.log_softmax(logits_qbit, dim=-1)
        return logps_gate, logps_qbit
    
    def as_probs(self, logits, as_numpy=False):
        logits_gate, logits_qbit = self.divide_gate_and_qubit(logits)
        probs_gate = torch.softmax(logits_gate.view(self.ng, -1), dim=-1)
        probs_qbit = torch.softmax(logits_qbit.view(self.ng, -1), dim=-1)
        if not as_numpy:
            return probs_gate, probs_qbit
        return (
            probs_gate.detach().numpy(),
            probs_qbit.detach().numpy(),
        )
    
    def sample_from_probs(self, probs, clist):
        return [
            self.rng.choice(clist, p=prob)
            for prob in probs
        ]
    
    def sample(self, x, num_candidates: int = 1):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if x.ndim < 2:
            x = x.unsqueeze()
        logits = self.policy.forward(x)
        probs_gate, probs_qbit = self.as_probs(logits, as_numpy=True)

        candidates = [
            self.uman.from_info_and_qubits(
                self.sample_from_probs(probs_gate, list(self.gset.values())),    # gate infos
                self.sample_from_probs(probs_qbit, [i for i in range(self.nq)]), # qubits
            )
            for _ in range(num_candidates)
        ]
        return candidates
