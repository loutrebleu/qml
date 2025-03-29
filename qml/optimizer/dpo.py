# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..model.gate import Gateset


def selected_logps(logits, onehot):
    logps = nn.functional.log_softmax(logits, dim=-1)
    selected_logps = (logps * onehot).sum(dim=-1)
    return selected_logps


def calc_logps(policy, batch):
    logits = policy(batch.wserieses)

    logits_gate = logits[..., :policy.dim_gindices].view(batch.best.onehot_gindices.shape)
    logits_qbit = logits[..., policy.dim_gindices:].view(batch.best.onehot_qubits.shape)

    logps_best = selected_logps(logits_gate, batch.best.onehot_gindices) + selected_logps(logits_qbit, batch.best.onehot_qubits)
    logps_best = logps_best.sum(dim=-1)
    logps_others = selected_logps(logits_gate, batch.others.onehot_gindices) + selected_logps(logits_qbit, batch.others.onehot_qubits)
    logps_others = logps_others.sum(dim=-1)
    
    return logps_best, logps_others


def calc_loss_dpo(logrp, logrp_ref, beta=0.5):
    return -1 * nn.functional.logsigmoid(beta * (logrp - logrp_ref)).mean()


def calc_loss_llh(logp_best):
    return torch.exp(logp_best).mean()


def hard_update(source: nn.Module, target: nn.Module):
    target.load_state_dict(source.state_dict())


class Policy(nn.Module):

    def __init__(
            self,
            dim_wavelet: int,
            num_qubits: int,
            num_gates: int,
            dim_hiddens: list[int]
    ):
        super().__init__()
        self.nq = num_qubits
        self.ng = num_gates
        self.gset = gset = Gateset.set_num_qubits(num_qubits)

        self.dim_gindices = gset.size * self.ng
        self.dim_qubits = self.ng * self.nq
        self.dim_output = dim_output = self.dim_gindices + self.dim_qubits
        self.dim_input = dim_input = 2 ** dim_wavelet - 1

        dim_units = dim_hiddens.copy()
        dim_units.append(dim_output)
        dim_units.insert(0, dim_input)

        self.net = nn.Sequential(
            *sum([
                self.build_layer(din, dout, activation=(l < len(dim_hiddens)))
                for l, din, dout
                in zip(range(len(dim_units)+1), dim_units[:-1], dim_units[1:])
            ], [])
        )
    
    @staticmethod
    def build_layer(din, dout, activation=True):
        layer = [nn.Linear(din, dout)]
        if activation:
            layer.append(nn.ReLU())
        return layer
    
    def forward(self, x):
        feat = self.net(x)
        return feat
    
    def as_logps(self, logits):
        logits_gate = logits[..., :self.dim_gindices]
        logits_qbit = logits[..., self.dim_gindices:]
        logps_gate = nn.functional.log_softmax(logits_gate, dim=-1)
        logps_qbit = nn.functional.log_softmax(logits_qbit, dim=-1)
        return logps_gate, logps_qbit
    
