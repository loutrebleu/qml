# -*- coding: utf-8 -*-

import xsim
import time
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
from IPython.display import clear_output
from qml import optimizer as xoptim
from qml.model.gate import Gateset
from qml.db import dpo as xdpo
from qml.db import target as xtarget
from qml.db.ml import MLDatasetGenerator
from qml.db.dpo.decoder import DPODataDecoder, DPOData
from qml.db.dpo.loader import DPODataBatch
from qml.model.model import Model
from qml.optimizer import dpo as xdpopt
from qml.optimizer import evaluator as xeval
from qml.tools.validation import validate_old
from qml.tools.random import XRandomGenerator
from qml.tools.sampler import CandidateSampler
from qml.tools.config import Config
# from qml.tools.validation import validate
from qml.tools.validation import get_base_qc
from qml.tools.random import XRandomGenerator

Loss = namedtuple("Loss", ["total", "dpo", "llh"])


def hard_copy(target, source):
    target.net.load_state_dict(source.net.state_dict())


def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)



def prepare_policy(cf):
    policy = xdpopt.Policy(
        cf.ocg.dim_wavelet,
        cf.nq,
        cf.ng,
        cf.ocg.policy.dim_hiddens,
    )
    reference = xdpopt.Policy(
        cf.ocg.dim_wavelet,
        cf.nq,
        cf.ng,
        cf.ocg.policy.dim_hiddens,
    )

    hard_copy(reference, policy)
    optimizer = optim.Adam(policy.parameters(), lr=cf.dpo.training.lr)
    sampler = CandidateSampler(policy)
    return policy, reference, optimizer, sampler


def calc_logits(logits, sampler, cf):
    logits_gate, logits_qbit = sampler.divide_gate_and_qubit(logits)
    logits_gate = logits_gate.view(len(logits), cf.ng, -1)
    logits_qbit = logits_qbit.view(len(logits), cf.ng, -1)
    return logits_gate, logits_qbit


def select_logps(logits, onehot):
    logps = nn.functional.log_softmax(logits, dim=-1).unsqueeze(dim=1)
    selected_logps = (logps * onehot).sum(dim=-1).sum(dim=-1)
    return selected_logps


def calc_logps_(logits, subbatch):
    logits_gate, logits_qbit = logits
    logps_gate = select_logps(logits_gate, subbatch.onehot_gindices)
    logps_qbit = select_logps(logits_qbit, subbatch.onehot_qubits)
    logps = (logps_gate + logps_qbit).mean(dim=-1)
    return logps


def calc_logps(policy, batch, sampler, cf):
    logits = policy(batch.wserieses)
    logits = calc_logits(logits, sampler, cf)
    logps_best = calc_logps_(logits, batch.best)
    logps_others = calc_logps_(logits, batch.others)
    return logps_best, logps_others


def calc_loss_dpo(logps_pol_best, logps_pol_others, logps_ref_best, logps_ref_others, cf):
    dlogps_pol = logps_pol_best - logps_pol_others
    dlogps_ref = logps_ref_best - logps_ref_others
    ddlogps = dlogps_pol - dlogps_ref
    loss_dpo = -1 * nn.functional.logsigmoid(cf.dpo.training.beta * ddlogps).mean()
    return loss_dpo


def calc_loss_llh(logps_best):
    return -1 * logps_best.mean()


def train_once(policy, reference, optimizer, batch, sampler, cf):
    logps_pol_best, logps_pol_others = calc_logps(policy, batch, sampler, cf)
    with torch.no_grad():
        logps_ref_best, logps_ref_others = calc_logps(reference, batch, sampler, cf)
    loss_dpo = calc_loss_dpo(logps_pol_best, logps_pol_others, logps_ref_best, logps_ref_others, cf)
    loss = loss_dpo

    if cf.dpo.training.cpo:
        loss_llh = 0.1 * calc_loss_llh(logps_pol_best)
        loss = loss_dpo + loss_llh
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if cf.dpo.training.cpo:
        return Loss(loss.item(), loss_dpo.item(), loss_llh.item())
    else:
        return Loss(loss.item(), loss_dpo.item(), 0)


def validate(sampler, datasets, cf):
    time_avons = time.time()
    vresults = validate_old(
        sampler,
        datasets,
        cf.nq,
        cf.dpo.validation.num_rounds,
        cf.qml.num_train,
        cf.ocg.dim_wavelet,
        cf.wavelet,
        cf.dpo.validation.reg_loss,
        shots=cf.shots,
    )
    time_apres = time.time()
    time_mean = (time_apres - time_avons) / len(datasets)
    return dict(loss=vresults, time=time_mean)


def plot_results(logger, vlogger, cf):
    ret = xsim.Retriever(logger)
    res_dict = dict(
        epoch=ret.epoch(),
        loss=ret.loss(),
        loss_dpo=ret.loss_dpo(),
    )
    plot_labels = ["loss_dpo"]
    if cf.dpo.training.cpo:
        res_dict["loss_llh"] = ret.loss_llh()
        plot_labels.append("loss_llh")
    plot_labels.append("loss")
    res = pd.DataFrame(res_dict)

    vret = xsim.Retriever(vlogger)
    vres = pd.DataFrame(dict(
        epoch=vret.epoch(),
        loss=vret.loss(),
    ))

    clear_output()
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)
    res.plot(x="epoch", y=plot_labels, ax=axes[0])
    vres.plot(x="epoch", y=["loss"], ax=axes[1])
    plt.show()


def generate_batch(sampler, cf, seed=None):
    rng = XRandomGenerator(seed)
    gset = Gateset.set_num_qubits(cf.nq)
    # validation datasets
    tfun = xtarget.PolynominalTargetFunctionGenerator(cf.qml.db.dim_polynomial, seed=rng.new_seed())
    tgen = MLDatasetGenerator(tfun, seed=rng.new_seed())
    Dqml = tgen.generate(cf.qml.db.size)
    # Dqml = tgen.generate(5)

    base_model = get_base_qc(cf.nq, cf.qml.db.dim_input, cf.qml.db.dim_output, cf.shots)
    weval = xeval.WaveletEvaluator(cf.wavelet, Dqml, wavelet_dim=cf.ocg.dim_wavelet)

    dataset = []
    for round in range(1, cf.dpo.validation.num_rounds+1):
        # 1. measure the wavelet series
        vresult = weval(base_model.trainable_parameters, base_model)

        # 2. estimate the candidate unit
        wseries = vresult.powers
        candidates = [sampler.sample(wseries) for _ in range(3)]
        base_model.fix_trainable_units()

        losses = []
        for candidate in tqdm(candidates):
            model = Model(
                base_model.nq, base_model.nc,
                base_model.input_units,
                base_model.fixed_units,
                candidate,
            )

            # 4. train the model
            optimizer = xoptim.LocalSearchOptimizer(Dqml)
            tresult = optimizer.optimize(model, cf.qml.num_train, verbose=False)
            # tresult = optimizer.optimize(model, 10, verbose=False)
            losses.append(float(tresult.first.energy))

        # 5. store data
        jdata = json.dumps(dict(
            wseries=wseries.tolist(),
            units=[candidate[0].to_dict() for candidate in candidates],
            losses=losses,
        ))
        dataset.append(DPODataDecoder.from_json(jdata, gset, dim_wavelet=cf.ocg.dim_wavelet))

        # 6. update model
        # 6-1. select winner
        idx_best = np.argmin(losses)
        candidate_best = candidates[idx_best]

        # 6-2. update model
        base_model = Model(
            base_model.nq, base_model.nc,
            base_model.input_units,
            base_model.fixed_units,
            candidate_best,
        )

    # 7. generate batch
    batch_data = DPOData(
        [data.wseries for data in dataset],
        [data.gindices for data in dataset],
        [data.qubits for data in dataset],
        [data.losses for data in dataset],
    )
    return DPODataBatch(batch_data, cf.nq, gset.size)