# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from qml.model.model import Model
from qml.model.encoding import EncodingUnitManager
from qml.tools.dataset import Dataset
from qml.optimizer import evaluator as xeval
from qml import optimizer as xoptim


def get_base_qc(num_qubits, dim_input, dim_output, shots):
    return Model(
        num_qubits, dim_output,
        EncodingUnitManager.AngleEncoding(dim_input, num_qubits, repeat=True),
        [], [], shots=shots
    )


def validate_old(sampler, datasets, num_qubits, num_rounds, num_train_steps, dim_wavelet, wavelet, reg_loss, shots=50):

    losses = []
    
    for dataset in tqdm(datasets):
        model = get_base_qc(num_qubits, dataset.dim_input, dataset.dim_output, shots)
        veval = xeval.WaveletEvaluator(wavelet, dataset, wavelet_dim=dim_wavelet)

        losses_vdb = []

        for round in range(1, num_rounds+1):
            # 1. measure the wavelet series
            vresult = veval(model.trainable_parameters, model)

            # 2. estimate the candidate unit
            wseries = vresult.powers
            candidate = sampler.sample(wseries)
            
            # 3. update the trainable unit
            model.fix_trainable_units()
            model = Model(
                model.nq, model.nc,
                model.input_units,
                model.fixed_units,
                candidate,
                shots=model.shots,
            )

            # 4. train the model
            optimizer = xoptim.LocalSearchOptimizer(dataset)
            tresult = optimizer.optimize(model, num_train_steps, verbose=False)

            # 5. update the model parameters
            model.update_parameters(tresult.first.x)

            # logging
            losses_vdb.append(vresult.mse)
            if vresult.mse < reg_loss:
                break
        losses.append(losses_vdb)

    return np.mean(losses)


def validate(sampler, datasets, cf):
    return validate_old(
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