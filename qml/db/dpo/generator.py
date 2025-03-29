# -*- coding: utf-8 -*-

import numpy as np
import json
from tqdm import tqdm

from qml.db import target as xtarget
from qml.db.ml import MLDatasetGenerator
from qml.tools.random import XRandomGenerator


# step 0
from qml.db.ml import MLDatasetGenerator

# step 1
from qml.model.encoding import EncodingUnitManager
from qml.model.model import Model

# step 2
from qml.optimizer import evaluator as xeval

# step 3
from qml.model.unit import UnitManager

# step 4: train candidates
from qml import optimizer as xoptim



class DPODatasetGenerator:

    WAVELET_CLASS = xeval.Haar

    def __init__(
            self,
            num_qubits: int,
            num_gates: int,
            max_order: int,
            max_repeat_for_target: int = 3,
            # step 0
            qml_db_size: int = 50,
            db_filename = "dpo_databsae.txt",
            # step 1
            # step 2
            wavelet_class: xeval.Wavelet = None,
            wavelet_dim: int = 4,
            # step 3
            num_candidates: int = 5,
            # step 4
            qml_max_iter: int = 50,
            qml_batch_size: int = None,
            seed: int = None,
    ) -> None:
        self.rng = rng = XRandomGenerator(seed)
        self.num_qubits = nq = num_qubits
        self.num_gates = ng = num_gates
        self.max_repeat_for_target = max_repeat_for_target

        # step 0 target function
        self.max_order = d = max_order
        self.tfun = xtarget.PolynominalTargetFunctionGenerator(d, rng.new_seed())
        self.tgen = MLDatasetGenerator(self.tfun, rng.new_seed())
        self.qml_db_size = qml_db_size
        self.db_filename = db_filename

        # step 1 prepare mqc

        # step 2 calc wavelet series
        if wavelet_class is None:
            wavelet_class = self.WAVELET_CLASS
        self.wavelet_class = wavelet_class
        self.wavelet_dim = wavelet_dim
        
        # step 3
        self.uman = UnitManager(nq, ng, rng.new_seed())
        self.num_candidates = num_candidates

        # step 4 tain candidate
        self.qml_max_iter = qml_max_iter
        self.qml_batch_size = qml_batch_size
    
    def initialize_qml_and_model(self):
        # step 0 generate qml database
        qml_dataset = self.tgen.generate(self.qml_db_size)

        # step 1 prepare mqc
        eunit = EncodingUnitManager.AngleEncoding(1, self.num_qubits, repeat=True)
        model = Model(self.num_qubits, 1, eunit, [], [])
        
        return qml_dataset, model
    
    def generate(
            self,
            size: int,
            db_filename: str = None,
            batch_size: int = None,
    ):
        if db_filename is None:
            db_filename = self.db_filename
        batch_size = self.qml_batch_size if batch_size is None else batch_size

        num_data = 0
        
        while True:
            print("num_data:", num_data)

            # step 0. generate qml database with target func
            # step 1. prepare mqc
            qml_dataset, model = self.initialize_qml_and_model()

            for _ in range(self.max_repeat_for_target):
                print(f"round:{num_data:>4d}", end="\t")
                # step 2. calc wavelet seriese
                wave_evaluator = xeval.WaveletEvaluator(self.wavelet_class(), qml_dataset, model, wavelet_dim=self.wavelet_dim)
                wave_result = wave_evaluator()

                model.fix_trainable_units()

                loss_results = []
                candidate_units = []
                candidate_models = []
                for _ in tqdm(range(self.num_candidates)):
                    # step 3. append candidate units
                    candidate_unit = self.uman.generate_random_unit()
                    candidate_units.append(candidate_unit)
                    candidate_model = Model(model.nq, 1, model.input_units, model.fixed_units, candidate_unit)
                    candidate_models.append(candidate_model)

                    # step 4. train candidates
                    train_optimizer = xoptim.LocalSearchOptimizer(qml_dataset)
                    train_result = train_optimizer.optimize(
                        candidate_model,
                        self.qml_max_iter,
                        batch_size=batch_size,
                        verbose=False
                    )

                    # step 5. calc loss after training
                    loss_evaluator = xeval.MSEEvaluator(qml_dataset, candidate_model)
                    loss_result = loss_evaluator(train_result.first.x, candidate_model)
                    loss_results.append(loss_result)

                # step 6. store as db
                data = dict(
                    wseries=wave_result.powers.tolist(),
                    units=[candidate_unit.to_dict() for candidate_unit  in candidate_units],
                    losses=[float(loss_result.loss) for loss_result in loss_results],
                )
                print("[info] saving data into:", db_filename)
                with open(db_filename, mode="a") as f:
                    print(json.dumps(data), file=f)
                num_data += 1

                # check generated data size
                if num_data >= size:
                    return db_filename

                # if trained model reaches near-optimal --> create new problem set
                if np.min(data["losses"]) <= 1e-3:
                    break
                
                # update current model
                best_candidate_no = np.argmin(data["losses"])
                model = candidate_models[best_candidate_no]
