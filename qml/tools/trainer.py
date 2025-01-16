# -*- coding: utf-8 -*-

import numpy as np


def calc_gradients(model, x, shots=100):
    trainable_params = model.trainable_parameters
    tp_shapes = [len(tp) for tp in trainable_params]
    tp_shapes.insert(0, 0)
    tp_shape_idxs = np.cumsum(tp_shapes)

    trainable_params = np.hstack(trainable_params)
    demi_pi = np.pi / 2
    deux_pi = np.pi * 2

    def deflatten(flattened):
        return [
            flattened[idx_de:idx_to]
            for idx_de, idx_to
            in zip(tp_shape_idxs[:-1], tp_shape_idxs[1:])
        ]

    def calc_gradient_idx(idx):
        shifted_pos = trainable_params.copy()
        shifted_neg = trainable_params.copy()
        shifted_pos[idx] = (trainable_params[idx] + demi_pi) % deux_pi
        shifted_neg[idx] = (trainable_params[idx] - demi_pi) % demi_pi

        predict_pos = model.forward(
            x,
            params=deflatten(shifted_pos),
            shots=shots
        )
        predict_neg = model.forward(
            x,
            params=deflatten(shifted_neg),
            shots=shots
        )
        grad = (predict_pos - predict_neg) / 2
        return grad

    grads = np.asarray([
        calc_gradient_idx(idx)
        for idx in range(len(trainable_params))
    ])

    return deflatten(grads)
