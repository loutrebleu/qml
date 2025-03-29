# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable


class Wavelet:

    def __init__(self):
        pass

    def get_pattern_applied_func(self, a: float, b: float) -> Callable:
        pass

    def calc_wavelet_params(self, dim):
        return sum([
            [
                (a, b)
                for b in np.arange(-1, 1, a)
            ]
            for a in 2 / 2 ** np.arange(dim)
        ], [])
    
    @staticmethod
    def get_wavelet_range(wavelet_param):
        a, b = wavelet_param
        return [b, a + b]


class Haar(Wavelet):

    def __init__(self):
        pass

    def get_pattern_applied_func(self, a: float, b: float, xs) -> Callable:
        shifted_xs = (xs - b) / a
        yneg = np.where((-1 <= shifted_xs) & (shifted_xs < 0), -1, 0)
        ypos = np.where((0 <= shifted_xs) & (shifted_xs < 1), 1, 0)
        return (yneg + ypos) / np.sqrt(a)


class WaveletTransform:

    def __init__(self, wavelet: Wavelet) -> None:
        self._wavelet = wavelet
    
    def transform(self, xs, ys, dim=4):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        wparams = self.generate_wavelet_params(dim)
        powers = np.asarray([
            np.mean(
                self._wavelet.get_pattern_applied_func(*wparam, xs) * ys
            )
            for wparam in wparams
        ])
        return powers

    @staticmethod
    def generate_wavelet_params(dim):
        return np.asarray(sum([
            [
                (a, b)
                for b in np.arange(-1, 1, a)
            ]
            for a in 2 / 2 ** np.arange(dim)
        ], []))
    
