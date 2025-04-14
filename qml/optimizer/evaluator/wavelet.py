# -*- coding: utf-8 -*-

import numpy as np

from numpy.typing import NDArray
from typing import Callable

from .base import Evaluator, EvalResult
from .error import ErrorEvaluator
from ...model.model import Model
from ...tools.dataset import Dataset


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



class WaveletEvalResult(EvalResult):

    def __init__(self, errors: NDArray, powers: NDArray, xs: NDArray, ys: NDArray):
        super().__init__(xs, ys)
        self._es = np.asarray(errors)
        self._ps = np.asarray(powers)
    
    @property
    def es(self):
        return self._es.copy()
    
    @property
    def errors(self):
        return self.es
    
    @property
    def mse(self):
        return np.mean(self.errors ** 2) * 0.5
    
    @property
    def ps(self):
        return self._ps.copy()
    
    @property
    def powers(self):
        return self.ps


class WaveletEvaluator(Evaluator):

    def __init__(
            self,
            wavelet: Wavelet,
            dataset: Dataset,
            model: Model = None,
            wavelet_dim: int = 4,
            batch_size: int = None,
            shots: int = 50,
            seed: int = None,
            raise_iteration_error: bool = None,
    ):
        super().__init__(dataset, model, batch_size, shots, seed, raise_iteration_error)
        self._wavelet = wavelet
        self._wtrans = WaveletTransform(wavelet)
        self._wdim = wavelet_dim
    
    def __call__(
            self,
            params: NDArray = None,
            model: Model = None
    ) -> EvalResult:
        if model is None:
            model = self.model
        return self.evaluate(
            self._wtrans, self._wdim, model, params, self.dataset.xs, self.dataset.ys, shots=self.shots
        )

    @classmethod
    def evaluate(
            cls,
            wtrans: WaveletTransform,
            wdim: int,
            model: Model,
            params: NDArray,
            xs: NDArray,
            ys: NDArray,
            shots: int = None,
    ):
        eres = ErrorEvaluator.evaluate(model, params, xs, ys, shots=shots)
        powers = wtrans.transform(xs, eres.errors, dim=wdim)
        return WaveletEvalResult(eres.errors, powers, xs, ys)
