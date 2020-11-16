# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Modules to generate perturbations."""

import numpy as np
from scipy.ndimage.filters import gaussian_filter

_Array = np.ndarray

__all__ = [
    'BaseReplacement',
    'Constant',
    'GaussianBlur',
    'RandomPerturb',
]


class BaseReplacement:
    """
    Base class of generator for generating different replacement for perturbations.

    Args:
        kwargs: Optional args for generating replacement. Derived class need to
            add necessary arg names and default value to '_necessary_args'.
            If the argument has no default value, the value should be set to
            'EMPTY' to mark the required args. Initializing an object will
            check the given kwargs w.r.t '_necessary_args'.

    Raise:
        ValueError: Raise when provided kwargs not contain necessary arg names with 'EMPTY' mark.
    """
    _necessary_args = {}

    def __init__(self, **kwargs):
        self._replace_args = self._necessary_args.copy()
        for key, value in self._replace_args.items():
            if key in kwargs.keys():
                self._replace_args[key] = kwargs[key]
            elif key not in kwargs.keys() and value == 'EMPTY':
                raise ValueError(f"Missing keyword arg {key} for {self.__class__.__name__}.")

    def __call__(self, inputs):
        raise NotImplementedError()


class Constant(BaseReplacement):
    """Generator to provide constant-value replacement for perturbations."""
    _necessary_args = {'base_value': 'EMPTY'}

    def __call__(self, inputs: _Array) -> _Array:
        replacement = np.ones_like(inputs, dtype=np.float32)
        replacement *= self._replace_args['base_value']
        return replacement


class GaussianBlur(BaseReplacement):
    """Generator to provided gaussian blurred inputs for perturbation"""
    _necessary_args = {'sigma': 0.7}

    def __call__(self, inputs: _Array) -> _Array:
        sigma = self._replace_args['sigma']
        replacement = gaussian_filter(inputs, sigma=sigma)
        return replacement


class RandomPerturb(BaseReplacement):
    """Generator to provide replacement by randomly adding noise."""
    _necessary_args = {'radius': 0.2}

    def __call__(self, inputs: _Array) -> _Array:
        radius = self._replace_args['radius']
        outputs = inputs + (2 * np.random.rand(*inputs.shape) - 1) * radius
        return outputs
