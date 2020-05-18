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

"""Utils for verification config."""

import numpy as np

from . import keyword
from .other_util import select_from_config_tuple


def get_input_config(d):
    """
    Get input config.

    Args:
        d (tuple): Config item in form of ([2, 2], {'dtype': np.float32, 'scale': 1}).
    Returns:
        Tuple, (shape, dtype, scale).
    """
    s = select_from_config_tuple(d, 0, d)
    dtype = np.float32
    scale = 1
    if isinstance(d, tuple) and isinstance(d[-1], dict):
        ext_config = d[-1]
        dtype = ext_config.get(keyword.dtype, np.float32)
        scale = ext_config.get(keyword.scale, 1)
    return s, dtype, scale


def get_expect_config(d):
    """
    Get input config.

    Args:
        d (tuple): Config item in form of (file_path, {'dtype': np.float32,
                   'scale': 1, 'max_error': 1e-3, 'check_tolerance': False, 'relative_tolerance': 0.0,
                   'absolute_tolerance': 0.0}).
    Returns:
        Tuple, (file_path, dtype, scale, max_error, check_tolerance, relative_tolerance, absolute_tolerance).
    """
    s = select_from_config_tuple(d, 0, d)
    dtype = np.float32
    scale = 1
    max_error = 1e-3
    check_tolerance = False
    relative_tolerance = 0.0
    absolute_tolerance = 0.0
    if isinstance(d, tuple) and isinstance(d[-1], dict):
        ext_config = d[-1]
        dtype = ext_config.get(keyword.dtype, np.float32)
        scale = ext_config.get(keyword.scale, 1)
        max_error = ext_config.get(keyword.max_error, 1e-3)
        check_tolerance = ext_config.get(keyword.check_tolerance, False)
        relative_tolerance = ext_config.get(keyword.relative_tolerance, 0.0)
        absolute_tolerance = ext_config.get(keyword.absolute_tolerance, 0.0)
    return s, dtype, scale, max_error, check_tolerance, relative_tolerance, absolute_tolerance


def get_function_config(function):
    """
    Get input config.

    Args:
        function (dict): Config item in form of {'delta': 1e-3, 'max_error': 1e-3, 'input_selector': [0, 1],
                         'output_selector': 0, 'sampling_times': 10, 'reduce_output': True, 'init_param_with': None,
                         'split_outputs': True, 'exception': Exception}.
    Returns:
        Tuple, (delta, max_error, input_selector, output_selector, sampling_times,
                reduce_output, init_param_with, split_outputs, exception).
    """
    delta = function.get(keyword.delta, 1e-3)
    max_error = function.get(keyword.max_error, 1e-3)
    input_selector = function.get(keyword.input_selector, [])
    output_selector = function.get(keyword.output_selector, [])
    sampling_times = function.get(keyword.sampling_times, -1)
    reduce_output = function.get(keyword.reduce_output, True)
    init_param_with = function.get(keyword.init_param_with, None)
    split_outputs = function.get(keyword.split_outputs, True)
    exception = function.get(keyword.exception, Exception)
    error_keywords = function.get(keyword.error_keywords, None)
    return delta, max_error, input_selector, output_selector, sampling_times, \
           reduce_output, init_param_with, split_outputs, exception, error_keywords


def get_grad_checking_options(function, inputs):
    """
    Get input config.

    Args:
        function (dict): Config item in form of {'block': XCell, 'delta': 1e-3, 'max_error': 1e-3, 'input_selector':
                         [0, 1], 'output_selector': 0, 'sampling_times': 10, 'reduce_output': True,
                         'init_param_with': None, 'split_outputs': True, 'exception': Exception}.
        inputs (dict): Config item in form of {'desc_inputs': [[2, 2]]}.
    Returns:
        Tuple, (f, args, delta, max_error, input_selector, output_selector, sampling_times, reduce_output).
    """
    f = function[keyword.block]
    args = inputs[keyword.desc_inputs]
    delta, max_error, input_selector, output_selector, sampling_times, reduce_output, _, _, _, _ = \
        get_function_config(function)
    return f, args, delta, max_error, input_selector, output_selector, sampling_times, reduce_output
