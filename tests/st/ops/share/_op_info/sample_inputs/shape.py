# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
# pylint: skip-file
"""
Custom sample input functions for operation testing.
This module provides sample input builders for reshape/flatten/repeat/split/broadcast/permute various ops, including
basic, extra, dynamic, and error case generators.
"""
import functools

import numpy as np
import torch
import random
import warnings
import math
import itertools
import mindspore as ms
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array
)
from tests.st.ops.share._op_info.sample_inputs.common import _small_large_values


# ====================================custom sample inputs functions=================================== #
def basic_sample_inputs_tensor_h(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        (1, 3),
        (3, 5),
        (5, 1),
    )
    for input_shape in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}",
        )
    # empty tensor
    x_shape = (2, 0)
    yield OpSampleInput(
        op_input=make_func(x_shape),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_tensor_h(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        (9, 7),
        (4, 15),
        (2, 512),
        (1024, 13),
    )

    for input_shape in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    # large mix small values
    values = _small_large_values(dtype)
    if values:
        input_np_array = np.random.choice(values, size=(5, 3))
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 4), t), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_tensor_h(op_info: OpInfo, dtype, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_inputs_unary_op_tensor_h(op_info:OpInfo, dtype=None, device=None, **kwargs):
    # input is not 2D tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.Tensor(shape=(1, 2, 3), dtype=ms.float32),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_error_input_not_2D",
        ),
        op_error_type=ValueError,
        op_error_info=f'input tensor not 2D is not supported by {op_info.name}.',
    )
