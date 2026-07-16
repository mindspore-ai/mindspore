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
"""linalg sample input functions for operation testing.

This module provides linalg sample input builders for various ops, including
basic, extra, dynamic, and error case generators.
"""
import functools

import numpy as np
import torch
import mindspore as ms
from mindspore import mint, mutable
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo, ReductionOpInfo, \
    extra_reference_inputs_unary_op_common_func

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array, _tensor_to_discontiguous
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_extra_values
)

def basic_reference_inputs_tensor_adjoint_func(
        op_info: OpInfo,
        dtype,
        device=None,
        **kwargs
):
    """Yield typical shape cases for tensor.adjoint ops.

    Covers scalars, vectors, singleton dims, medium 2D/3D, and empty-dimension cases.

    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    """
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )
    # 0D, 1D adjoint invalid
    shapes = (
        (M, S),
        (S, S, L),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for input_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low,
                           high=op_info.input_high, exclude_value=op_info.input_exclude_value)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_args=(),
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )


def error_inputs_tensor_adjoint_func(
        op_info: OpInfo,
        dtype,
        device=None,
):
    """
    Generate error inputs for common op, inputs not support Tensor and not_support dtypes for op
    """
    # input is scalar
    if not op_info.name.startswith("Tensor"):
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=2,
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_error_python_scalar",
            ),
            op_error_type=TypeError,
            op_error_info=f'Python scalar input is not supported by {op_info.name}.',
        )
    # input dtype not supported; #0D, 1D adjoint invalid
    S = SMALL_DIM_SIZE
    if op_info.not_support_dtypes is not None:
        for input_dtype in op_info.not_support_dtypes:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_tensor(shape=(S, S), dtype=input_dtype, device=device),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dtype_{input_dtype}_error",
                ),
                op_error_type=(TypeError, RuntimeError),
                op_error_info=f'dtype error input is not supported for {op_info.name}',
            )
