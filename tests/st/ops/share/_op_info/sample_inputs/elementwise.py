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
This module provides sample input builders for unary/binary/logical/clamp various ops, including
basic, extra, dynamic, and error case generators.
"""
import functools

import mindspore as ms
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_info import (
    extra_reference_inputs_binary_op_common_func,
    _generate_binary_op_tensors_sample_inputs_func,
    _generate_binary_op_small_value_tensor_inputs_func,
    _generate_binary_op_large_value_tensor_inputs_func,
    _generate_binary_op_scalar_inputs_func,
    _generate_binary_op_extremal_value_tensor_inputs_func,
)

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpErrorInput, make_tensor,
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
)
from tests.st.ops.share._op_info.sample_inputs.common import (
    _generate_binary_inplace_op_broadcasting_and_discontiguous_tensor_inputs_func,
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

def error_inputs_binary_op_tensor_add_common(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    # other is tuple or str
    test_cases = ((1, 2), "False")
    for other in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=(1, 2), dtype=ms.float32),
                op_args=(other,),
                op_kwargs={},
                sample_name=f"{op_info.name}_other_error",
            ),
            op_error_type=TypeError,
            op_error_info=f'other {other} is not supported for {op_info.name}',
        )
    # alpha is float, but input/other is not float
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(5,), dtype=ms.int32),
            op_args=(make_func(shape=(1,), dtype=ms.int32),),
            op_kwargs={'alpha': 2.5},
            sample_name=f"{op_info.name}_input_other_dtype_error",
        ),
        op_error_type=(TypeError,RuntimeError),
        op_error_info=f'alpha is float, but input and other is not float is not supported for {op_info.name}',
    )

def error_inputs_binary_op_tensor_add(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    yield from error_inputs_binary_op_tensor_add_common(op_info, dtype, device, **kwargs)
    # alpha is bool, but input/other is not bool
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(5,), dtype=ms.float16),
            op_args=(make_func(shape=(1,), dtype=ms.float16),),
            op_kwargs={'alpha': False},
            sample_name=f"{op_info.name}_input_other_dtype_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'alpha is bool, but input and other is not bool is not supported for {op_info.name}',
    )

def error_inputs_binary_op_tensor_add_(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    yield from error_inputs_binary_op_tensor_add_common(op_info, dtype, device, **kwargs)
    cases = (
        # output with shape doesn't match the broadcast shape
        ((1,), (5,), ms.int32, ms.int32),
        # promote dtype complex can't be cast to the desired output type int
        ((5,), (1,), ms.int16, ms.complex64),
    )
    for input_shape, other_shape, input_dtype, other_dtype in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=input_shape, dtype=input_dtype),
                op_args=(make_func(shape=other_shape, dtype=other_dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_shape_dtype_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'not match broadcast shape, complex to int is not supported for {op_info.name}',
        )

def extra_sample_inputs_tensor_add_(op_info:OpInfo, dtype=None, device=None, **kwargs):
    # tensors with many kinds of shapes
    yield from _generate_binary_op_tensors_sample_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with small value
    if not op_info.disable_small_value_tensor_inputs and dtype != ms.bool_:
        yield from _generate_binary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if not op_info.disable_large_value_tensor_inputs and dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_binary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # broadcasting tensors and contiguous or discontiguous tensors
    if not op_info.disable_broadcasting_and_discontiguous_tensor_inputs:
        yield from _generate_binary_inplace_op_broadcasting_and_discontiguous_tensor_inputs_func(
            op_info, dtype, device, **kwargs)
    # scalarxtensor, tensorxscalar and scalarxscalar
    if not op_info.disable_scalar_inputs:
        yield from _generate_binary_op_scalar_inputs_func(op_info, dtype, device, **kwargs)
    # tensor with extremal value
    if not op_info.disable_extremal_value_tensor_inputs and (dtype.is_floating_point or dtype.is_complex):
        yield from _generate_binary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)

    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((1,), ()),
        ((5, 1, 1, 1, 8, 7), (5, 1, 1, 1, 8, 7)),
        ((5, 5, 11, 2, 5, 4, 5, 2), (5, 5, 1, 2, 5, 4, 5, 2)),
    )

    for input_shape, other_shape in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(make_func(shape=other_shape),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    # other is float or bool
    if dtype.is_floating_point or dtype.is_complex:
        others = (0.4, False)
    elif dtype is ms.bool_:
        others = (True, False)
    else:
        others = (4, False)
    for other in others:
        yield OpSampleInput(
            op_input=make_func(shape=(11, 20)),
            op_args=(other,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{other}_{dtype}_extra",
        )

# sample inputs functions for Tensor.add(other)
def basic_sample_inputs_tensor_add_overload(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for Tensor.add(other).
    Reference torch's sample_inputs_add_sub and basic_reference_inputs_binary_op_common_func.
    '''
    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((), (1,)),
        ((50,), (1,)),
        ((S, S), (S, S)),
        ((2, 0), (1,)),
    )
    if dtype is ms.bool_:
        other_dtype = ms.float32
    else:
        other_dtype = dtype
    for input_shape, other_shape in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input_shape),
            op_args=(make_arg(shape=other_shape, dtype=other_dtype),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_basic"
        )

def extra_sample_inputs_tensor_add_overload(op_info:OpInfo, dtype=None, device=None, **kwargs):
    yield from extra_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((1, 13, 50),(2, 13, 50)),
        ((5, 1, 1, 1, 8, 7), (5, 1, 1, 1, 8, 7)),
        ((5, 5, 11, 2, 5, 4, 5, 2), (5, 5, 1, 2, 5, 4, 5, 2)),
    )

    for input_shape, other_shape in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(make_func(shape=other_shape),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    # other is float or bool
    others = (0.4, False)
    for other in others:
        yield OpSampleInput(
            op_input=make_func(shape=(11, 20)),
            op_args=(other,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{other}_{dtype}_extra",
        )

def error_inputs_binary_op_tensor_add_overload(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # other is tuple or list, case can pass
    # other is str
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(1, 2)),
            op_args=("False",),
            op_kwargs={},
            sample_name=f"{op_info.name}_other_is_str_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'other is str is not supported for {op_info.name}',
    )
