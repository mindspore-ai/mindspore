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
import random
import warnings
import itertools
import mindspore as ms
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_info import (
    _generate_unary_op_large_value_tensor_inputs_func,
    _generate_unary_op_small_value_tensor_inputs_func,
)
from tests.st.ops.share._op_info.sample_inputs.common import make_extra_tensor

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array, _tensor_to_discontiguous
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_extra_uint,
    dtype_to_np_dtype,
    dtypes_extra_values,
    small_large_values
)

def sample_inputs_matmul(op_info, dtype, device, is_rmatmul=False, **kwargs):
    S = SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    test_cases = (((L,), (L,)),
                  ((S, M), (M,)),
                  ((M,), (M, S)),
                  ((S, M), (M, S)),
                  ((S, 0), (0, M)),
                  ((S, S, M), (M,)),
                  ((S, S, M), (M, S)),
                  ((S, S, 0), (0, S)),
                  ((M,), (S, M, S)),
                  ((S, M), (S, M, S)),
                  ((0, 0), (S, 0, 0)),
                  ((S, S, M, M), (S, S, M, S)),
                  ((S, S, M, M), (M,)),
                  ((M,), (S, S, M, S)),
                  ((S, S, S), (1, S, S))
                  )
    for lhs_shape, rhs_shape in test_cases:
        lhs = make_arg(lhs_shape)
        rhs = make_arg(rhs_shape)
        if not is_rmatmul:
            yield OpSampleInput(lhs, op_args=(rhs,), sample_name=op_info.name)
        else:
            yield OpSampleInput(rhs, op_args=(lhs,), sample_name=op_info.name)

def extra_sample_inputs_mint_matmul(op_info, dtype=None, device=None, is_rmatmul=False, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    extremal_cases = [
     float('inf'),
     float('-inf'),
      ]
    for extremal_val in extremal_cases:
        lhs = make_arg((3,))
        lhs[2] = extremal_val
        rhs = make_arg((3,))
        if not is_rmatmul:
            yield OpSampleInput(lhs, op_args=(rhs,), sample_name=op_info.name)
        else:
            yield OpSampleInput(rhs, op_args=(lhs,), sample_name=op_info.name)

    #discontiguous_tensor
    yield OpSampleInput(
        op_input=make_arg((5, 3), discontiguous=True),
        op_args=(make_arg((3,), discontiguous=True),),
        sample_name=f"{op_info.name}_discontiguous_tensor_inputs",
        )

def error_inputs_mint_matmul(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.float32),
            op_args=((1,),),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='other is not tensor',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=True,
            op_args=make_tensor(shape=(2,), dtype=ms.float32),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='input is not tensor',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.float64),
            op_args=make_tensor(shape=(3,), dtype=ms.float32),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='input and other dtype must the same',
    )


def basic_sample_inputs_mint_linalg_matrix_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )
    cases = [
        ((0, 2, 4), {'ord': 'fro', 'dim': (-2, -1), 'keepdim': False, 'dtype': dtype},),
        ((2, 4, 3), {'ord': float('inf'), 'dim': (0, 1), 'keepdim': False, 'dtype': dtype},),
        ((3, 5), {'ord': 'nuc', 'dim': (0, 1), 'keepdim': True, 'dtype': dtype},),
        ((1, 1, 2, 4), {'ord': float('-inf'), 'dim': (1, 2), 'keepdim': False, 'dtype': dtype},),
        ((2, 3, 4, 5, 6), {'ord': 1, 'dim': (1, 2), 'keepdim': False, 'dtype': dtype},),
        ((2, 1), {'ord': -1, 'dim': (-2, -1), 'keepdim': False, 'dtype': dtype},),
        ((0, 5, 7), {'ord': 2, 'dim': (-2, -1), 'keepdim': True, 'dtype': dtype},),
        ((2, 3, 7, 7), {'ord': -2, 'dim': (1, 2), 'keepdim': True, 'dtype': dtype},),
    ]
    for input_shape, p_kwargs in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

def error_inputs_mint_linalg_matrix_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32, )
    cases = [
        (1, {'ord': 'fro', 'dim': (-2, -1), 'keepdim': False, 'dtype': ms.float32},),
        (make_func(shape=(2, 3, 4)), {'ord': 'LESS', 'dim': (0, 1), 'keepdim': False, 'dtype': ms.float32},),#ord value error
        #(make_func(shape=(3, 5)), {'ord': True, 'dim': (0, 1), 'keepdim': True, 'dtype': ms.float32},),#ord type error, not raise error
        (make_func(shape=(3, 5)), {'ord': 0, 'dim': (0, 1), 'keepdim': True, 'dtype': ms.float32},),  # ord value 0
        (make_func(shape=(1, 1, 2, 4)), {'ord': float('-inf'), 'dim': "(1, 2)", 'keepdim': False, 'dtype': ms.float32},),#dim type error
        (make_func(shape=(2, 3, 4, 5, 6)), {'ord': 1, 'dim': (1, 2), 'keepdim': 5, 'dtype': ms.float32},),#keepdim type error
        (make_func(shape=(2, 1)), {'ord': -1, 'dim': (2, 3), 'keepdim': False, 'dtype': ms.float32},),#dim value outer error
        (make_func(shape=()), {'ord': -1, 'dim': (2, 3), 'keepdim': False, 'dtype': ms.float32},),
    ]
    for input_shape, p_kwargs  in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(),
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError, AttributeError),
            op_error_info='other ops mint.tril error values',
        )

def extra_sample_inputs_mint_linalg_matrix_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )
    cases = [
        ## discontinuous
        (ms.Tensor(np.random.randn(6, 8)[::2, ::2], dtype=dtype), {'ord': float('inf'), 'dim': (-2, -1),
                                                                   'keepdim': True, 'dtype': dtype},),
        (make_func(shape=(2, 3, 5, 7, 9, 4)), {'ord': 1, 'dim': (1, 2), 'keepdim': False, 'dtype': dtype},),
    ]
    for input_shape, p_kwargs in cases:
        yield OpSampleInput(
            op_input=input_shape,
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

    # small/large values
    values = []
    if dtype == ms.uint8:
        values = dtypes_extra_values['_uint_small_values']
    elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
        values = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    elif dtype.is_floating_point:
        if dtype in [ms.float16]:
            values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp16_large_values']
        else:
            values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp32_large_values']
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_args=(),
                op_kwargs={'ord': float('inf'), 'dim': (-2, -1), 'keepdim': False, 'dtype': dtype},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(),
                    op_kwargs={'ord': float('inf'), 'dim': (-2, -1), 'keepdim': False, 'dtype': dtype},
                    sample_name=op_info.name,
                )

def basic_sample_inputs_mint_linalg_vector_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )
    cases = [
        ((), {'ord': 2, 'dim': None, 'keepdim': False, 'dtype': dtype},),
        ((2, 4), {'ord': float('inf'), 'dim': (0, 1), 'keepdim': True, 'dtype': dtype},),
        ((3, 5, 7), {'ord': float('-inf'), 'dim': (-1, 1), 'keepdim': True, 'dtype': dtype},),
        ((1, 1, 2, 4), {'ord': 0, 'dim': (2,), 'keepdim': False, 'dtype': dtype},),
        ((2, 3, 4, 5, 6), {'ord': 3.0, 'dim': (1, 2), 'keepdim': False, 'dtype': dtype},),#other int/or float
    ]
    for input_shape, p_kwargs in cases:
        current_ord = p_kwargs.get('ord', 'default')
        if current_ord == float('inf'):
            ord_str = 'inf'
        elif current_ord == float('-inf'):
            ord_str = '-inf'
        else:
            ord_str = str(current_ord)
        dynamic_name = f"{op_info.name}_ord_{ord_str}"
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=dynamic_name,
        )

def error_inputs_mint_linalg_vector_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32, )
    cases = [
        (1, {'ord': 2, 'dim': None, 'keepdim': False, 'dtype': dtype},),# x not tensor
        (make_func(shape=(2, 3, 4)), {'ord': 'LESS', 'dim': (0, 1), 'keepdim': False, 'dtype': dtype},),#ord value error
        (make_func(shape=(3, 5)), {'ord': True, 'dim': (-1, 1), 'keepdim': True, 'dtype': dtype},),#ord type error
        (make_func(shape=(1, 1, 2, 4)), {'ord': float('-inf'), 'dim': "(1, 2)", 'keepdim': False, 'dtype': dtype},),#dim type error
        (make_func(shape=(2, 3, 4, 5, 6)), {'ord': 1, 'dim': (1, 2), 'keepdim': 5, 'dtype': dtype},),#keepdim type error
        (make_func(shape=(2, 1)), {'ord': -1, 'dim': (2, 3), 'keepdim': False, 'dtype': dtype},),#dim value outer error
    ]
    for input_shape, p_kwargs in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(),
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.tril error values',
        )

def basic_sample_inputs_mint_linalg_qr(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )#only support >= 2D
    cases = [
        ((0, 2, 4), {'mode': 'reduced'},),
        ((2, 3, 3), {'mode': 'complete'},),
        #((5, 2), {'mode': 'r'},),
        ((2, 3, 7, 7), {'mode': 'reduced'},),#default value
    ]
    for input_shape, p_kwargs in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

def error_inputs_mint_linalg_qr(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32, )  # only support >= 2D
    cases = [
        (make_func(shape=(0, 2, 4)), {'mode': 8},),#mode type error
        (1, {'mode': 'complete'},),#A not tensor
        (make_func(shape=(3,)), {'mode': 'r'},),#A 1D
        (make_func(shape=(2, 3, 7, 7)), {'mode': 'r122'}),#mode value error
    ]
    for input_shape, p_kwargs in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(),
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.tril error values',
        )

def extra_sample_inputs_mint_linalg_qr(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )
    cases = [
        # discontinuous
        (ms.Tensor(np.random.randn(6, 8)[::2, ::2], dtype=dtype), {'mode': 'complete'},),
        (make_func(shape=(2, 3, 5, 7, 9, 4)), {'mode': 'complete'},),
    ]
    for input_shape, p_kwargs in cases:
        yield OpSampleInput(
            op_input=input_shape,
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

    # small/large values
    values = small_large_values(dtype, exclude_values=(1e+20, -1e+20))
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_args=(),
                op_kwargs={'mode': 'complete'},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.nan]:#inf loss error
                input_values = np.full((1, 1, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(),
                    op_kwargs={'mode': 'complete'},
                    sample_name=op_info.name,
                )

def basic_sample_inputs_mint_linalg_inv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)#only support >= 2D,
    # the last two dim must be the same.
    cases = [
        (2, 2),
        (3, 3),
        (10, 10),
        (5, 4, 4),
        (2, 3, 4, 4),
        (2, 2, 2, 2, 3, 3),
        (1, 5, 5)
    ]
    for input_shape in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_mint_linalg_inv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32,)  # only support >= 2D,
    cases = [
        (2, 2, 2, 2, 2, 2, 2),# >6d
        (),#0d scalar
        (3, 2),#non invertible matrix
        (4, 5),
        (2, 3, 4),
        (5,),#1d<2d
    ]
    for input_shape in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=input_shape),
                op_args=(),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.linalg.inv error values',
        )

def extra_sample_inputs_mint_linalg_inv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    #only support >= 2D,
    # the last two dim must be the same.
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )
    cases = [
        # discontinuous
        ms.Tensor(np.random.randint(0, 255, (2, 1, 2, 2, 2)), dtype=dtype)[::2, ::2],
        #empty shape
        make_func(shape=(2, 0, 1, 1)),
        #inf/nan
        ms.Tensor([[1., 2.], [float('inf'), 4.]], ms.float32),
        ms.Tensor([[1., 2.], [float('-inf'), 4.]], ms.float32),
        #ms.Tensor([[1., 2.], [float('nan'), 4.]], ms.float32),#raise error
    ]
    for input_shape in cases:
        yield OpSampleInput(
            op_input=input_shape,
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # small/large values
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_func(shape=(2, 3, 4, 4)),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

def basic_sample_inputs_mint_tril(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor(np.random.randn(7, 3), dtype=dtype), 0),#default
        (ms.Tensor(np.random.randn(8, 9, 5), dtype=dtype), 6),
        (ms.Tensor(np.random.randn(4, 4, 7, 8, 9, 9, 4), dtype=dtype), 2),
        (ms.Tensor(np.random.randint(-128, 127, (7, 3, 3, 4, 4, 4, 8, 3)), dtype=ms.int8), -3),
    ]

    for sorted_sequence, diagonal in cases:
        yield OpSampleInput(
            op_input=sorted_sequence,
            op_args=(diagonal,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_triu(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate dynamic inputs for mint.triu ops.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpDynamicInput objects.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(1, None, None), dtype=dtype),
                op_args=(0,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 6, 5)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # add/sub_ext dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 3)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 6)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_inputs_mint_tril(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (0, 0),#input int
        (None, 0),# input int
        (ms.Tensor(np.random.randn(8, 9, 5), dtype=dtype), "6"),#diagonal not int
        (ms.Tensor(np.random.randn(4, 4, 7, 8, 9, 9, 4), dtype=dtype), [1, 2]),#diagonal list
        (ms.Tensor(np.random.randn(), dtype=dtype), -3)#input 0d, must >2d
    ]

    for sorted_sequence, diagonal in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=sorted_sequence,
                op_args=(diagonal,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.tril error values',
        )

def extra_sample_inputs_mint_tril(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        # inf/nan
        (ms.Tensor(np.full((4, 9), np.inf), dtype=dtype), -2),
        (ms.Tensor(np.full((7, 3, 9, 5, 5, 7), np.nan), dtype=dtype), 0),
        # shape 0
        (ms.Tensor(np.random.randn(8, 0, 5), dtype=dtype), 1),
        # broadcast
        (ms.Tensor(np.random.randn(1, 4, 1), dtype=dtype), -1),
        # discontinuous
        (ms.Tensor(np.random.randn(6, 8)[::2, ::2], dtype=dtype), 1),
        (ms.Tensor(np.random.randn(6, 8), dtype=dtype), 0),
    ]

    for sorted_sequence, diagonal in cases:
        yield OpSampleInput(
            op_input=sorted_sequence,
            op_args=(diagonal,),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # small/large values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_args=(0,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

def basic_sample_inputs_mint_cross(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((3,), (3,), None),
        ((8, 9, 3, 5, 6, 7), (8, 9, 3, 5, 6, 7), 2),
        ((7, 4, 5, 9, 7, 8, 8, 3), (7, 4, 5, 9, 7, 8, 8, 3), -1),
        ((3, 7, 8, 6, 7, 5, 9), (3, 1, 8, 6, 7, 5, 9), 0)
    )
    for input, other, dim in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input),
            op_args=(make_arg(shape=other), ),
            op_kwargs={"dim": dim},
            sample_name=f"{op_info.name}_{dtype}"
        )

def extra_sample_inputs_mint_cross(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases =(
        ((0, 3, 2,), (0, 3, 2), 1), #empty shape
        ((1024, 3), (1024, 3), None), #large shape
    )
    for input, other, dim in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input),
            op_args=(make_arg(shape=other), ),
            op_kwargs={"dim": dim},
            sample_name=f"{op_info.name}_{dtype}"
        )
    #small/large values
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(2, 3, 4)), dtype=dtype),
        op_args=(ms.Tensor(np.random.choice(values, size=(2, 3, 4)), dtype=dtype),),
        op_kwargs={"dim": 1},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    #inf/nan values
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(3, 3, 4)), dtype=dtype),
        op_args=(ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(3, 3, 4)), dtype=dtype),),
        op_kwargs={"dim": 0},
        sample_name=f"{op_info.name}_{dtype}_extremal_values"
    )
    #discontiguous value
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg(shape=(4, 3), discontiguous=discontiguous),
            op_args=(make_arg(shape=(4, 3), discontiguous=discontiguous),),
            op_kwargs={"dim": None},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def error_inputs_other_op_mint_cross_func(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device)
    error_cases = (
        ((2, 2, 4), (2, 2, 4), None, ValueError, "None of the dim is 3"),
        ((2, 3, 4), (2, 3, 4), 0, ValueError, "value of dimension is not 3"),
        ((2, 3, 4), (2, 3, 5), 0, ValueError, "dimensions of input and other does not match"),
    )
    for x1, x2, dim, error_type, msg in error_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=x1),
                op_args=(make_arg(shape=x2),),
                op_kwargs={"dim":dim},
                sample_name=f"{op_info.name}"
            ),
            op_error_type=error_type,
            op_error_info=msg
        )
    #x1 is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=np.random.uniform(0, 10, size=(2, 3, 4)),
            op_args=(make_arg(shape=(2, 3, 4)),),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info="input is not Tensor"
    )
    # x2 is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(2, 3, 4)),
            op_args=(np.random.uniform(0, 10, size=(2, 3, 4)), ),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info="other is not Tensor"
    )

def basic_sample_inputs_mint_cdist(op_info: OpInfo, dtype=None, device=None, **kwargs):
    np_dtype = dtype_to_np_dtype[dtype]
    default_loss = 5e-3 if dtype == ms.float16 else None

    def make_from_np(np_array):
        return make_tensor_with_np_array(np_array.astype(np_dtype), device=device)

    yield OpSampleInput(
        op_input=make_from_np(np.array([[[1.0, 1.5, -2.0], [0.5, -3.0, 4.0]]], dtype=np.float32)),
        op_args=(make_from_np(np.array([[[0.0, -1.0, 2.5], [2.0, 2.0, -1.0], [-1.0, 1.0, 0.5]]], dtype=np.float32)),
                 1.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_p1_3d",
        loss=default_loss,
    )

    rng = np.random.default_rng(0)
    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(-3.0, 3.0, size=(10, 4)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(-3.0, 3.0, size=(11, 4)).astype(np.float32)), 2.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_p2_2d",
        loss=default_loss,
    )

    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(0.1, 10.0, size=(2, 4, 8)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(0.1, 10.0, size=(2, 3, 8)).astype(np.float32)), 2.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_p2_3d",
        loss=default_loss,
    )

    rng = np.random.default_rng(5)
    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(0.1, 10.0, size=(4, 8, 16)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(0.1, 10.0, size=(4, 8, 16)).astype(np.float32)), 2.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_square_output_shape_3d",
        loss=default_loss,
    )

    yield OpSampleInput(
        op_input=make_from_np(np.array([[[-4.258, 4.8]]], dtype=np.float32)),
        op_args=(make_from_np(np.array([[[0.0672, 1.397]]], dtype=np.float32)), 0.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_p0_backward",
        loss=1e-3 if dtype == ms.float16 else None,
    )

    rng = np.random.default_rng(1)
    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(0.1, 10.0, size=(1, 5, 20)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(0.1, 10.0, size=(5, 3, 20)).astype(np.float32)), 2.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_batch_broadcast",
        loss=default_loss,
    )


def extra_sample_inputs_mint_cdist(op_info: OpInfo, dtype=None, device=None, **kwargs):
    np_dtype = dtype_to_np_dtype[dtype]
    default_loss = 5e-3 if dtype == ms.float16 else None

    def make_from_np(np_array):
        return make_tensor_with_np_array(np_array.astype(np_dtype), device=device)

    rng = np.random.default_rng(2)
    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(-5.0, 5.0, size=(2, 5, 4)).astype(np.float32)),),
        op_kwargs={},
        sample_name=f"{op_info.name}_default_p",
        loss=default_loss,
    )

    yield OpSampleInput(
        op_input=make_from_np(np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)),
        op_args=(make_from_np(np.array([[[3.0, 3.0], [3.0, 3.0]]], dtype=np.float32)), 8.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_p8_forward",
        loss=default_loss,
    )

    rng = np.random.default_rng(3)
    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(-2.0, 2.0, size=(4, 8, 5)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(-2.0, 2.0, size=(4, 9, 5)).astype(np.float32)), 5.49830733954465),
        op_kwargs={},
        sample_name=f"{op_info.name}_p_general_3d",
        loss=default_loss,
    )

    rng = np.random.default_rng(4)
    yield OpSampleInput(
        op_input=make_from_np(rng.uniform(-1.0, 1.0, size=(4, 7, 9)).astype(np.float32)),
        op_args=(make_from_np(rng.uniform(-1.0, 1.0, size=(4, 6, 9)).astype(np.float32)), 30.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_p_large_3d",
        loss=default_loss,
    )

    yield OpSampleInput(
        op_input=make_from_np(np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)),
        op_args=(make_from_np(np.array([[[3.0, 3.0], [3.0, 3.0]]], dtype=np.float32)), float("inf")),
        op_kwargs={},
        sample_name=f"{op_info.name}_p_inf_forward",
        loss=default_loss,
    )

def error_inputs_other_op_mint_cdist_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_from_np = functools.partial(make_tensor_with_np_array, device=device)
    test_cases = [
        (
            "1",
            (make_from_np(np.array([[1.0, 2.0]], dtype=np.float32)), 2.0),
            (TypeError, ValueError, RuntimeError),
            "input type error",
        ),
        (
            make_from_np(np.array([1.0, 2.0], dtype=np.float32)),
            (make_from_np(np.array([[1.0, 2.0]], dtype=np.float32)), 2.0),
            (ValueError, RuntimeError),
            "rank less than 2",
        ),
        (
            make_from_np(np.ones((2, 3, 4, 5), dtype=np.float32)),
            (make_from_np(np.ones((2, 4, 4, 5), dtype=np.float32)), 2.0),
            (ValueError, RuntimeError),
            "rank greater than 3",
        ),
        (
            make_from_np(np.ones((2, 3), dtype=np.float32)),
            (make_from_np(np.ones((1, 4, 3), dtype=np.float32)), 2.0),
            (ValueError, RuntimeError),
            "rank mismatch",
        ),
        (
            make_from_np(np.ones((2, 5, 4), dtype=np.float32)),
            (make_from_np(np.ones((3, 6, 4), dtype=np.float32)), 2.0),
            (ValueError, RuntimeError),
            "batch shape mismatch",
        ),
        (
            make_from_np(np.ones((2, 5, 4), dtype=np.float32)),
            (make_from_np(np.ones((2, 3, 5), dtype=np.float32)), 2.0),
            (ValueError, RuntimeError),
            "feature dim mismatch",
        ),
        (
            make_from_np(np.ones((2, 5, 4), dtype=np.float32)),
            (make_from_np(np.ones((2, 3, 4), dtype=np.float16)), 2.0),
            (TypeError, ValueError, RuntimeError),
            "dtype mismatch",
        ),
        (
            make_from_np(np.ones((2, 5, 4), dtype=np.float32)),
            (make_from_np(np.ones((2, 3, 4), dtype=np.float32)), -1.0),
            (ValueError, RuntimeError),
            "negative p",
        ),
        (
            make_from_np(np.ones((2, 5, 4), dtype=np.float32)),
            (make_from_np(np.ones((2, 3, 4), dtype=np.float32)), 1),
            (TypeError, ValueError, RuntimeError),
            "integer p",
        ),
        (
            make_from_np(np.ones((2, 5, 4), dtype=np.float32)),
            (make_from_np(np.ones((2, 3, 4), dtype=np.float32)), "none"),
            (TypeError, ValueError, RuntimeError),
            "string p",
        ),
        (
            [66.9],
            ([23.9],),
            (TypeError, ValueError, RuntimeError),
            "list input type error",
        ),
    ]
    for op_input, op_args, op_error_type, op_error_info in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=op_args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=op_error_type,
            op_error_info=f"other ops {op_info.name} {op_error_info}",
        )

    input_not_support_dtypes = (
        ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.float64, ms.complex64, ms.complex128
    )
    rng = np.random.default_rng(0)
    for input_dtype in input_not_support_dtypes:
        np_dtype = dtype_to_np_dtype[input_dtype]
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_from_np(rng.uniform(-3.0, 3.0, size=(10, 4)).astype(np_dtype)),
                op_args=(make_from_np(rng.uniform(-3.0, 3.0, size=(11, 4)).astype(np_dtype)), 2.0),
                op_kwargs={},
                sample_name=f"{op_info.name}_input_dtype_{input_dtype}_error",
            ),
            op_error_type=(RuntimeError, TypeError),
            op_error_info=f"input dtype {input_dtype} is not supported for {op_info.name}",
        )

def dynamic_sample_inputs_mint_cdist(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=-2.0, high=2.0)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype), 1.0),
                op_kwargs={},
                grad_position=(),
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_3d",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 7, 9)),
                    op_args=(make_func(shape=(3, 8, 9)), 1.0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3d",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 5)),
                    op_args=(make_func(shape=(4, 9, 5)), 5.49830733954465),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3d",
                ),
            ),
        )

        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype), 2.0),
                op_kwargs={},
                grad_position=(),
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_broadcast",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 5, 20)),
                    op_args=(make_func(shape=(5, 3, 20)), 2.0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_broadcast",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 6, 10)),
                    op_args=(make_func(shape=(4, 3, 10)), 5.49830733954465),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_broadcast",
                ),
            ),
        )

def extra_sample_inputs_addmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (4, 4), ((4, 4),(4, 4)), {},
                                  dtype=dtype, device=device,exclude_value=(-5062757.7, 5062757.7, 1e20,-1e20,))

    yield from make_extra_tensor(op_info, (0, 0), ((0, 4), (4, 0)), {},
                                 dtype=dtype, device=device, empty_tensor=True)

def error_sample_inputs_mint_addmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_ = functools.partial(make_tensor, device=device, dtype=ms.float32)

    A=3
    B=6
    C=10

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, C)),
            op_args=(make_((2, A, B)), make_((B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of mat1 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((B, C)),
            op_args=(make_((B, )), make_((B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of mat1 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, B)),
            op_args=(make_((A, B)), make_((B, )),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of mat2 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, C)),
            op_args=(make_((A, B)), make_((B, C, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of mat2 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((1, C-1)),
            op_args=(make_((A, B)), make_((B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the input.shape should be broadcasted by mat1.shape@mat2.shape',
    )

def dynamic_sample_inputs_mint_addmm(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # dyncamic shape
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(None, None), dtype=dtype),
            op_args=(ms.Tensor(shape=(None, None), dtype=dtype),ms.Tensor(shape=(None, None), dtype=dtype),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(3, 10) ),
                op_args=(make_func(shape=(3, 6) ),make_func(shape=(6, 10) ) ),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(5, 3) ),
                op_args=(make_func(shape=(5, 10) ),make_func(shape=(10, 3) ) ),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )

def basic_sample_inputs_addmv(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ((2, ), (2, 10,), (10,)  ),
        ((3, ), (3, 9), (9, )  ),
        ((1, ), (8, 2), (2, )  ),
    )

    for input, mat, vec  in case:

        yield OpSampleInput(
            op_input=make_arg(input),
            op_args=(make_arg(mat), make_arg(vec),),
            op_kwargs={},
            sample_name=op_info.name,
        )

        if dtype.is_floating_point:
            yield OpSampleInput(
                op_input=make_arg(input),
                op_args=(make_arg(mat), make_arg(vec),),
                op_kwargs={'beta':2.6, 'alpha':3},
                sample_name=op_info.name,
            )

            yield OpSampleInput(
                op_input=make_arg(input),
                op_args=(make_arg(mat), make_arg(vec),),
                op_kwargs={'beta':3, 'alpha':5.7},
                sample_name=op_info.name,
            )
        else:
            yield OpSampleInput(
                op_input=make_arg(input),
                op_args=(make_arg(mat), make_arg(vec),),
                op_kwargs={'beta': 3, 'alpha': 5},
                sample_name=op_info.name,
            )

def error_sample_inputs_addmv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    N = 3
    M = 10
    C = 6
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, )),
            op_args=(make_arg((N, M)), make_arg((C, ))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='vec.shape[0] should equal mat.shape[-1]',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N,)),
            op_args=(make_arg((N, M)), make_arg((M, C))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='vec.dim should be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N,)),
            op_args=(make_arg((N, M, C)), make_arg((C, ))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='mat.dim should be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, )),
            op_args=(make_arg((N, )), make_arg((N, ))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='mat.dim should be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N,)),
            op_args=(make_arg((M, C)), make_arg((C, ))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.shape should be  broadcasted by  mat@vec.shape',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, M,)),
            op_args=(make_arg((M, C)), make_arg((C,))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='input.dim should be 1',
    )

def extra_sample_inputs_addmv(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (16, ), ((16, 16), (16, )), {},
                                 dtype=dtype, device=device, exclude_value=(-5062757.7, 5062757.7, 1e20,-1e20))

    yield from make_extra_tensor(op_info, (0, ), ((0, 7), (7, )), {},
                                 dtype=dtype, device=device, empty_tensor=True)

def extra_sample_inputs_mint_trace(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate extra sample inputs for mint.trace.
    """
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(
            np.arange(30).reshape(10, 3).astype(dtype_to_np_dtype[dtype]),
            dtype=dtype,
            device=device,
        ),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_tall_matrix",
    )


def dynamic_sample_inputs_mint_trace(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic-shape/rank inputs for mint.trace.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

# basic op_basic_reference_inputs_func for glu ops
def basic_sample_inputs_glu_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield typical shape cases for glu ops.

    Covers vectors, singleton dims, medium 2D/3D, and empty-dimension cases.

    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    """
    S = SMALL_DIM_SIZE - 1
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
        random_method='randn',
    )

    shapes = (
        (S,),
        (M, S),
        (S, S, L),
        (3, 0, 2),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 2),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for input_shape in shapes:
        _input = make_func(input_shape)

        yield OpSampleInput(
            _input,
            op_args=(),
            sample_name=op_info.name,
        )

        # randomDim
        yield OpSampleInput(
            _input,
            op_args=(random.choice([dimIndex for dimIndex, dimValue in enumerate(input_shape) if dimValue % 2 == 0]),),
            sample_name=op_info.name,
        )

# op_extra_reference_inputs_func for glu ops
def _generate_glu_tensors_sample_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate single-tensor sample inputs for glu.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering empty/scalar/various shapes.
    """
    shapes = (
        (0,),        # empty tensors
        (1, 0, 4),   # empty tensors
        (20,),       # 1D tensors with small size
        (812,),      # 1D tensors with medium size
        (1029, 918), # 2D tensors with large size
    )

    for shape in shapes:
        yield OpSampleInput(
            op_input=make_tensor(shape, dtype, device=device, random_method='randn'),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_tensor_inputs",
        )

def _generate_glu_discontiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate contiguous and discontiguous tensor inputs for glu ops.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering contiguous/non-contiguous memory layouts.
    """

    shapes = (
        (2,),
        (3, 2),
        (1, 2, 4),
    )

    for shape, discontiguous in itertools.product(shapes, [False, True]):
        yield OpSampleInput(
            op_input=make_tensor(shape, dtype=dtype, device=device, discontiguous=discontiguous, random_method='randn'),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_discontiguous_tensor_inputs",
        )

def _generate_glu_extremal_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor with extremal value for glu.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering extremal value.
    """
    # inf and nan is unsupported on Ascend910 devices.
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    S = SMALL_DIM_SIZE - 1
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.full((S, S), np.inf), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=op_info.name,
    )
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.full((S, S), -np.inf), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=op_info.name,
    )
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.full((S, S), np.nan), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=op_info.name,
    )

def extra_sample_inputs_glu_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate comprehensive reference inputs for glu.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Aggregated samples from multiple input generators.
    """
    if dtype in dtypes_extra_uint:
        return
    # tensors with many kinds of shapes
    yield from _generate_glu_tensors_sample_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with small value
    if dtype != ms.bool_:
        yield from _generate_unary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_unary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # contiguous or discontiguous tensors
    yield from _generate_glu_discontiguous_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensor with extremal value
    if dtype.is_floating_point or dtype.is_complex:
        yield from _generate_glu_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)

# op_error_inputs_func for ops
def error_inputs_glu_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for glu ops.
    '''
    # dim size is not even while dim is default
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(1,), dtype=ms.float32),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='dim size is not even while dim is default',
    )
    # dim size is not even while dim is not default
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,2,1,2,3), dtype=ms.float32),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='dim size is not even while dim is not default',
    )
    # dim is not int
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2, 1, 2, 3), dtype=ms.float32),
            op_args=(1.0,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info=f'dim error type is not supported for {op_info.name}',
    )
    # dim is not in [-r, r): like [-5, 5)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2, 1, 2, 3), dtype=ms.float32),
            op_args=(5,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=IndexError,
        op_error_info=f'dim is out range, error input is not supported for {op_info.name}',
    )

# op_dynamic_inputs_func for glu ops
def dynamic_sample_inputs_glu_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate dynamic-shape/rank inputs for glu.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    """
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
                    op_input=make_func(shape=(2, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 6)),
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
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def basic_sample_inputs_outer(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ((2,), (10,)),
        ((3,), (5,)),
        ((8,), (2,)),
    )

    for input, vec in case:
        yield OpSampleInput(
            op_input=make_arg(input),
            op_args=(make_arg(vec),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def make_error_tensor_with_matrix_multiply(op_info: OpInfo, input_dim, other_dim, is_addition=False, device=None,):

    B = 2
    N = 4
    C = 6
    M = 8
    B2 =3
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    if input_dim == 2 and other_dim ==1:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((N, C)),
                op_args=(make_arg((M,)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.shape[-1] should equal other.shape[0]',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((N, C)),
                op_args=(make_arg((C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='other.dim should equal 1',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((N, C)),
                op_args=(make_arg(()),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='other.dim should equal 1',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((B, N, C)),
                op_args=(make_arg((C, )),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.dim should equal 2',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((C, )),
                op_args=(make_arg((C, )),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.dim should equal 2',
        )
    elif  input_dim == 2 and other_dim ==2:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((N, C)),
                op_args=(make_arg((M, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.shape[-1] should equal other.shape[0]',
        )
    elif input_dim == 3 and other_dim ==3:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((B, N, C)),
                op_args=(make_arg((B, M, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.shape[-1] should equal other.shape[1]',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((B, N, C)),
                op_args=(make_arg((B2, C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.shape[0] should equal other.shape[0]',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((B2, B, N, C)),
                op_args=(make_arg((B, C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.shape should be equal 3',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((B, N, C)),
                op_args=(make_arg((C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='other.shape should be equal 3',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((N, C)),
                op_args=(make_arg((B, C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.shape should be equal 3',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((B, N, C)),
                op_args=(make_arg((B2, B, C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='other.shape should be equal 3',
        )
    elif input_dim == 1 and other_dim ==1:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((C, )),
                op_args=(make_arg((C, M)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='other.dim should equal 1',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((C,)),
                op_args=(make_arg(()),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='other.dim should equal 1',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((C, M)),
                op_args=(make_arg((M, )),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.dim should equal 1',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(()),
                op_args=(make_arg((M,)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='input.dim should equal 1',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((C, )),
                op_args=(make_arg((M, )),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=RuntimeError,
            op_error_info='input.shape should be equal other.shape',
        )

def error_sample_inputs_outer(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from  make_error_tensor_with_matrix_multiply(op_info, 2, 2, device=device,)

def extra_sample_inputs_outer(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (16, ), ((16, ),), {},
                                 dtype=dtype, device=device, dtype_promotion=True)

    yield from make_extra_tensor(op_info, (0,), ((0,),), {}, dtype=dtype,
                                 device=device, empty_tensor=True)

def basic_sample_inputs_dot(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case=(
        (0,),
        (3,),
        (10,)
    )
    for shape in case:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(make_arg(shape),),
            op_kwargs={},
            sample_name=f'{op_info.name}_1d_length_{shape[0]}_{dtype}',
        )


def error_sample_inputs_dot(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_error_tensor_with_matrix_multiply(op_info, 1, 1, device=device)

def extra_sample_inputs_dot(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (16, ), ((16, ),), {},
                                 dtype=dtype, device=device, exclude_value=(1e20,-1e20))

    yield  OpSampleInput(
            op_input=make_tensor((512,), dtype=dtype),
            op_args=(make_tensor((512,), dtype=dtype),),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_large_dim',
        )

def dynamic_inputs_tensor_dot(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(None, ), dtype=dtype),
            op_args=(ms.Tensor(shape=(None, ), dtype=dtype),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(2, )),
                op_args=(make_func(shape=(2, )), ),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(4, )),
                op_args=(make_func(shape=(4, )), ),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )

def basic_sample_inputs_baddbmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_=functools.partial(make_tensor, device = device, dtype = dtype)

    case = (
        ( (2, 5, 7), (2, 7, 3), (1, 5, 1) ),
        ( (4, 1, 6), (4, 6, 9), (4, 1, 1) ),
    )

    for batch1, batch2, input in case:
        yield OpSampleInput(
            make_(input),
            op_args=(make_(batch1), make_(batch2)),
            op_kwargs={},
            sample_name=op_info.name
        )
        yield OpSampleInput(
            make_(input),
            op_args=(make_(batch1), make_(batch2)),
            op_kwargs={'beta': 3, 'alpha': 6},
            sample_name=op_info.name
        )
        yield OpSampleInput(
            make_(input),
            op_args=(make_(batch1), make_(batch2)),
            op_kwargs={'beta': 1.2, 'alpha': 3.4},
            sample_name=op_info.name
        )



def extra_sample_inputs_baddbmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from  make_extra_tensor(op_info, (2, 4, 4), ((2, 4, 3),(2, 3, 4)),
                        {}, dtype=dtype, device=device, exclude_value=(-5062757.7, 5062757.7, -1e20, 1e20))

    yield from make_extra_tensor(op_info, (2, 4, 4), ((2, 4, 0), (2, 0, 4)),
                                 {}, dtype=dtype, device=device, empty_tensor=True)

def error_sample_inputs_baddbmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_ = functools.partial(make_tensor, device=device, dtype=ms.float32)

    A=3
    B=6
    C=10
    batch=1
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, C)),
            op_args=(make_((2, batch, A, B)), make_((batch, B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the dim of mat1 must be 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((B, C)),
            op_args=(make_((batch, B, )), make_((batch, B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the dim of mat1 must be 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, B)),
            op_args=(make_((batch, A, B)), make_((batch, B, )),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the dim of mat2 must be 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, C)),
            op_args=(make_((batch, A, B)), make_((batch, B, C, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the dim of mat2 must be 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((1, C-1)),
            op_args=(make_((batch, A, B)), make_((batch, B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the input.shape should be broadcasted by mat1.shape@mat2.shape',
    )

def basic_sample_inputs_traingular_solve(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_ = functools.partial(make_tensor, device=device, dtype=dtype)

    for upper, transpose, unitriangular in itertools.product([True,False],repeat = 3):
        for  i  in  range (5):
            yield OpSampleInput(
                make_(tuple(np.random.choice([1, 3], size = i)) + (3, 4) ),
                op_args=(make_(tuple(np.random.choice([1, 3], size = i)) + (3, 3)), ),
                op_kwargs={'upper': upper, 'transpose': transpose, 'unitriangular':unitriangular},
                sample_name=op_info.name
            )
            yield OpSampleInput(
                make_(tuple(np.random.choice([1, 3], size = i)) + (3, 3)),
                op_args=(make_(tuple(np.random.choice([1, 3], size = i)) + (3, 3)), ),
                op_kwargs={'upper': upper, 'transpose': transpose, 'unitriangular':unitriangular},
                sample_name=op_info.name
            )

def error_sample_inputs_traingular_solve(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_ = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((2, )),
            op_args=(make_((2, )), ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the dim of input A and B must be [2, 6]',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((2, 2, 2, 2, 2, 2, 2,)),
            op_args=(make_((2, 2, 2, 2, 2, 2, 2,)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the dim of input A and B must be [2, 6]',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((4, 3)),
            op_args=(make_((3, 4)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='A must be batchs of square matrices',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((5, 4)),
            op_args=(make_((4, 4)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='B.shape[-2] must be equal A.shape[-2]',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((2, 5, 4)),
            op_args=(make_((3, 4, 4)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='B.batch must be broadcasted by A.batch',
    )

def extra_sample_inputs_traingular_solve(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for upper, transpose, unitriangular in itertools.product([True, False], repeat=3):
        yield from make_extra_tensor(op_info, (3, 7, 6), ((7, 7),),
                                     {'upper': upper, 'transpose': transpose, 'unitriangular': unitriangular},
                                     dtype=dtype, device=device, exclude_value=(-1e20, 1e20, -5062757.7, 5062757.7))

        yield from make_extra_tensor(op_info, (3, 0, 7), ((0, 0),),
                                     {'upper': upper, 'transpose': transpose, 'unitriangular': unitriangular},
                                     dtype=dtype, device=device, empty_tensor=True)

def basic_sample_inputs_inverse(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    for i in range(5):

        yield OpSampleInput(
            op_input=make_arg(tuple(np.random.randint(1, 9, size=i)) + (3, 3)),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_sample_inputs_inverse(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2,3)),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the lase two dim should be same.',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5, 6, 2, 2)),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the input tensor cannot be lager than 6 dim.',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, )),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the input tensor must have at least 2 dim.',
    )

def extra_sample_inputs_inverse(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (5, 5 ), (), {}, dtype=dtype, device=device,
                                 exclude_value=(0.0, -0.0))

    yield from make_extra_tensor(op_info, (0, 0), (), {},
                                 dtype=dtype, device=device, empty_tensor=True)

def basic_sample_inputs_mm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ( (2, 7), (7, 1)  ),
        ( (1, 8), (8, 9)  )
    )

    for input, mat2  in case:

        yield OpSampleInput(
            op_input=make_arg(input),
            op_args=(make_arg(mat2),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_sample_inputs_mm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_error_tensor_with_matrix_multiply(op_info, 2, 2, device=device,)


def extra_sample_inputs_mm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (4, 5), ((5, 7),), {},
                                 dtype=dtype, device=device, exclude_value=( -5062757.7, 5062757.7, -1e20, 1e20))

    yield from make_extra_tensor(op_info, (4, 0), ((0, 7),), {},
                                 dtype=dtype, device=device, empty_tensor=True)

def basic_sample_inputs_mv(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ( (2, 7), (7, )  ),
        ( (1, 8), (8, )  )
    )

    for input, vec  in case:

        yield OpSampleInput(
            op_input=make_arg(input),
            op_args=(make_arg(vec),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_sample_inputs_mv(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_error_tensor_with_matrix_multiply(op_info, 2, 1, device=device, )


def extra_sample_inputs_mv(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (1, 16 ), ((16, ),), {},
                                 dtype=dtype, device=device, exclude_value=( -5062757.7, 5062757.7, -1e20, 1e20))

    yield from make_extra_tensor(op_info, (1, 0), ((0, ),), {},
                                 dtype=dtype, device=device, empty_tensor=True)

def basic_sample_inputs_bmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ( (2, 4, 7), (2, 7, 1)  ),
        ( (3, 3, 9 ), (3, 9, 3)  ),
        ( (8, 7, 2), (8, 2, 1)  ),
    )

    for input, mat2  in case:

        yield OpSampleInput(
            op_input=make_arg(input),
            op_args=(make_arg(mat2),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_sample_inputs_bmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_error_tensor_with_matrix_multiply(op_info, 3, 3, device=device,)

def extra_sample_inputs_bmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (1, 3, 6 ), ((1, 6 ,3),), {},
                                 dtype=dtype, device=device, exclude_value=( -5062757.7, 5062757.7, -1e20, 1e20))

    yield from make_extra_tensor(op_info, (1, 3, 0), ((1, 0, 7),), {}, dtype=dtype,
                                 device=device, empty_tensor=True)

def dynamic_sample_inputs_mint_bmm(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # dyncamic shape
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(2, None, None), dtype=dtype),
            op_args=(ms.Tensor(shape=(2, None, None), dtype=dtype),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(2, 2, 3), ),
                op_args=(make_func(shape=(2, 3, 4), ),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 4, 5),),
                op_args=(make_func(shape=(2, 5, 3), ),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )

def basic_sample_inputs_addbmm(op_info, dtype=None, device=None, **kwargs):
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)
    cases=(
        ((), (8, 7, 7), (8, 7, 9)),
        ((7, ), (2, 9, 5), (2, 5, 7)),
        ((5, 8), (9, 5, 4), (9, 4, 8)),
    )
    factor=(0, 1, 0.5, -2)
    for case, beta, alpha in itertools.product(cases, factor, factor):
        yield OpSampleInput(
            op_input=make_input(case[0]),
            op_args=(make_input(case[1]), make_input(case[2]),),
            op_kwargs={"beta":beta, "alpha":alpha},
            sample_name=f"{op_info.name}_{dtype}"
        )

def extra_sample_inputs_addbmm(op_info, dtype=None, device=None, **kwargs):
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #for empty/large shape
    cases=(
        ((0, 0), (0, 0, 0), (0, 0, 0)),
        ((5, 6), (3, 5, 0), (3, 0, 6)),
        ((5, 0), (0, 5, 0), (0, 0, 0)),
        ((1024, 10 ), (812, 1024, 20), (812, 20, 10))
    )
    for case in cases:
        yield OpSampleInput(
            op_input=make_input(case[0]),
            op_args=(make_input(case[1]), make_input(case[2]),),
            sample_name=f"{op_info.name}_{dtype}_extra_shape"
        )
    #for small/large values, some errors happens and has been record, skip...
    #for inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((5, 6), t), dtype=dtype),
            op_args=(ms.Tensor(np.full((3, 5, 3), t), dtype=dtype),
                     ms.Tensor(np.full((3, 3, 6), t), dtype=dtype),),
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )
    #for contiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_input((5, 6), discontiguous=discontiguous),
            op_args=(make_input((3, 5, 4), discontiguous=discontiguous),
                     make_input((3, 4, 6), discontiguous=discontiguous),),
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def error_sample_inputs_addbmm(op_info, dtype=None, device=None, **kwargs):
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #shape of input exceeds 2d
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((3, 4, 5)),
            op_args=(make_input((3, 4, 5)),
                     make_input((3, 5, 5)), ),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_shape_exceeds",
        ),
        op_error_type=RuntimeError,
        op_error_info='input shape cannot be large than 2d',
    )
    #input shape can not broadcast with batch
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((5, 5)),
            op_args=(make_input((3, 4, 5)),
                     make_input((3, 5, 5)),),
            op_kwargs={},
            sample_name=f"{op_info.name}_shape_not_broadcast",
        ),
        op_error_type=RuntimeError,
        op_error_info='input shape cannot broadcast with batch',
    )
    #batch is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((5, 5)),
            op_args=(np.random.randn(3, 4, 5),
                     make_input((3, 5, 5)),),
            op_kwargs={},
            sample_name=f"{op_info.name}_shape_not_broadcast",
        ),
        op_error_type=TypeError,
        op_error_info='batch is not tensor',
    )

def basic_sample_inputs_mint_einsum(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        ("ij,jk->ik", (ms.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype=dtype),
                 ms.Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), dtype=dtype),),),
        ("...->", (ms.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype=dtype),),),
        ("i->", (ms.Tensor(np.array([1.0, 2.0, 4.0]),dtype=dtype),),)
    ]
    for op_inputs, args in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_einsum(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        # broadcast
        ("i,j->ij", (ms.Tensor([1.0, 2.0, 3.0], dtype=dtype), ms.Tensor([2.0, 4.0], dtype=dtype),),),
        # discontinuous
        ("i,i->", (ms.Tensor(np.array([1.0, 2.0, 3.0]), dtype=dtype)[::2], ms.Tensor(np.array([2.0, 4.0, 6.0]),
                                                                                     dtype=dtype)[::2]),),
        # shape 0
        ("ij->i", (ms.Tensor(np.array([]), dtype=dtype).reshape(2, 0),),),
    ]
    for op_inputs, args in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

    # small/large values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input="ij->i",
                op_args=(make_tensor_with_np_array(np_array, dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input="ij->i",
                op_args=(make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_mint_einsum(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (0, (ms.Tensor(np.random.randn(8), dtype=dtype),),),#equation is int
        ("i,j ->ij", (True,)),#operands is bool
        (None, (ms.Tensor([1, 2, 3, 4], dtype=dtype), [random.randint(52, 60)],
                ms.Tensor([1, 2], dtype=dtype)),), #not 0-52
    ]
    for op_inputs, args in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.einsum error values',
        )

def basic_reference_inputs_tensor_addr_func(op_info: OpInfo, dtype, device=None, **kwargs):

    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)

    shapes = (
        ((), (S,), (S,), 1, 1),
        ((S,),(S,), (S,), 1, 1),
        ((M, S), (M, ), (S,), 2, 1),
        ((M, S), (M,), (S,), 0, 1),#input ignore
    )
    #torch beta,alpha type number;torch input only 2D;input_shape, vec1, vec2 must 1D
    for input_shape, vec1, vec2, beta, alpha in shapes:
        _input = make_arg(input_shape)
        _tensor1 = make_arg(vec1)
        _tensor2 = make_arg(vec2)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(_tensor1, _tensor2, beta, alpha,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def basic_reference_inputs_tensor_cholesky_solve_func(op_info: OpInfo, dtype, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)

    shapes = (
        ((S, M), (S, S), False),
        ((S, M, L),(S, M, M), True),
        ((S, L), (S, S), True),
        ((S, L, M), (S, L, L), False),#input ignore
    )
    #input(*, N, M), input2(*, N, N)
    for input_shape, input2, upper in shapes:
        _input = make_arg(input_shape)
        _tensor1 = make_arg(input2)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(_tensor1, upper,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def basic_sample_inputs_tensor_addmm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (
             ((2, 6), (2, 4), (4, 6),),
             ((5, 3), (5, 8), (8, 3),),
             ((1, 15), (1, 12), (12, 15)),
             ((1024, 1), (1024, 2), (2, 1)),
             ((2, 0), (2, 1), (1, 0),),
            )

    make_ = functools.partial(make_tensor, device=device, dtype=dtype)
    for input, mat1, mat2 in cases:
        yield OpSampleInput(
            make_(input),
            op_args=(make_(mat1),make_(mat2)),
            op_kwargs={},
            sample_name=op_info.name)
        yield OpSampleInput(
            make_(input),
            op_args=(make_(mat1), make_(mat2)),
            op_kwargs={'beta': 2, 'alpha': 3.9},
            sample_name=op_info.name)
        yield OpSampleInput(
            make_(input),
            op_args=(make_(mat1), make_(mat2)),
            op_kwargs={'beta': 3.1, 'alpha': 8},
            sample_name=op_info.name)

def basic_reference_inputs_tensor_addcdiv_func(op_info: OpInfo, dtype, device=None,**kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    base_shapes = (
        ((S,),(S, 1), (S, S, S), 0.1),
        ((M, S), (1, S), (1, S), 0.2),
        ((S, S, L), (S, L), (S, L), 0.2),
        ((2, 0, 3, 2), (2, 0, 3, 1), (2, 0, 3, 1), 0.1),
        ((2, 3, 4, 1, 2), (2, 1, 1, 1, 1), (2, 1, 1, 1, 1), 0.1),
        ((2, 1, 2, 2, 1, 2), (2, 2), (1, 1), 0.3),
        ((2, 1, 2, 2, 1, 2, 1), (2, 1, 1, 1, 1), (2, 1, 1, 1, 1), 0.2),
        ((2, 1, 2, 2, 1, 2, 1, 2), (2, 1, 1, 1, 1), (2, 1, 1, 1, 1), 0.4),
    )
    base_shapes_addcmul = (
        ((S,), (S, 1), (S, S, S), 0.1),
        ((M, S), (1, S), (1, S), 0.2),
        ((S, S, L), (S, L), (S, L), 0.2),
        ((2, 3, 4, 1, 2), (2, 1, 1, 1, 1), (2, 1, 1, 1, 1), 0.1),
        ((2, 1, 2, 2, 1, 2), (2, 2), (1, 1), 0.3),
        ((2, 1, 2, 2, 1, 2, 1), (2, 1, 1, 1, 1), (2, 1, 1, 1, 1), 0.2),
        ((2, 1, 2, 2, 1, 2, 1, 2), (2, 1, 1, 1, 1), (2, 1, 1, 1, 1), 0.4),
    )
    if op_info.name == "Tensor.addcdiv":
        shapes = base_shapes
    else:
        shapes = base_shapes_addcmul
    #torch value type number;input_shape, tensor1, tensor2 must broadcast
    for input_shape, tensor1, tensor2, values in shapes:
        _input = make_arg(input_shape)
        _tensor1 = make_arg(tensor1)
        _tensor2 = make_arg(tensor2)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(_tensor1, _tensor2,),
            op_kwargs={"value": values,},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def basic_reference_inputs_tensor_addmv_func(op_info: OpInfo, dtype, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)

    shapes = (
        ((), (S, S), (S,), 1, 1),
        ((S,),(S, S), (S,), 1, 1),
        ((M,), (M, 1), (1,), 0, 1),#input ignore
    )
    #input 1d, mat 2d, vec 1d
    for input_shape, mat, vec, beta, alpha in shapes:
        _input = make_arg(input_shape)
        _tensor1 = make_arg(mat)
        _tensor2 = make_arg(vec)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(_tensor1, _tensor2, beta, alpha,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_tensor_addr(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    yield OpSampleInput(
        op_input=make_arg((1024, 1)),
        op_args=(make_arg((1024,)), make_arg((1,)), 2, 1,),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}",
    )
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values))
        _tensor1 = np.array(_float_small_values[:len(_float_small_values)])
        _tensor2 = np.array(_float_small_values[:len(_float_small_values)])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     make_tensor_with_np_array(_tensor2, dtype=dtype, device=device), 1, 1),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        _tensor1 = np.array(_large_values[:len(_large_values)])
        _tensor2 = np.array(_large_values[:len(_large_values)])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     make_tensor_with_np_array(_tensor2, dtype=dtype, device=device),1, 1,),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 3), t)
                _tensor1 = np.ones((1,))
                _tensor2 = np.ones((3,))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                             make_tensor_with_np_array(_tensor2, dtype=dtype, device=device), 1, 1,),
                    op_kwargs={},
                    sample_name=op_info.name,
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (1, 3)
        _tensor1 = (1,)
        _tensor2 = (3,)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(make_arg(_tensor1, discontiguous=discontiguous),
                     make_arg(_tensor2, discontiguous=discontiguous), 1, 1,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_tensor_cholesky_solve(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(4, 2, 2)
        _tensor1 = np.array(_float_small_values[:len(_float_small_values)]).reshape(4, 2, 2)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device), False,),
            op_kwargs={},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(2, 5)
        _tensor1 = np.array(_large_values[:4]).reshape(2, 2)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     False,),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 3, 5), t)
                _tensor1 = np.ones((1, 3, 3))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                             False,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (1, 3, 5)
        _tensor1 = (1, 3, 3)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(make_arg(_tensor1, discontiguous=discontiguous), False,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_tensor_addcdiv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=-9, high=0)
    yield OpSampleInput(
        op_input=make_arg((),),
        op_args=(make_arg((3, 3, 3),), make_arg((3, 3),)),
        op_kwargs={"value": 0.1,},
        sample_name=f"{op_info.name}_{dtype}",
    )
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values))
        _tensor1 = np.array(_float_small_values[:len(_float_small_values)]).reshape(1, 1, len(_float_small_values))
        _tensor2 = np.array(_float_small_values[:len(_float_small_values)]).reshape(1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     make_tensor_with_np_array(_tensor2, dtype=dtype, device=device),),
            op_kwargs={"value": 0.1},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values))
        _tensor1 = np.array(_large_values[:len(_large_values)]).reshape(1, 1, len(_large_values))
        _tensor2 = np.array(_large_values[:len(_large_values)]).reshape(1, 1, len(_large_values))
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     make_tensor_with_np_array(_tensor2, dtype=dtype, device=device),),
            op_kwargs={"value": 0.1},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 3), t)
                _tensor1 = np.ones((1, 1, 3))
                _tensor2 = np.ones((1, 1, 3))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                             make_tensor_with_np_array(_tensor2, dtype=dtype, device=device),),
                    op_kwargs={"value": 0.1},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (1, 3, 3)
        _tensor1 = (3, 3, 3)
        _tensor2 = (3, 3, 3)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(make_arg(_tensor1, discontiguous=discontiguous),
                     make_arg(_tensor2, discontiguous=discontiguous),),
            op_kwargs={"value": 0.1},
            sample_name=op_info.name,
        )

def extra_sample_inputs_tensor_addmv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # input 1d, mat 2d, vec 1d
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values)
        mat = np.array(_float_small_values[:len(_float_small_values)]).reshape(len(_float_small_values), 1)
        vec = np.array(_float_small_values[:1])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(mat, dtype=dtype, device=device),
                     make_tensor_with_np_array(vec, dtype=dtype, device=device), 1, 1),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values)
        _tensor1 = np.array(_large_values[:len(_large_values)]).reshape(len(_large_values), 1)
        _tensor2 = np.array(_large_values[:1])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     make_tensor_with_np_array(_tensor2, dtype=dtype, device=device),1, 1,),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((3,), t)
                _tensor1 = np.ones((3, 1))
                _tensor2 = np.ones((1,))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                             make_tensor_with_np_array(_tensor2, dtype=dtype, device=device), 1, 1,),
                    op_kwargs={},
                    sample_name=op_info.name,
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (3,)
        _tensor1 = (3, 1)
        _tensor2 = (1,)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(make_arg(_tensor1, discontiguous=discontiguous),
                     make_arg(_tensor2, discontiguous=discontiguous), 1, 1,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_tensor_addbmm(op_info, dtype=None, device=None, **kwargs):
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(4, 4)
        _tensor1 = np.array(_float_small_values[:len(_float_small_values)]).reshape(1, 4, 4)
        _tensor2 = np.array(_float_small_values[:len(_float_small_values)]).reshape(1, 4, 4)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(_tensor1, dtype=dtype, device=device),
                     make_tensor_with_np_array(_tensor2, dtype=dtype, device=device), 1, 1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_small",
        )

    #for empty/large shape
    cases=(
        ((0, 0), (0, 0, 0), (0, 0, 0)),
        ((5, 6), (3, 5, 0), (3, 0, 6)),
        ((5, 0), (0, 5, 0), (0, 0, 0)),
        #((1024, 10 ), (812, 1024, 20), (812, 20, 10)); float16 loss error
    )
    for case in cases:
        yield OpSampleInput(
            op_input=make_input(case[0]),
            op_args=(make_input(case[1]), make_input(case[2]),),
            sample_name=f"{op_info.name}_{dtype}_extra_shape"
        )
    #for inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((5, 6), t), dtype=dtype),
            op_args=(ms.Tensor(np.full((3, 5, 3), t), dtype=dtype),
                     ms.Tensor(np.full((3, 3, 6), t), dtype=dtype),),
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )
    #for contiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_input((5, 6), discontiguous=discontiguous),
            op_args=(make_input((3, 5, 4), discontiguous=discontiguous),
                     make_input((3, 4, 6), discontiguous=discontiguous),),
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

# ====================================linalg error sample inputs functions===================================
def error_inputs_other_op_tensor_addr_func(op_info, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (2, make_arg((S,)), make_arg((S,)), 1, 1),#input is not tensor
        (make_arg((M, S)), 1, make_arg((S,)), 0, 1),#vec1 is not tensor
        (make_arg((1024, 1)), make_arg((1024,)), 1, 2, 1),#vec2 is not tensor
        (make_arg((2, 0)), make_arg((2,)), make_arg((1,)), "1", 1),#beta is str
        (make_arg((2, 0)), make_arg((2,)), make_arg((1,)), 1, "1"),  # alpha is str
    )
    for input_shape, vec1, vec2, beta, alpha in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(vec1, vec2, beta, alpha),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.addr error values')

def error_inputs_other_op_cholesky_solve_func(op_info, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (make_arg(()), make_arg((S, S)), False),#input empty tensor
        (make_arg((S, M, L, L)), make_arg((S, M, M)), False),#INPUT 4D
        (1.0, make_arg((S, S)), True),#input scalar
        (make_arg((S, L, M)), make_arg(()), False),#input2 empty tensor
        (make_arg((S, L, M)), 1.0, False),#input2 scalar
        (make_arg((S, L, M)), make_arg((S, L, L, L)), False),  # input2 4D
        (make_arg((S, L, M)), make_arg((S, L, L)), 1.33),#UPPER not bool
    )
    for input_shape, input2, upper in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(input2, upper,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.cholesky_solve error values')

def error_inputs_other_op_tensor_addcdiv_func(op_info, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_ = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((S, S, L)),
            op_args=(make_((S, L)), make_((S, L)),),
            op_kwargs={"value": "0.1"},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError, TypeError),
        op_error_info='value type must be number',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=10.1,
            op_args=(make_((S, L)), make_((S, L)),),
            op_kwargs={"value": 0.1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError, TypeError, AttributeError),
        op_error_info='input must be tensor',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((S,)),
            op_args=(1.0, make_((S, S)),),
            op_kwargs={"value": 0.1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError, TypeError),
        op_error_info='tensor1 must be tensor',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((S,)),
            op_args=(make_((S, S)), 1.0,),
            op_kwargs={"value": 0.1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError, TypeError),
        op_error_info='tensor2 must be tensor',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((S, 2, 3)),
            op_args=(make_((5, 7)), make_((4, 9, 6)),),
            op_kwargs={"value": 0.1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError, TypeError, AttributeError),
        op_error_info='shape not broadcast',
    )

def error_sample_inputs_tensor_addmv(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_ = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((2,)),
            op_args=(make_((2, 3, 5)), make_((3,)),),
            op_kwargs={"beta": 1, "alpha": 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of mat must be 2D',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((1, 3, 6)),
            op_args=(make_((1, 3)), make_((3,)),),
            op_kwargs={"beta": 1, "alpha": 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of input must be 1D',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((2,)),
            op_args=(make_((2, 3)), make_((3, 1, 2)),),
            op_kwargs={"beta": 1, "alpha": 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the dim of VEC must be 1D',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=2,
            op_args=(make_((2, 3)), make_((3,)),),
            op_kwargs={"beta": 1, "alpha": 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError, TypeError, AttributeError),
        op_error_info='the dim of input must be tensor',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((2,)),
            op_args=(make_((3, 1)), make_((1,)),),
            op_kwargs={"beta": 1, "alpha": 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='the mat.shape should be broadcast by input, vec',
    )

# ====================================linalg dynamic sample inputs functions===================================
def dynamic_sample_inputs_tensor_addr(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),
                         ms.Tensor(shape=None, dtype=dtype), 1, 1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(10,), dtype=dtype),
                             ms.Tensor(shape=(3,), dtype=dtype), 1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3,), dtype=dtype),
                    op_args=(ms.Tensor(shape=(3,), dtype=dtype), ms.Tensor(shape=(3,), dtype=dtype), 1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3), dtype=dtype),
                op_args=(ms.Tensor(shape=(None,), dtype=dtype),
                         ms.Tensor(shape=(None,), dtype=dtype), 1, 1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(1,), dtype=dtype), ms.Tensor(shape=(3,), dtype=dtype), 1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(10, ), dtype=dtype), ms.Tensor(shape=(3,), dtype=dtype),1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def dynamic_sample_inputs_tensor_addmm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    #mat1, mat2, input 2d
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),
                         ms.Tensor(shape=None, dtype=dtype), 1, 1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(10, 1), dtype=dtype),
                             ms.Tensor(shape=(1, 3), dtype=dtype), 1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 1), dtype=dtype),
                    op_args=(ms.Tensor(shape=(3, 2), dtype=dtype), ms.Tensor(shape=(2, 1), dtype=dtype), 1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=dtype),
                         ms.Tensor(shape=(None, None), dtype=dtype), 1, 1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(1, 2), dtype=dtype), ms.Tensor(shape=(2, 3), dtype=dtype), 1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(10, 1), dtype=dtype), ms.Tensor(shape=(1, 3), dtype=dtype),1, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
def basic_reference_inputs_tensor_exponential_func(op_info: OpInfo, dtype, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    generator = ms.Generator()
    generator.manual_seed(42)

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    base_shapes = (
        ((S,), 1, None),
        ((M, S), 1, generator),
        ((S, S, L), 1.0, generator),
        ((2, 0, 3, 2), 1.0, None),
        ((2, 3, 4, 1, 2), 2.0, generator),
        ((2, 1, 2, 2, 1, 2), 1.0, None),
        ((2, 1, 2, 2, 1, 2, 1), 2.0, generator),
    )
    #0d < input <8d
    for input_shape, lambd, generator in base_shapes:
        _input = make_arg(input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(lambd,),
            op_kwargs={"generator": generator},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_tensor_exponential_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    cases = (
        ((0,), 1, None),
        ((20,), 1.0, None),
        ((812,),1.0, None),
        ((1029, 917), 1.0, None),
    )
    for input_shape, lambd, generator in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input_shape),
            op_args=(lambd,),
            op_kwargs={"generator": generator},
            sample_name=f"{op_info.name}_{dtype}",
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(True,),
            op_kwargs={},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_shape=(1, 1, len(_large_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(1.0,),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 1, 3), t), dtype=dtype, device=device),
                    op_args=(1.0,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3, 4), discontiguous=discontiguous),
            op_args=(1.0,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_exponential_func(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (1.0, 1, None),#input is not tensor
        (make_arg((2, 1)), "1", None),#lambd is str
        (make_arg((2, 3, 4)), 1, True),#generator is bool
        (make_arg((2, 1)), [1,2], None),  # lambd is list
    )
    for input_shape, lambd, generator in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(lambd,),
                op_kwargs={'generator': generator,},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.exponential_ error values')

def dynamic_sample_inputs_tensor_exponential_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    #mat1, mat2, input 2d
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, None,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10, 3), dtype=dtype),
                    op_args=(1, None,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 1, 2), dtype=dtype),
                    op_args=(1, None,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

def dynamic_sample_inputs_tensor_addbmm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype),
                         ms.Tensor(shape=(None, None, None), dtype=dtype)),
                op_kwargs={"beta": 1, "alpha": 1},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(10, 8, 5), dtype=dtype), ms.Tensor(shape=(10, 5, 3), dtype=dtype)),
                    op_kwargs={"beta": 1, "alpha": 1},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 3), dtype=dtype),
                    op_args=(ms.Tensor(shape=(10, 6, 7), dtype=dtype), ms.Tensor(shape=(10, 7, 3), dtype=dtype)),
                    op_kwargs={"beta": 1, "alpha": 1},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


def dynamic_sample_inputs_tensor_addcdiv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),
                         ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={"value": 0.1},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 6, 7, 9), dtype=dtype),
                    op_args=(ms.Tensor(shape=(1, 6, 1, 1), dtype=dtype),
                             ms.Tensor(shape=(1, 1, 1, 1), dtype=dtype),),
                    op_kwargs={"value": 0.1},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 6, 8), dtype=dtype),
                    op_args=(ms.Tensor(shape=(1, 6, 1), dtype=dtype), ms.Tensor(shape=(1, 1, 1), dtype=dtype),),
                    op_kwargs={"value": 0.1},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

def dynamic_sample_inputs_tensor_cholesky_solve(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype), False,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_2D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4)),
                    op_args=(make_func(shape=(2, 3, 3)), False,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4 ,5, 7)),
                    op_args=(make_func(shape=(4 ,5, 5)), False,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3D",
                ),
            ),
        )

def basic_sample_inputs_tensor_baddbmm(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    # batch1, batch2 must 3D, input 0~3D
    cases = (
        ((), (1, 5, 3), (1, 3, 5), 1, 1),
        ((1,), (5, 4, 4), (5, 4, 4), 0.001, 1),
        ((1, 1), (1, 5, 3), (1, 3, 5), 0, 0),
        ((1, 5, 5), (1, 5, 3), (1, 3, 5), 0, 1.1),
        ((2, 4, 4), (2, 4, 0), (2, 0, 4), -2.6, -2.4),
        ((4, 6), (2, 4, 0), (2, 0, 6), 2.2, 2.8),
    )

    for input_shape, batch1_shape, batch2_shape, beta, alpha in cases:
        _input = make_arg(input_shape)
        _batch1 = make_arg(batch1_shape)
        _batch2 = make_arg(batch2_shape)
        slen = len(input_shape)
        # input dtype is bfloat16, beta/alpha is not int causes error in kbk
        if isinstance(beta, float) or isinstance(alpha, float):
            beta_alpha_float = True
        else:
            beta_alpha_float = False
        yield OpSampleInput(
            op_input= _input,
            op_args=(_batch1, _batch2),
            op_kwargs={'beta': beta, 'alpha': alpha},
            sample_name=f"{op_info.name}_{dtype}_beta_alpha_float_{beta_alpha_float}_{slen}d",
        )

def extra_sample_inputs_tensor_baddbmm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((8, 5, 5), (8, 5, 3), (8, 3, 5), 0.002, 0),
        ((7, 12, 12), (7, 12, 6), (7, 6, 12), -3, 2.1),
        ((128, 16, 16), (128, 16, 8), (128, 8, 16), 40, 8),
        ((11, 10, 10), (11, 10, 5), (11, 5, 10), -3.3, -2.8),
        ((512, 17, 17), (512, 17, 6), (512, 6, 17), 3.7, -1.2),
    )

    for input_shape, batch1_shape, batch2_shape, beta, alpha in cases:
        _input = make_arg(input_shape)
        _batch1 = make_arg(batch1_shape)
        _batch2 = make_arg(batch2_shape)
        # input dtype is bfloat16, beta/alpha is not int causes error in kbk
        if isinstance(beta, float) or isinstance(alpha, float):
            beta_alpha_float = True
        else:
            beta_alpha_float = False
        yield OpSampleInput(
            op_input=_input,
            op_args=(_batch1, _batch2),
            op_kwargs={'beta': beta, 'alpha': alpha},
            sample_name=f"{op_info.name}_{dtype}_beta_alpha_float_{beta_alpha_float}_extra",
        )
    # large mix small values
    values = small_large_values(dtype)
    if values:
        input_np_array = np.random.choice(values, size=(2, 3))
        np_array1 = np.random.choice(values, size=(1, 2, 3))
        np_array2 = np.random.choice(values, size=(1, 3, 3))
        for discontiguous in [True, False]:
            _batch1 = make_tensor_with_np_array(np_array1, dtype=dtype, discontiguous=discontiguous)
            _batch2 = make_tensor_with_np_array(np_array2, dtype=dtype, discontiguous=discontiguous)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(_batch1, _batch2),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            _batch1 = make_tensor_with_np_array(np.full((1, 4, 3), t), dtype=dtype)
            _batch2 = make_tensor_with_np_array(np.full((1, 3, 3), t), dtype=dtype)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((4, 3), t), dtype=dtype),
                op_args=(_batch1, _batch2),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_tensor_baddbmm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype), ms.Tensor(shape=(None, None, None), dtype=dtype)),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 2, 4)),
                    op_args=(make_func(shape=(1, 2, 4)), make_func(shape=(1, 4, 4))),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 4)),
                    op_args=(make_func(shape=(2, 4, 3)), make_func(shape=(2, 3, 4))),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype), ms.Tensor(shape=None, dtype=dtype)),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5,)),
                    op_args=(make_func(shape=(2, 3, 5)), make_func(shape=(2, 5, 5))),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4, 1)),
                    op_args=(make_func(shape=(3, 4, 6)), make_func(shape=(3, 6, 8))),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_tensor_baddbmm(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.bfloat16, device=device)
    shapes = (
        # input is not tensor
        (2, make_arg((1, 2, 3)), make_arg((1, 3, 2)), 1, 1),
        # batch1 is not tensor
        (make_arg((1, 2, 1)), 1, make_arg((1, 3, 2)), 0, 1.1),
        # batch2 is not tensor
        (make_arg((1, 2, 1)), make_arg((1, 2, 3)), 1, 2.1, 1),
        # beta is str
        (make_arg((1, 2, 1)), make_arg((1, 2, 3)), make_arg((1, 3, 2)), "1", 1),
        # alpha is tuple
        (make_arg((1, 2, 1)), make_arg((1, 2, 3)), make_arg((1, 3, 2)), 1, (1, 2)),
        # batch1 is (C, W, T), but batch2 is not (C, T, H)
        (make_arg((1, 2, 1)), make_arg((1, 2, 3)), make_arg((1, 2, 3)), 1, 2),
        # batch1 is not 3D
        (make_arg((1, 2, 1)), make_arg((1, 2)), make_arg((1, 2, 3)), 1, 2),
        # batch2 is not 3D
        (make_arg((1, 2, 1)), make_arg((1, 2, 3)), make_arg((3, 2)), 1, 2),
        # input can not broadcast with batch1/batch2
        (make_arg((1, 2, 2)), make_arg((1, 2, 3)), make_arg((1, 3, 3)), 1, 2),
        # input is 4D
        (make_arg((1, 1, 2, 3)), make_arg((1, 2, 3)), make_arg((1, 3, 3)), 1, 2),
    )
    for input, batch1, batch2, beta, alpha in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input,
                op_args=(batch1, batch2),
                op_kwargs={'beta': beta, 'alpha': alpha},
                sample_name=f"{op_info.name}_error_params",
            ),
            op_error_type=(AttributeError, TypeError, RuntimeError),
            op_error_info=f'{op_info.name}: error params.'
        )


def extra_sample_inputs_tensor_inner(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    cases = (
        (0,),
        (1, 0, 3),
        (),
        (812,),
    )
    for input_shape in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input_shape),
            op_args=(make_arg(shape=input_shape),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}",
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg((len(_float_small_values),)),),
            op_kwargs={},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_shape = (1, len(_large_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg((len(_large_values),)),),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 3), t), dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(np.full((3,), t), dtype=dtype, device=device),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3, 4), discontiguous=discontiguous),
            op_args=(make_arg((2, 3, 4), discontiguous=discontiguous),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_cross_func(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device)
    error_cases = (
        ((2, 2, 4), (2, 2, 4), None, ValueError, "None of the dim is 3"),
        ((2, 3, 4), (2, 3, 4), 0, ValueError, "value of dimension is not 3"),
        ((2, 3, 4), (2, 3, 5), 0, ValueError, "dimensions of input and other does not match"),
    )
    for x1, x2, dim, error_type, msg in error_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=x1),
                op_args=(make_arg(shape=x2),),
                op_kwargs={"dim":dim},
                sample_name=f"{op_info.name}"
            ),
            op_error_type=error_type,
            op_error_info=msg
        )

def dynamic_sample_inputs_tensor_cross(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # dynamic shape
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(3, None, None), dtype=dtype),
            op_args=(ms.Tensor(shape=(3, None, None), dtype=dtype),),
            op_kwargs={"dim": 1},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(3, 3, 2), ),
                op_args=(make_func(shape=(3, 3, 2), ),),
                op_kwargs={"dim": 1},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(3, 3, 5), ),
                op_args=(make_func(shape=(3, 3, 5), ),),
                op_kwargs={"dim": 1},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )

def basic_reference_inputs_tensor_square_matrix(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    L = LARGE_DIM_SIZE
    M = MEDIUM_DIM_SIZE
    S = SMALL_DIM_SIZE

    cases = (
        (S, S),
        (M, M),
        (L, L),
        (0, 2, 2),
        (3, S, S),
        (5, 7, M, M),
        (2, 6, 3 ,L, L),
        (3, 1, 2, 1, S, S)
    )

    # inverse not support 8d
    cases = cases +  ((1, 2, 1, 3, 1, 4, M, M), ) if 'eigvals' in op_info.name else cases

    for input_shape in cases:

        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_reference_inputs_tensor_square_matrix(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (3, 6, 6), (), {}, dtype=dtype, device=device,
                                 exclude_value=(-1e20, 1e20) if 'inverse' in op_info.name else () )

    yield OpSampleInput(
            op_input=make_tensor((512, 512), dtype=dtype),
            op_args=(),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}__large_dim',
        )

def dynamic_sample_inputs_tensor_square_matrix(op_info: OpInfo, dtype, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2 )),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
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
                    op_input=make_func(shape=(2, 2)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_sample_inputs_tensor_square_matrix(op_info: OpInfo, dtype, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func((3, 4)),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='x should be square(squares)'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func((3,)),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='x should be square(squares)'
    )

def basic_sample_inputs_tensor_ger(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    # input/vec2 must be 1D
    cases = (
        ((2,), (5,)),
        ((10,), (3,)),
        ((16,), (13,)),
        ((0,), (2,)),
    )

    for input_shape, vec2_shape in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg(vec2_shape),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_basic",
        )

def extra_sample_inputs_tensor_ger(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((128,), (256,)),
        ((512,), (128,)),
        ((1024,), (1024,)),
    )

    for input_shape, vec2_shape in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg(vec2_shape),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    # large mix small values
    values = small_large_values(dtype)
    if values:
        input_np_array = np.random.choice(values, size=(5,))
        for discontiguous in [True, False]:
            vec2 = make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(vec2,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            vec2 = make_tensor_with_np_array(np.full((4,), t), dtype=dtype)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((6,), t), dtype=dtype),
                op_args=(vec2,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_tensor_ger(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(ms.Tensor(shape=(None,), dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2,)),
                    op_args=(make_func(shape=(3,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5,)),
                    op_args=(make_func(shape=(6,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5,)),
                    op_args=(make_func(shape=(2,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(make_func(shape=(6,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_tensor_ger(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device)
    cases = (
        # input is not tensor
        (2, make_arg((3,))),
        # vec2 is not tensor
        (make_arg((3,)), 1),
        # input is not 1D tensor
        (make_arg((1, 2, 1)), make_arg((3,))),
        # vec2 is not 1D tensor
        (make_arg((3,)), make_arg((2, 1))),
        # the dtype of input and vec2 are not the same
        (make_arg((3,)), ms.Tensor(shape=(3,), dtype=ms.float16)),
    )
    for input, vec2 in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input,
                op_args=(vec2,),
                op_kwargs={},
                sample_name=f"{op_info.name}_error_params",
            ),
            op_error_type=(AttributeError, TypeError, ValueError),
            op_error_info=f'{op_info.name}: error params.'
        )

def basic_sample_inputs_tensor_geqrf(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    # input must be 2D
    cases = (
        (2, 5),
        (10, 3),
        (16, 13),
        (0, 3),
    )

    for input_shape in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_basic",
        )

def extra_sample_inputs_tensor_geqrf(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        (128, 50),
        (512, 100),
        (1024, 104),
    )

    for input_shape in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    # large mix small values
    values = small_large_values(dtype, exclude_values=op_info.exclude_value)
    if values:
        input_np_array = np.random.choice(values, size=(5, 4))
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
                op_input=make_tensor_with_np_array(np.full((6, 9), t), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_tensor_geqrf(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 9)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(15, 12)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_tensor_geqrf(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device)
    cases = (
        # input is not tensor
        # input is not 2D tensor
        2, make_arg((1, 2, 1)),
    )
    for input in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input,
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_error_input",
            ),
            op_error_type=(AttributeError, ValueError),
            op_error_info=f'{op_info.name}: error input.'
        )
