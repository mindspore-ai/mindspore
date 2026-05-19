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
"""Custom sample input functions for operation testing.

This module provides sample input builders for various ops, including
basic, extra, dynamic, and error case generators.
"""
import functools

import numpy as np
import torch
import random
import warnings
import math
import inspect
import itertools
import mindspore as ms
from mindspore import mint, mutable
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo, ReductionOpInfo, \
    extra_reference_inputs_unary_op_common_func
from tests.st.ops.share._op_info.op_info import (
    _generate_unary_op_extremal_value_tensor_inputs_func,
    _generate_unary_op_large_value_tensor_inputs_func,
    _generate_unary_op_small_value_tensor_inputs_func,
    _generate_unary_op_tensors_sample_inputs_func,
    _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func,
    _generate_binary_op_scalar_inputs_func,
    _generate_binary_op_extremal_value_tensor_inputs_func,
    basic_reference_inputs_binary_op_common_func,
    basic_reference_inputs_reduction_op_common_func,
    basic_reference_inputs_unary_op_common_func,
    extra_reference_inputs_binary_op_common_func,
    extra_reference_inputs_reduction_op_common_func,
)
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_extra_uint,
    dtypes_integral,
    dtype_to_np_dtype,
    op_dtype_need_convert,
    dtypes_extra_values
)

# ====================================custom sample inputs functions===================================

def basic_reference_inputs_empty(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    if device and device.lower() == "ascend":
        formatted_device = "Ascend"
    else:
        formatted_device = device
    yield OpSampleInput(
        op_input=ms.Tensor([[1, 2, 3], [4, 5, 6]]),
        op_args=(),
        op_kwargs={'dtype': dtype, 'device': formatted_device, 'pin_memory': False},
        sample_name=f"{op_info.name}_{dtype}_{device}_empty_like"
    )

def error_inputs_unary_op_mint_empty_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        # input, dtype, device,pin_memory
        ("1", dtype, device, False),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), "0", device, False),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), dtype, "1", False),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), dtype, device, "True"),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), dtype, "Ascend", True)
    ]
    for op_inputs, dtypes, devices, pin_memory in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(),
                op_kwargs={'dtype': dtypes, 'device': devices, 'pin_memory': pin_memory},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f"{op_info.name}_{dtype}_{device}_error_empty"
        )

# op_basic_reference_inputs_func for ops
def basic_sample_inputs_add_sub_ext(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate sample inputs for add/sub ops including extra alpha cases.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    '''
    yield from basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)

    S = SMALL_DIM_SIZE
    # Adds alpha kwarg cases
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    _input = make_arg((S, S))
    _other = make_arg((S, S))
    if dtype is not ms.bool_:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': 2},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': True},
            sample_name=op_info.name
        )

    neg_alpha = -0.1415 if (dtype.is_floating_point or dtype.is_complex) else -3
    if dtype in dtypes_extra_uint:
        neg_alpha = abs(neg_alpha)

    _input = make_arg((S, S))
    _other = make_arg((S, S))
    if dtype is not ms.bool_:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': neg_alpha},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': False},
            sample_name=op_info.name
        )

# op_dynamic_inputs_func for ops
def dynamic_sample_inputs_add_sub_ext(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate dynamic inputs for add/sub_ext ops.
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
        # add/sub_ext dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, 1, None), dtype=dtype),),
                op_kwargs={"alpha": mutable(input_data=3.3, dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5, 8, 5, 4)),
                    op_args=(make_func(shape=(5, 5, 8, 1, 4)),),
                    op_kwargs={"alpha": mutable(input_data=4.3, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 8, 8, 4)),
                    op_args=(make_func(shape=(9, 9, 8, 1, 4)),),
                    op_kwargs={"alpha": mutable(input_data=-2.1, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # add/sub_ext dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={"alpha": mutable(input_data=2.33, dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5)),
                    op_args=(make_func(shape=(5, 5)),),
                    op_kwargs={"alpha": mutable(input_data=9.6, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 7)),
                    op_args=(make_func(shape=(9, 9, 7)),),
                    op_kwargs={"alpha": mutable(input_data=10.10, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

# op_error_inputs_func for ops
def error_inputs_binary_op_ext_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for binary ops.
    '''
    # other shape does not match input
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.float32),
            op_args=(make_tensor(shape=(3,), dtype=ms.float32),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='other shape does not match input',
    )
    # other is not tensor or number
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.float32),
            op_args=((1, 2),),
            op_kwargs={'alpha':1},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='other is not tensor or number',
    )

# op_error_inputs_func for ops with alpha
def error_inputs_binary_op_with_alpha(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.int32),
            op_args=(make_tensor(shape=(2,), dtype=ms.int32),),
            op_kwargs={'alpha': 1.2},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='if alpha is of the float type,but input and other are not of the float type',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.int32),
            op_args=(make_tensor(shape=(2,), dtype=ms.int32),),
            op_kwargs={'alpha': True},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='if alpha is of the bool type,but input and other are not of the bool type',
    )

# op_error_inputs_func for ops mint div
def error_inputs_binary_op_mint_div(op_info: OpInfo, dtype=None, device=None, **kwargs):
    rounding_mode = ['floor', 'trunc']
    for error_dtype  in [ms.complex64, ms.complex128]:
        for value in rounding_mode:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_tensor(shape=(2,), dtype=error_dtype),
                    op_args=(make_tensor(shape=(2,), dtype=error_dtype),),
                    op_kwargs={'rounding_mode': value},
                    sample_name=op_info.name
                ),
                op_error_type=RuntimeError,
                op_error_info=f'dtype {dtype} not supported in rounding_mode {value} ',
            )

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

# op_error_inputs_func for ops
def error_inputs_unary_op_prelu_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for prelu ops.
    '''
    # num_parameters not supported float
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.tensor(np.array([1, 2, 3, 4, 5])),
            op_args=(),
            op_kwargs={'num_parameters': 1.0},
            sample_name=f"{op_info.name}_num_parameters_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'num_parameters error input is not supported for {op_info.name}',
    )

    # init not supported int
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.tensor(np.array([1, 2, 3, 4, 5])),
            op_args=(),
            op_kwargs={'init': (1, 0.25)},
            sample_name=f"{op_info.name}_init_type_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'init error input is not supported for {op_info.name}',
    )

# op_error_inputs_func for ops
def error_inputs_unary_op_leaky_relu_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for leaky_relu ops.
    '''
    # negative_slope not float or int
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2, 1, 2, 3), dtype=ms.float32),
            op_args=(),
            op_kwargs={'negative_slope': (1, 1.0)},
            sample_name=f"{op_info.name}_negative_slope_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'negative_slope error tpye is not supported for {op_info.name}',
    )

def error_inputs_gelu_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error approximate for gelu ops.

    Return:
        List[OpErrorInput]: add approximate kwargs error type
    '''
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor((2, 2,), dtype=ms.float32, device=device, ),
            op_kwargs={'approximate': 2},
            sample_name=f'{op_info.name}_{dtype}_approximate_error-dim_2_2',
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='gelu ops approximate not support int type',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor((2, 2,), dtype=ms.float32, device=device, ),
            op_kwargs={'approximate': 'tab'},
            sample_name=f'{op_info.name}_{dtype}_approximate_error-dim_2_2',
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='gelu ops approximate not support tab value',
    )

def error_inputs_reduction_keepdim_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error keepdim/dim for reduction ops.

    Return:
        List[OpErrorInput]: add approximate kwargs error type
    '''
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(1,), dtype=ms.float32, device=device),
            op_args=(),
            op_kwargs={'keepdim': 2},
            sample_name=f'{op_info.name}_{dtype}_keepdim_error',
        ),
        op_error_type=(TypeError, RuntimeError),
        op_error_info='reductions ops keepdim not support int type',
    )
    # dim isnot int, tuple(int), list(int)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2), dtype=ms.float32),
            op_args=(),
            op_kwargs={'dim': (1, 1.0)},
            sample_name=f"{op_info.name}_dim_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'dim error tpye is not supported for {op_info.name}',
    )

# op_error_inputs_func for ops
def error_inputs_reduction_op_normal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for mint.normal ops.
    '''
    # mean isnot float, Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2), dtype=ms.float32),
            op_args=(),
            op_kwargs={'mean': (1, 1.0)},
            sample_name=f"{op_info.name}_mean_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'mean error tpye is not supported for {op_info.name}',
    )
    # std isnot float, Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2), dtype=ms.float32),
            op_args=(),
            op_kwargs={'mean': 1.0, 'std': '1'},
            sample_name=f"{op_info.name}_std_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'std error tpye is not supported for {op_info.name}',
    )
    # std is less than 0
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2), dtype=ms.float32),
            op_args=(),
            op_kwargs={'std': -1.0},
            sample_name=f"{op_info.name}_std_value_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'std error value is not supported for {op_info.name}',
    )

    # std is nan
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 2), dtype=ms.float32),
            op_args=(),
            op_kwargs={'std': np.nan},
            sample_name=f"{op_info.name}_std_value_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'std error value is not supported for {op_info.name}',
    )

    # mean and std shape is different
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor((1, 3, 8, 8),  dtype=ms.float32),
            op_args=(make_tensor((3, 3, 3, 3), dtype=ms.float32, low=0),),
            sample_name=f"{op_info.name}_diff_shape",
        ),
        op_error_type=ValueError,
        op_error_info='mean std shape must be equal',
    )

def basic_sample_inputs_mint_repeat_interleave(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.repeat_interleave.

    Cases covered (aligned with PyTorch repeat_interleave samples, adapted to MindSpore):
      - scalar tensor with repeats as Python int
      - 3D tensor with repeats as Python int (no dim / with dim)
      - 3D tensor with repeats as Tensor (per-index repeats) with dim
      - 2D tensor with repeats as Tensor and explicit output_size
    '''
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    # () with repeats=2
    yield OpSampleInput(
        op_input=make_input(()),
        op_args=(2,),
        op_kwargs={},
        sample_name=f"{op_info.name}_scalar_repeats_int"
    )

    # (2, 3, 4) with repeats=2 (no dim)
    yield OpSampleInput(
        op_input=make_input((2, 3, 4)),
        op_args=(2,),
        op_kwargs={},
        sample_name=f"{op_info.name}_nd_repeats_int_nodim"
    )

    # (2, 3, 4) with repeats=2, dim=1
    yield OpSampleInput(
        op_input=make_input((2, 3, 4)),
        op_args=(2,),
        op_kwargs={'dim': 1},
        sample_name=f"{op_info.name}_nd_repeats_int_dim1"
    )

    # (2, 3, 4) with repeats as Tensor([0, 1, 2]), dim=1
    repeats_tensor_dim1 = ms.Tensor([0, 1, 2], dtype=ms.int32)
    yield OpSampleInput(
        op_input=make_input((2, 3, 4)),
        op_args=(repeats_tensor_dim1,),
        op_kwargs={'dim': 1},
        sample_name=f"{op_info.name}_nd_repeats_tensor_dim1"
    )

    # (4, 1) with repeats as Tensor([0,1,2,3]), dim=0, output_size=6
    repeats_tensor_dim0 = ms.Tensor([0, 1, 2, 3], dtype=ms.int32)
    yield OpSampleInput(
        op_input=make_input((4, 1)),
        op_args=(repeats_tensor_dim0,),
        op_kwargs={'dim': 0, 'output_size': 6},
        sample_name=f"{op_info.name}_nd_repeats_tensor_dim0_output_size"
    )

    # (3,) with repeats as random Tensor([...]), dim=0
    rand_repeats = ms.Tensor(np.random.randint(1, 10, (3,)), dtype=ms.int32)
    yield OpSampleInput(
        op_input=make_input((3,)),
        op_args=(rand_repeats,),
        op_kwargs={'dim': 0},
        sample_name=f"{op_info.name}_1d_rand_repeats_tensor_dim0"
    )

def _small_large_values(dtype):
    values = []
    if dtype == ms.uint8:
        values = dtypes_extra_values['_uint_small_values']
    elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
        values = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    else:
        if dtype in [ms.float16]:
            values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp16_large_values']
        else:
            float_vals = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp32_large_values']
            if dtype.is_complex:
                values = [complex(r, i) for r, i in itertools.product(float_vals, float_vals)]
            else:
                values = float_vals
    return values

def extra_sample_inputs_for_shape_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
        Generate extra sample inputs for shape func ops.

        Cases covered (aligned with PyTorch repeat_interleave samples, adapted to MindSpore):
          - inf/nan/-inf values
          - large/small values
          - contignous values
          - empty shape
        '''
    # large mix small values
    values = _small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )
    # empty shape
    yield OpSampleInput(
        op_input=make_tensor(shape=(1, 0, 3), dtype=dtype, device=device),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_shape",
    )

def extra_sample_inputs_mint_repeat_interleave(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate extra sample inputs for mint.repeat_interleave.

    Cases covered (aligned with PyTorch repeat_interleave samples, adapted to MindSpore):
      - inf/nan/-inf values
      - large/small values
      - contignous values
    '''
    # large mix small values
    for sample_input in extra_sample_inputs_for_shape_func(op_info, dtype, device):
        sample_input.op_args=(2, 0)
        yield sample_input

def dynamic_sample_inputs_mint_repeat_interleave(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.repeat_interleave.

    Args:
        op_info(OpInfo): Operator metadata.
        dtype(mindspore.dtype): Data type of tensors to generate.
        device(str): Target device.
        kwargs(dict): Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, None, None), dtype=dtype),
                op_args=(1,),
                op_kwargs={'dim': 0, 'output_size': None},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 3, 9, 7)),
                    op_args=(1,),
                    op_kwargs={'dim': 0, 'output_size': None},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 8, 4)),
                    op_args=(1,),
                    op_kwargs={'dim': 0, 'output_size': None},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1,),
                op_kwargs={'dim': 0, 'output_size': None},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 9, 7, 5, 5, 3, 4)),
                    op_args=(1,),
                    op_kwargs={'dim': 0, 'output_size': None},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 8, 4, 5, 9)),
                    op_args=(1,),
                    op_kwargs={'dim': 0, 'output_size': None},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1,),
                op_kwargs={'dim': 0, 'output_size': None},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 5, 5)),
                    op_args=(1,),
                    op_kwargs={'dim': 0, 'output_size': None},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2,)),
                    op_args=(1,),
                    op_kwargs={'dim': 0, 'output_size': None},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def error_inputs_other_op_mint_repeat_interleave_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, repeats, dim, output_size
        (make_tensor(shape=(2,), dtype=ms.float32), ("0", 1), {}),
        (make_tensor(shape=(2,), dtype=ms.float32), (2, "true"), {}),
        (make_tensor(shape=(2,), dtype=ms.float32), (2,), {"output_size": '1'})]
    for op_inputs, p_args, p_kwargs in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=p_args,
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.repeat_interleave error values'
        )

def error_inputs_func_tensor_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3, 4), dtype=ms.float32),
            op_args=(),
            op_kwargs={"start_dim":1, "end_dim":0},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='start_dim can not come after end_dim',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3, 4), dtype=ms.float32),
            op_args=(),
            op_kwargs={"start_dim": -4, "end_dim": 2},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='start_dim not in [-dim , dim-1]',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3, 4), dtype=ms.float32),
            op_args=(),
            op_kwargs={"start_dim": 0, "end_dim": 3},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='end_dim not in [-dim , dim-1]',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3, 4), dtype=ms.float32),
            op_args=(),
            op_kwargs={"start_dim": 1.1, "end_dim": 2},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='start_dim not int',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3, 4), dtype=ms.float32),
            op_args=(),
            op_kwargs={"start_dim": 0, "end_dim": 1.5},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='end_dim not int',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3, 4), dtype=ms.float32),
            op_args=(),
            op_kwargs={"order": "D" ,"start_dim": 0, "end_dim": 1.5},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='order is not C or F',
    )

def basic_sample_inputs_mint_arange(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.arange, aligned with PyTorch's sample_inputs_arange.
    Each tuple is (start, end, step), where None means "omit that positional".
    The outer dtype is forwarded via op_kwargs['dtype'] to fix output dtype (like PyTorch does).
    '''
    int_samples = (
        (-1, 2, 2),          # positive direction
        (2, -3, -1),         # negative direction
        (-3, -10, -2),       # additional negative direction with even step
        (1, 1, 1),           # start == end
        (1, 1, -1),          # start == end with negative step
        (0, -8, -4),         # divides evenly (negative)
        (1, 5, 2),           # divides evenly (positive)
        (False, True, True), # bool inputs
        (0, 1, None),        # default step
        (None, 3, None),     # default start (single-arg form)
    )

    def to_float(start, end, step):
        start = (start + 0.1) if start is not None else None
        end = end + 0.1
        step = float(step) if step is not None else None
        return start, end, step

    float_samples = (
        (0.0, -8.0 - 1e-6, -4.0),  # includes endpoint
        (1.0, 5.0 + 1e-6,  2.0),   # includes endpoint
        (0.0, -8.0,       -4.0),
        (1.0, 5.0,         2.0),
        *(to_float(s, e, t) for (s, e, t) in int_samples),
    )

    large_samples = (
        (0, 10000, None),
    )

    samples = int_samples
    # Only add float_samples when output dtype is floating-point;
    # for integer dtypes, mixing float ranges can cause length mismatches vs MindSpore behavior.
    if dtype is not None and getattr(dtype, "is_floating_point", False):
        samples += float_samples
    if dtype not in (ms.int8, ms.uint8):
        samples += large_samples

    for start, end, step in samples:
        if start is None:
            op_input = end
            op_args = ()
        else:
            op_input = start
            op_args = (end,) if step is None else (end, step)

        op_kwargs = {}
        if dtype is not None:
            op_kwargs['dtype'] = dtype

        yield OpSampleInput(
            op_input=op_input,
            op_args=op_args,
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_arange_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        ("0", 1, 2, None),
        (0, "1", 2, None),
        (0, 1, "2", None),
        (0, 1, 2, "float32")]
    for start, end, step, dtypes in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=(),
                op_args=(start, end, step),
                op_kwargs={'dtype': dtypes},
                sample_name=op_info.name,
            ),
            op_error_type=TypeError,
            op_error_info='other ops mint.arange error values'
        )

def basic_op_sample_inputs_dtype_grad_position_cover(basic_func, no_grad_dtype):
    def basic_op_sample_inputs(op_info,dtype, device=None, **input_kwargs):
        for sample in basic_func(op_info, dtype, device, **input_kwargs):
            if dtype == no_grad_dtype:
                sample.grad_position=()
            yield sample
    return basic_op_sample_inputs

def basic_op_sample_inputs_grad_position_cover(basic_func, no_grad_dtype):
    def basic_op_sample_inputs(op_info,dtype, device=None, **input_kwargs):
        for sample in basic_func(op_info, dtype, device, **input_kwargs):
            if dtype in no_grad_dtype:
                sample.grad_position=()
            yield sample
    return basic_op_sample_inputs

def sample_inputs_broadcast_to(op_info, dtype, device, **kwargs):
    S = SMALL_DIM_SIZE

    test_cases = (
        ((S, S, S, S, 1, 1), (S, S, S, S, S, S)),
        ((S, 1, 1), (S, S, S)),
        ((S, 1, S), (S, S, S)),
        ((S, 1), (S, S, S)),
        ((1,), (S, S, S)),
        ((1, S), (1, 1, S)),
        ((), ()),
        ((), (1, 3, 2)),
    )

    return (
        OpSampleInput(
            make_tensor(size, dtype=dtype, device=device, low=None, high=None),
            op_args=(shape,),
            sample_name=op_info.name,
        ) for size, shape in test_cases)


def error_inputs_other_op_mint_broadcast_to_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, shape,
        ("1", (1,)),
        (make_tensor(shape=(2,), dtype=ms.float32), ('0',)),
        (make_tensor(shape=(2,), dtype=ms.float32), ((-1, -2),))]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.broadcast_to error values'
        )

def extra_sample_inputs_mint_broadcast_to(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.broadcast_to
    Reference torch's module tests for broadcast_to.
    '''
    cases = (
        (ms.Tensor(np.full((7, 1, 7), np.inf), dtype=ms.float32), (7, 7, 7, 7),),
        (ms.Tensor(np.full((3, 1, 8, 9, 1, 5), np.nan), dtype=ms.float32), (3, 5, 8, 9, 1, 5),),
        (ms.Tensor(np.random.randn(2, 3, 3, 4, 1, 2, 1, 1), dtype=dtype), (2, 3, 3, 4, 1, 2, 3, 4),),
        (ms.Tensor(np.array([]), dtype=dtype), (0,),), #0 dim tensor
        (ms.Tensor(np.array([]).reshape(2, 0), dtype=dtype), (2, 0),),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(4, 6).astype(np.float32)[::2, :], dtype=dtype), (2, 6),),
        (ms.Tensor(np.random.randn(2, 3, 4), dtype=dtype), (2, 3, 4),),
        (ms.Tensor(np.array([np.finfo(np.float32).min]), dtype=dtype), (5,),),
        (ms.Tensor(np.array([np.finfo(np.float32).max]), dtype=dtype), (3, 2),),
    )

    for inputs, shapes in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(shapes,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_broadcast_to(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.broadcast_to.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, None, None), dtype=dtype),
                op_args=(ms.mutable((6, 3, 3, 3)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 1, 1)),
                    op_args=(ms.mutable((6, 3, 3, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 1)),
                    op_args=(ms.mutable((2, 3, 10, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.mutable((6, 3, 3, 3)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 1, 1)),
                    op_args=(ms.mutable((6, 3, 3, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 1)),
                    op_args=(ms.mutable((2, 3, 10, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.mutable((6, 3, 3, 3)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 6, 1, 1)),
                    op_args=(ms.mutable((1, 6, 3, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 6, 10, 1)),
                    op_args=(ms.mutable((2, 6, 10, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def sample_inputs_binary_cross_entropy(op_info, dtype, device, logits=False, **kwargs):
    S = SMALL_DIM_SIZE

    make = functools.partial(make_tensor, device=device, dtype=dtype)
    # Lower bounds must be greater than 'eps' defined in gradcheck.py::gradgradcheck() -> eps
    # otherwise perturbation calculation causes Tensor value to become negative triggering
    # a device-side hardware assertion
    make_prob = functools.partial(make, low=0, high=1)

    reductions = ("mean", "sum", "none")

    shapes_and_kwargs = [
        *[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))],
        # Now framework does not support passing parameters in an arbitrary order during the backward process.
        # *[((S, S), dict(reduction=reduction)) for reduction in reductions],
        *[((S, S), {"weight": make((S, S)), "reduction": reduction}) for reduction in reductions],
    ]

    if logits:
        shapes_and_kwargs.extend(
            [((S, S), {"reduction": reduction, "pos_weight": make((S, ), low=0)}) for reduction in reductions]
        )

    for shape, p_kwargs in shapes_and_kwargs:
        yield OpSampleInput(
            (make if logits else make_prob)(shape),
            op_args=(make_prob(shape),),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

def sample_inputs_binary_cross_entropy_with_logits(op_info, dtype, device, **kwargs):
    S = SMALL_DIM_SIZE

    make = functools.partial(make_tensor, device=device, dtype=dtype)
    make_prob = functools.partial(make, low=0, high=1)
    reductions = ("mean", "sum", "none")

    def make_weight_shape_kwargs():
        kwargs = []
        for shape in ((1,), (1, S), (S), (S, S)):
            kwargs.extend([((S, S), {"weight": make(shape), "reduction": reduction}) for reduction in reductions])
        return kwargs

    shapes_and_kwargs = [
        *[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))],
        # *[((S, S), dict(reduction=reduction)) for reduction in reductions],
        *make_weight_shape_kwargs(),
        # *[((S, S), dict(reduction=reduction, pos_weight=make((S,), low=0))) for reduction in reductions],
        *[
            ((S, S), {"weight": make((S, S)), "reduction": reduction, "pos_weight": make((S,), low=0)})
            for reduction in reductions
        ],
    ]

    for shape, p_kwargs in shapes_and_kwargs:
        yield OpSampleInput(
            make(shape),
            op_args=(make_prob(shape),),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

def sample_inputs_loss(op_info, dtype, device, **kwargs):
    S = SMALL_DIM_SIZE

    _make_tensor = functools.partial(make_tensor, device=device, dtype=dtype)

    # Although most losses also support the reduce and size_average combination instead of reduce, the former is
    # deprecated since 0.4.1 and thus is not tested
    shapes_and_kwargs = (
        ((), None),
        ((S,), {"reduction": "mean"}),
        ((S,), {"reduction": "sum"}),
        ((S,), {"reduction": "none"}),
        ((S, S), None),
        ((S, 1), None),
        ((S, S, S), None),
    )

    for shape, p_kwargs in shapes_and_kwargs:
        yield OpSampleInput(
            _make_tensor(shape), op_args=(_make_tensor(shape),), op_kwargs=p_kwargs, sample_name=op_info.name
        )

def sample_inputs_l1_loss(op_info, dtype, device, **kwargs):
    yield from sample_inputs_loss(op_info, dtype, device, **kwargs)

# Used for log_softmax, softmax, softmin
def sample_inputs_softmax_variant(
    op_info,
    dtype,
    device,
    use_zero_dimensions=True,
    **kwargs,
):
    S = SMALL_DIM_SIZE
    M = MEDIUM_DIM_SIZE

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = [
        ((S,), (0,)),
        ((S, S), (0,)),
        ((S, S), (1,)),
        ((S, S), (-1,)),
        ((S, M, S), (2,)),
        ((S, M, S, M), (-1,)),
        *([((S, 0, 0), (-1,))] if use_zero_dimensions else []),
    ]

    return (
        OpSampleInput(make_arg(shape), op_args=dim, op_kwargs=kwargs, sample_name=op_info.name) for shape, dim in cases
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

def sample_inputs_batchnorm1d(op_info, dtype, device, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)


    param_configs = [
        {"num_features": 9, "eps":1e-5, "momentum":0.1, "affine":True, "track_running_stats":True, "dtype":dtype},
        {"num_features": 9, "eps":0.001, "momentum":0.2, "affine":False, "track_running_stats":True, "dtype":dtype},
        {"num_features": 4, "eps": 0.001, "momentum": 0.3, "affine": True, "track_running_stats": False,
         "dtype": dtype},
        {"num_features": 4, "eps": 0.001, "momentum": None, "affine": False, "track_running_stats": False,
         "dtype": None}
    ]
    input_shapes = [
        (5, 9),
        (5, 9, 3),
        (3, 4, 1),
        (3, 4)
    ]
    for i, (shape, params) in enumerate(zip(input_shapes, param_configs)):
        yield OpSampleInput(
            make_func(shape),
            op_args=tuple(params.values()),
            sample_name=f"{op_info.name}_case{i+1}")


def sample_inputs_batchnorm2d(op_info, dtype, device, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    param_configs = [
        {"num_features": 4, "eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True, "dtype": dtype},
        {"num_features": 3, "eps": 0.001, "momentum": 0.2, "affine": False, "track_running_stats": True,
         "dtype": dtype},
        {"num_features": 2, "eps": 0.001, "momentum": 0.3, "affine": True, "track_running_stats": False,
         "dtype": dtype},
        {"num_features": 4, "eps": 0.001, "momentum": None, "affine": False, "track_running_stats": False,
         "dtype": None}
    ]
    input_shapes = [
        (8, 4, 5, 3),
        (1, 3, 3, 4)
    ]
    for i, (shape, params) in enumerate(zip(input_shapes, param_configs)):
        yield OpSampleInput(
            make_func(shape),
            op_args=tuple(params.values()),
            sample_name=f"{op_info.name}_case{i + 1}")

def sample_inputs_batchnorm3d(op_info, dtype, device, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    param_configs = [
        {"num_features": 3, "eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True, "dtype": dtype},
        {"num_features": 3, "eps": 0.001, "momentum": 0.2, "affine": False, "track_running_stats": True,
         "dtype": dtype},
        {"num_features": 2, "eps": 0.001, "momentum": 0.3, "affine": True, "track_running_stats": False,
         "dtype": dtype},
        {"num_features": 4, "eps": 0.001, "momentum": None, "affine": False, "track_running_stats": False,
         "dtype": None}
    ]
    input_shapes = [
        (8, 3, 4, 8, 5),
        (1, 3, 3, 4, 5)
    ]
    for i, (shape, params) in enumerate(zip(input_shapes, param_configs)):
        yield OpSampleInput(
            make_func(shape),
            op_args=tuple(params.values()),
            sample_name=f"{op_info.name}_case{i + 1}")


def extra_sample_inputs_mint_nn_batchnorm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.nn.BatchNorm op where input tensor inf nan.
    '''
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        _large_values = dtypes_extra_values['_fp32_large_values']
        if op_info.name == "mint.nn.BatchNorm3d":
            S = (8, 5, 4, 8, 5)
            SM = (1, 1, 1, 1, len(_float_small_values))
            LG = (1, 1, 1, 1, len(_large_values))
            ZS = (0, 5, 4, 8, 5)
        elif op_info.name == "mint.nn.BatchNorm2d":
            S = (8, 5, 4, 3)
            SM = (1, 1, 1, len(_float_small_values))
            LG = (1, 1, 1, len(_large_values))
            ZS = (0, 5, 4, 3)
        else:
            S = (9, 5, 3)
            SM = (1, 1, len(_float_small_values))
            LG = (1, 1, len(_large_values))
            ZS = (0, 5, 3)
        total_elements = int(np.prod(S))
        case = (
            (ms.Tensor(np.full(S, np.nan), dtype=dtype),
             {"num_features": 5, "eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True,
              "dtype": dtype}),
            (ms.Tensor(np.full(S, np.inf), dtype=dtype),
             {"num_features": 5, "eps": 0.001, "momentum": 0.2, "affine": False, "track_running_stats": True,
              "dtype": dtype}),
            (ms.Tensor(np.arange(total_elements).reshape(S).astype(np.float32)[:, ::2, :], dtype=dtype),
             {"num_features": 3, "eps": 0.001, "momentum": 0.1, "affine": True, "track_running_stats": True,
              "dtype": dtype}),
            (ms.Tensor(np.ones(S), dtype=dtype),
             {"num_features": 5, "eps": 0.001, "momentum": 0.1, "affine": True, "track_running_stats": True,
              "dtype": dtype})
        )

        for inputs, p_kwargs in case:
            yield OpSampleInput(
                op_input=inputs,
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            )

        input_values = np.array(_float_small_values).reshape(SM)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(),
            op_kwargs= {"num_features": 1, "eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": False,
                        "dtype": dtype},
            sample_name=op_info.name,
        )

        input_values = np.array(_large_values).reshape(LG)
        #910A loss error
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                op_args=(),
                op_kwargs={"num_features": 1, "eps": 1e-5, "momentum": 0.1, "affine": True,
                           "track_running_stats": False,
                           "dtype": dtype},
                sample_name=op_info.name,
            )

        input_values = ms.Tensor(np.array([]).reshape(ZS), dtype=dtype)
        yield OpSampleInput(
            op_input=input_values,
            op_args=(),
            op_kwargs={"num_features": 5, "eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True,
                       "dtype": dtype},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_nn_batchnorm_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    base_params={
        'num_features': 4,
        'eps': 1e-5,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'dtype': None
    }
    error_cases=[
        ('op_input', ["1"], True),
        ('num_features', ["4", "5"], False),
        ('eps', ["1e-5"], False),
        ('momentum', ["0.1"], False),
        # ('affine', ["True"], False),
        # ('track_running_stats', ["True"], False), affine, track_running_stats set not bool type, not raise error
        #('dtype', ['none', ms.complex128, ms.complex64, ms.int8], False),
    ]
    for param_name, error_values, is_op_input in error_cases:
        for error_value in error_values:
            test_params = base_params.copy()
            if is_op_input:
                op_input = error_value
            else:
                op_input = ms.tensor([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]])
                test_params[param_name] = error_value
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=op_input,
                    op_args=(test_params['num_features'], test_params['eps'], test_params['momentum'],
                             test_params['affine'], test_params['track_running_stats'], test_params['dtype']),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info='other ops mint.nn.batchnorm error values'
            )

def sample_inputs_batch_norm(op_info, dtype, device, **kwargs):
    S = SMALL_DIM_SIZE

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    make_arg_without_requires_grad = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as: input shape, kwargs for training, momentum, eps
    cases: tuple[tuple[int], dict] = (  # type: ignore[assignment]
        ((S, S, S), {'training': True, 'momentum': 0.5, 'eps': 0.6}),
        ((3, 2, 4), {'training': False, 'momentum': -1.2}),
        ((3, 1), {'training': True, 'momentum': 0.0}),
        # empty value is not supported in mindspore
        # ((0,), {'training': True}),
        # ((0,), {'training': False}),
        ((3, 2, 3, 4), {'training': True, 'momentum': -1.0, 'eps': 0.5}),
        ((3, 2, 3, 4), {'training': False, 'momentum': -1.0, 'eps': 0.5}),
        ((2, 1), {}),
    )

    for input_shape, p_kwargs in cases:
        # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)

        yield OpSampleInput(
            make_arg(input_shape),
            op_args=(
                running_mean,
                running_var,
                weight,
                bias
            ),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

    # Checking for permutations of weights and biases as `None`
    is_training = [True, False, False]

    for training in is_training:
        yield OpSampleInput(
            make_arg(input_shape),
            op_args=(
                running_mean,
                running_var,
                make_arg(channels),
                make_arg(channels)
            ),
            op_kwargs={'training': training},
            sample_name=op_info.name,
        )

    # Test case for no optional kwargs
    # running_mean and running_var are required in evaluation mode (training: False) but not in training mode
    # yield OpSampleInput(
    #     make_arg((1, 2, 3)), op_args=(None, None, None, None), op_kwargs={'training': True}, sample_name=op_info.name
    # )

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

def _generate_tensor_is_contiguous_contiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate contiguous tensor inputs for tensor.is_contiguous.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering contiguous/non-contiguous memory layouts.
    """

    shapes = (
        (1,),
        (3, 1),
        (1, 2, 3),
    )

    for shape, discontiguous in itertools.product(shapes, [False]):
        yield OpSampleInput(
            op_input=make_tensor(shape, dtype=dtype, device=device, discontiguous=discontiguous),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_discontiguous_tensor_inputs",
        )

def extra_sample_inputs_tensor_is_contiguous_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate comprehensive reference inputs for tensor.is_contiguous.

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
    yield from _generate_unary_op_tensors_sample_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with small value
    if dtype != ms.bool_:
        yield from _generate_unary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_unary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # contiguous tensors
    yield from _generate_tensor_is_contiguous_contiguous_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensor with extremal value
    if dtype.is_floating_point or dtype.is_complex:
        yield from _generate_unary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)

def basic_sample_inputs_mint_conv3d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.nn.functional.conv3d.
    Reference torch's sample_inputs_conv3d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((9, 4, 5, 2, 3), (9, 4, 4, 1, 1), (9,), {'stride': 1, 'dilation': 1, 'groups': 1, 'padding': 'same' }),
        # With defaults
        ((1, 1, 4, 4, 4), (1, 1, 2, 2, 2), None, {}),
    )

    for input_shape, weight_shape, bias_shape, case_kwargs in cases:
        # Create kwargs dict with proper order: bias, stride, padding, dilation, groups
        # This matches the interface: conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        op_kwargs = {}
        # bias comes first
        if bias_shape is not None:
            op_kwargs['bias'] = make_arg(bias_shape, low=-1000, high=-0.1, random_method='uniform')
        else:
            op_kwargs['bias'] = None
        # Then other parameters in order: stride, padding, dilation, groups
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        # Batched input (5D: N, C, D, H, W)
        yield OpSampleInput(
            op_input=make_arg(input_shape, low=-1000, high=-0.1, random_method='uniform'),
            op_args=(make_arg(weight_shape, low=-1000, high=-0.1, random_method='uniform'),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )
        # Unbatched input (4D: C, D, H, W)
        yield OpSampleInput(
            op_input=make_arg(input_shape[1:], low=-1000, high=-0.1, random_method='uniform'),
            op_args=(make_arg(weight_shape, low=-1000, high=-0.1, random_method='uniform'),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )


# sample inputs functions for Tensor.masked_scatter
def basic_sample_inputs_tensor_masked_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for Tensor.masked_scatter.
    Reference torch's sample_inputs_masked_scatter.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    make_mask = functools.partial(make_tensor, device=device, dtype=ms.bool_)

    S = SMALL_DIM_SIZE

    # Case 1: Same shape mask
    yield OpSampleInput(
        op_input=make_arg((S, S)),
        op_args=(make_mask((S, S)), make_arg((S, S))),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Case 2: Broadcastable mask (1D mask for 2D input)
    yield OpSampleInput(
        op_input=make_arg((S, S)),
        op_args=(make_mask((S,)), make_arg((S, S))),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Case 3: Different shapes
    yield OpSampleInput(
        op_input=make_arg((2, 3)),
        op_args=(make_mask((2, 3)), make_arg((6,))),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Case 4: 1D input
    yield OpSampleInput(
        op_input=make_arg((8,)),
        op_args=(make_mask((8,)), make_arg((8,))),
        op_kwargs={},
        sample_name=op_info.name,
    )

    yield OpSampleInput(
        op_input=make_arg((8,)),
        op_args=(make_mask((1,)), make_arg((8,))),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Case 5: 3D input
    yield OpSampleInput(
        op_input=make_arg((2, 3, 4)),
        op_args=(make_mask((2, 3, 4)), make_arg((24,))),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Case 6: Broadcastable mask (scalar-like mask)
    yield OpSampleInput(
        op_input=make_arg((S, S)),
        op_args=(make_mask((1,)), make_arg((S, S))),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # Case 7: 0D input
    yield OpSampleInput(
        op_input=make_arg(()),
        op_args=(make_mask(()), make_arg(())),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # Case 8: 8D input
    yield OpSampleInput(
        op_input=make_arg((8, 5, 7, 9, 8, 5, 5, 5)),
        op_args=(make_mask((8, 5, 7, 9, 8, 5, 5, 5)), make_arg((8, 5, 7, 9, 8, 5, 5, 5))),
        op_kwargs={},
        sample_name=op_info.name,
    )


# sample inputs functions for Tensor.masked_scatter_ (reuse masked_scatter inputs)
def basic_sample_inputs_tensor_masked_scatter_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for Tensor.masked_scatter_.
    Reuse the same inputs as masked_scatter since they have the same interface.
    '''
    # Reuse the same sample inputs as masked_scatter
    yield from basic_sample_inputs_tensor_masked_scatter(op_info, dtype=dtype, device=device, **kwargs)


# sample inputs functions for Tensor.add_
def basic_sample_inputs_tensor_add_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for Tensor.add_.
    Reference torch's sample_inputs_add_sub and basic_reference_inputs_binary_op_common_func.
    '''
    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Alpha kwarg cases
    _input = make_arg((50,))
    _other = make_arg((1,))
    if dtype is not ms.bool_:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': -2.5},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': True},
            sample_name=op_info.name
        )

    neg_alpha = -0.1415 if (dtype.is_floating_point or dtype.is_complex) else -3
    if dtype in dtypes_extra_uint:
        neg_alpha = abs(neg_alpha)

    _input = make_arg((S, S))
    _other = make_arg((S, S))
    if dtype is not ms.bool_:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': neg_alpha},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': False},
            sample_name=op_info.name
        )

# sample inputs functions for mindspore.mint.nn.Conv1d
def basic_sample_inputs_mint_nn_conv1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.Conv1d.
    Reference torch's module tests for Conv1d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-2, high=2)

    # Cases: (input_shape, (in_channels, out_channels, kernel_size), kwargs_dict)
    cases = (
        # No batch dimension
        ((8, 5), (8, 4, 2), {'stride': 1, 'padding': (12,), 'dilation': 3, 'groups': 2, 'bias': True,
                              'padding_mode': 'zeros', 'dtype': dtype, 'weight_init': make_arg((4, 4, 2)),
                              'bias_init': make_arg((4,))}),
        ((5, 8, 4), (8, 4, 1), {'stride': 1, 'padding': 'same', 'dilation': 6, 'groups': 4, 'bias': True,
                                'padding_mode': 'replicate', 'dtype': dtype, 'weight_init': make_arg((4, 2, 1)),
                                'bias_init': make_arg((4,))}),
        ((8, 9, 7), (9, 3, 4), {'stride': 1, 'padding': 'valid', 'dilation': 1, 'groups': 3, 'bias': True,
                                'padding_mode': 'reflect', 'dtype': dtype, 'weight_init': make_arg((3, 3, 4)),
                                'bias_init': make_arg((3,))})
    )

    for input_shape, (in_channels, out_channels, kernel_size), case_kwargs in cases:
        op_kwargs = {}
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        if 'bias' in case_kwargs:
            op_kwargs['bias'] = case_kwargs['bias']
        if 'padding_mode' in case_kwargs:
            op_kwargs['padding_mode'] = case_kwargs['padding_mode']
        if 'dtype' in case_kwargs:
            op_kwargs['dtype'] = case_kwargs['dtype']
        if 'weight_init' in case_kwargs:
            op_kwargs['weight_init'] = case_kwargs['weight_init']
        if 'bias_init' in case_kwargs:
            op_kwargs['bias_init'] = case_kwargs['bias_init']
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(in_channels, out_channels, kernel_size),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_nn_conv1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.nn.Conv1d.
    Reference torch's module tests for Conv1d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-9, high=0)

    cases = (
        ((8, 5), (8, 4, 2), {'stride': 1, 'padding': "same", 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros', 'dtype': dtype, 'weight_init': make_arg((4, 8, 2)),
                             'bias_init': make_arg((4,))}),
        ((5, 8, 4), (8, 4, 1), {'stride': 1, 'padding': 'valid', 'dilation': 2, 'groups': 2, 'bias': True,
                                'padding_mode': 'replicate', 'dtype': dtype, 'weight_init': make_arg((4, 4, 1)),
                                'bias_init': make_arg((4,))}),
        ((8, 9, 7), (9, 3, 4), {'stride': 1, 'padding': 2, 'dilation': 1, 'groups': 3, 'bias': True,
                                'padding_mode': 'zeros', 'dtype': dtype, 'weight_init': make_arg((3, 3, 4)),
                                'bias_init': make_arg((3,))}),
        ((8, 9, 7), (9, 3, 4), {'stride': 1, 'padding': (1,), 'dilation': 1, 'groups': 3, 'bias': True,
                                'padding_mode': 'reflect', 'dtype': dtype, 'weight_init': make_arg((3, 3, 4)),
                                'bias_init': make_arg((3,))}),
        ((8, 9, 7), (9, 3, 4), {'stride': 1, 'padding': [0], 'dilation': 1, 'groups': 1, 'bias': True,
                                'padding_mode': 'reflect', 'dtype': dtype, 'weight_init': make_arg((3, 9, 4)),
                                'bias_init': make_arg((3,))})
    )

    for input_shape, (in_channels, out_channels, kernel_size), case_kwargs in cases:
        op_kwargs = {}
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        if 'bias' in case_kwargs:
            op_kwargs['bias'] = case_kwargs['bias']
        if 'padding_mode' in case_kwargs:
            op_kwargs['padding_mode'] = case_kwargs['padding_mode']
        if 'dtype' in case_kwargs:
            op_kwargs['dtype'] = case_kwargs['dtype']
        if 'weight_init' in case_kwargs:
            op_kwargs['weight_init'] = case_kwargs['weight_init']
        if 'bias_init' in case_kwargs:
            op_kwargs['bias_init'] = case_kwargs['bias_init']
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(in_channels, out_channels, kernel_size),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values))
        weight_values = np.array(_float_small_values[:1]).reshape(1, 1, 1)
        bias_values = np.array([0.0])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1, 1, 1),
            op_kwargs={'stride': 1, 'padding': 'valid', 'dilation': 1, 'groups': 1, 'bias': True,
                       'padding_mode': 'reflect', 'dtype': dtype, 'weight_init':
                           make_tensor_with_np_array(weight_values, dtype=dtype, device=device),
                       'bias_init': make_tensor_with_np_array(bias_values, dtype=dtype, device=device),},
            sample_name=op_info.name,
        )

    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values))
        weight_values = np.array(_large_values[:1]).reshape(1, 1, 1)
        bias_values = np.array([0.0])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1, 1, 1),
            op_kwargs={'stride': 1, 'padding': 'valid', 'dilation': 1, 'groups': 1, 'bias': True,
                       'padding_mode': 'reflect', 'dtype': dtype, 'weight_init':
                           make_tensor_with_np_array(weight_values, dtype=dtype, device=device),
                       'bias_init': make_tensor_with_np_array(bias_values, dtype=dtype, device=device), },
            sample_name=op_info.name,
        )

    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 3), t)
                weight_values = np.ones((1, 1, 1))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(1, 1, 1),
                    op_kwargs={'stride': 1, 'padding': 'valid', 'dilation': 1, 'groups': 1, 'bias': False,
                               'padding_mode': 'reflect', 'dtype': dtype, 'weight_init':
                                   make_tensor_with_np_array(weight_values, dtype=dtype, device=device),
                               'bias_init': None, },
                    sample_name=op_info.name,
                )

    for discontiguous in (True,):
        input_shape = (1, 3, 8)
        weight_shape = (3, 3, 3)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(1, 1, 1),
            op_kwargs={'stride': 1, 'padding': 'valid', 'dilation': 1, 'groups': 1, 'bias': False,
                       'padding_mode': 'reflect', 'dtype': dtype, 'weight_init':
                           make_arg(weight_shape, discontiguous=discontiguous),
                       'bias_init': None, },
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_nn_conv1d_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    base_params = {
        'in_channels': 120,
        'out_channels': 240,
        'kernel_size': 4,
        'stride': 1,
        'padding': "same",
        'dilation': 1,
        'groups': 1,
        'bias': True,
        'padding_mode': "zeros",
        'dtype': ms.float32,
        'weight': make_tensor(shape=(240, 120, 4), dtype=ms.float32),
        'bias_init': make_tensor(shape=(240,), dtype=ms.float32),
    }

    error_configs={
        'op_input':{"1"},
        'in_channels': {"120", -1},
        'out_channels': {"240", -2},
        'kernel_size': {"4", -3},
        'stride': {"1", -1},
        'padding': {True, -1},
        'dilation': {"1", -1},
        'groups': {"1", -1},
        'bias': {"True", 1},
        'padding_mode': {True, -1},
        'dtype': {"float32"},
        'weight': {"1"},
        'bias_init': {"1"},
    }
    for param_name, error_values in error_configs.items():
        for error_value in error_values:
            test_params = base_params.copy()
            if param_name == "op_input":
                op_input = error_value
            else:
                op_input = ms.Tensor(np.random.randn(2,120,10).astype(np.float32))
                test_params[param_name] = error_value
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=op_input,
                    op_args=(test_params['in_channels'], test_params['out_channels'], test_params['kernel_size'],
                             test_params['stride'], test_params['padding'], test_params['dilation'],
                             test_params['groups'], test_params['bias'], test_params['padding_mode'],
                             test_params['dtype']),
                    op_kwargs={'weight': test_params['weight'], 'bias_init': test_params['bias_init']},
                    sample_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError),
                op_error_info='other ops mint.nn.conv1d error values',
            )

# basic sample inputs functions for mindspore.mint.nn.Conv2d
def basic_sample_inputs_mint_nn_conv2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # test for 3d
    yield OpSampleInput(
        op_input=make_arg((6, 7, 8)),
        op_args=(6, 6, 2),
        op_kwargs={"stride":1, "padding":0, "dilation":1, "groups":1, "bias":True,
                   "padding_mode":"zeros", "dtype":None,
                   "weight_init": make_arg((6, 6, 2, 2)),
                   "bias_init":make_arg((6, ))},
        sample_name=f"{op_info.name}_{dtype}_input_without_batch",
    )
    #test for dtype
    if dtype != ms.float32:
        yield OpSampleInput(
            op_input=make_arg((10, 6, 7, 8)),
            op_args=(6, 6, 1),
            op_kwargs={"stride":1, "padding":0, "dilation":1, "groups":1, "bias":True,
                       "padding_mode":"zeros", "dtype": ms.float32,
                       "weight_init": make_arg((6, 6, 1, 1)),
                       "bias_init": make_arg((6,))},
            sample_name=f"{op_info.name}_{dtype}_input_with_dtype",
        )
    # Cases: (input_shape, (in_channels, out_channels, kernel_size), kwargs_dict)
    batch_size=1
    in_channels=6
    out_channels=9
    cases={
        "kernel_size": ((2, 3), ),
        "stride": (1, (2, 3,), ),
        "padding": ("valid", "same", (3, 2)),
        "dilation": ((2, 2),),
        'groups':(3, ),
        'padding_mode': ("zeros", "reflect", "replicate"),
    }
    for kernel_size, stride, dilation, padding, groups, padding_mode in itertools.product(
        cases["kernel_size"],
        cases["stride"],
        cases["dilation"],
        cases["padding"],
        cases["groups"],
        cases["padding_mode"],
    ):
        if padding=="same" and stride != 1:
            continue
        yield OpSampleInput(
            op_input=make_arg((batch_size, in_channels, 7, 8)),
            op_args=(in_channels, out_channels, kernel_size,),
            op_kwargs={"stride": stride, "padding": padding, "dilation": dilation, "groups": groups, "bias": True,
                       "padding_mode": padding_mode,  "dtype": dtype,
                       "weight_init": make_arg((out_channels, in_channels//groups, *kernel_size)),
                       "bias_init":make_arg((out_channels,))},
            sample_name=f"{op_info.name}_{dtype}",
        )

#extra sample inputs functions for mindspore.mint.nn.Conv2d
def extra_sample_inputs_mint_nn_conv2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # big batch size and channels
    yield OpSampleInput(
        op_input=make_arg((1024, 120, 10, 10)),
        op_args=(120, 120, 4),
        op_kwargs={"weight_init": make_arg((120, 120, 4, 4)), "bias_init": make_arg((120,))},
        sample_name=f"{op_info.name}_{dtype}_input_with_big_input",
    )
    # case with small/large values
    float_vals = dtypes_extra_values['_float_small_values']
    float_vals =float_vals + dtypes_extra_values['_fp16_large_values']
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8)), dtype=dtype),
        op_args=(6, 9, 2),
        op_kwargs={"weight_init":ms.Tensor(np.random.choice(float_vals, size=(9, 6, 2, 2)), dtype=dtype),
                   "bias": True,"bias_init": ms.Tensor(np.random.choice(float_vals, size=(9,)), dtype=dtype)},
        sample_name=f"{op_info.name}_{dtype}_small_large_values",
    )
    #case with inf/nan values
    for t in [np.nan, np.inf, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((6, 7, 8), t), dtype=dtype),
            op_args=(6, 9, 2),
            op_kwargs={"weight_init": make_arg((9, 6, 2, 2)),
                       "bias": True, "bias_init":make_arg((9, ))},
            sample_name=f"{op_info.name}_{dtype}_{t}",
        )

    #case with discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8), discontiguous=discontiguous),
            op_args=(6, 9, 2),
            op_kwargs={"weight_init": make_arg((9, 6, 2, 2), discontiguous=discontiguous),
                       "bias": True, "bias_init": make_arg((9, ), discontiguous=discontiguous)},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}",
        )


# error sample inputs functions for mindspore.mint.nn.Conv2d
def error_sample_inputs_mint_nn_conv2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #weight does not have right shape
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((120, 7, 8)),
            op_args=(120, 120, 2),
            op_kwargs={"weight_init":make_arg((120, 2, 2)), "bias":False},
            sample_name=f"{op_info.name}_weight_shape_not_right",
        ),
        op_error_type=ValueError,
        op_error_info='weight does not have right shape'
    )
    #input_channel of net not match with input_channel of input,
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((110, 7, 8)),
            op_args=(120, 120, 2),
            op_kwargs={"bias":False},
            sample_name=f"{op_info.name}_input_channel_not_match",
        ),
        op_error_type=ValueError,
        op_error_info='input channel not match'
    )
    #weight not match with channel
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((120, 7, 8)),
            op_args=(120, 120, 2),
            op_kwargs={"weight_init":make_arg((120, 110, 2, 2)), "bias":False},
            sample_name=f"{op_info.name}_weight_channel_not_match",
        ),
        op_error_type=ValueError,
        op_error_info='input channel not match with weight'
    )
    #kernel_value is bigger than the input_height/input_width
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((120, 7, 8)),
            op_args=(120, 120, 10),
            op_kwargs={"bias":False},
            sample_name=f"{op_info.name}_kernel_value_bigger",
        ),
        op_error_type=ValueError,
        op_error_info='kernel_value is bigger than the input_height/input_width'
    )
    #padding is same but stride is not 1
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((10, 7, 8)),
            op_args=(10, 10, 2),
            op_kwargs={"stride": 2, "padding": "same", "bias":False},
            sample_name=f"{op_info.name}_stride_2_padding_same",
        ),
        op_error_type=ValueError,
        op_error_info='padding is same but stride is not 1'
    )
    #input_channels/out_put_channel/kernel is 0
    for name, op_args in zip(("input_channel", "output_channel", 'kernel_size'),
                             ((0, 10, 1),(10, 0, 1),(10, 10, 0))):
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((10, 7, 8)),
                op_args=op_args,
                op_kwargs={"bias":False},
                sample_name=f"{op_info.name}_{name}_0",
            ),
            op_error_type=ValueError,
            op_error_info=f'{name} is small than 1'
        )
    #stride/diliation/ is 0
    for name in ['stride', 'dilation']:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((10, 7, 8)),
                op_args=(10, 10, 1),
                op_kwargs={name:0, "bias":False},
                sample_name=f"{op_info.name}_{name}_0",
            ),
            op_error_type=ValueError,
            op_error_info=f'{name} is small than 1'
        )
    #input_channel/output_channel cannot be divided by groups:
    for name, shape, op_args in zip(("input_channel", "output_channel"),
                                    ((10, 7, 8), (9, 7, 8)),
                                    ((10, 9, 1), (9, 10, 1))):
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape),
                op_args=op_args,
                op_kwargs={"groups":3, "bias":False},
                sample_name=f"{op_info.name}_{name}_groups_not_divided",
            ),
            op_error_type=ValueError,
            op_error_info=f'{name} cannot be divided by groups'
        )
# wrapper function for mindspore.mint.nn.Linear
def mint_nn_linear_ms(op_input, in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=None):
    """Wrapper for mindspore.mint.nn.Linear."""
    linear_layer = mint.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        weight_init=weight_init,
        bias_init=bias_init,
        dtype=dtype
    )
    return linear_layer(op_input)


def nn_linear_torch(op_input, in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=None):
    """Wrapper for torch.nn.Linear.
    Note: weight_init, bias_init, and dtype are kept for interface consistency but not used in PyTorch.
    """
    linear_layer = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype
    )
    linear_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        linear_layer.bias = torch.nn.Parameter(bias_init)
    return linear_layer(op_input)


# sample inputs functions for mindspore.mint.nn.Linear
def basic_sample_inputs_mint_nn_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.Linear.
    Reference torch's module_inputs_torch_nn_Linear.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-2, high=2)

    # Cases: (input_shape, (in_features, out_features), kwargs_dict)
    cases = (
        ((4, 10), (10, 8), {'bias': True, 'weight_init': make_arg((8, 10)),
                             'bias_init': make_arg((8,)), 'dtype': dtype}),
        ((4, 2), (2, 20), {
            'bias': True, 'weight_init': make_arg((20, 2)),
            'bias_init': make_arg((20,)), 'dtype': dtype
        }),
    )

    for input_shape, (in_features, out_features), case_kwargs in cases:
        # Create kwargs dict with proper order: bias, weight_init, bias_init, dtype
        # This matches the interface: Linear(in_features, out_features, bias=True, weight_init=None,
        # bias_init=None, dtype=None)
        op_kwargs = {}
        if 'bias' in case_kwargs:
            op_kwargs['bias'] = case_kwargs['bias']
        if 'weight_init' in case_kwargs and case_kwargs['weight_init'] is not None:
            op_kwargs['weight_init'] = case_kwargs['weight_init']
        if 'bias_init' in case_kwargs and case_kwargs['bias_init'] is not None:
            op_kwargs['bias_init'] = case_kwargs['bias_init']
        if 'dtype' in case_kwargs and case_kwargs['dtype'] is not None:
            op_kwargs['dtype'] = case_kwargs['dtype']
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(in_features, out_features),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_nn_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.nn.linear op where input tensor inf nan.
    '''
    case = (
        (ms.Tensor(np.full((8, 1), np.inf), dtype=dtype),
         {"in_features": 1, "out_features": 2, "bias": True,
          "weight_init": ms.Tensor(np.random.randn(2, 1), dtype=dtype),
          "bias_init": ms.Tensor(np.random.randn(2), dtype=dtype), "dtype": dtype}),
        (ms.Tensor(np.full((8, 8, 1), -np.inf), dtype=dtype),
         {"in_features": 1, "out_features": 2, "bias": True,
          "weight_init": ms.Tensor(np.random.randn(2, 1), dtype=dtype),
          "bias_init": ms.Tensor(np.random.randn(2), dtype=dtype), "dtype": dtype}),
        (ms.Tensor(np.random.randn(7, 3, 9, 7, 9, 4), dtype=dtype),
         {"in_features": 4, "out_features": 15, "bias": True,
          "weight_init": ms.Tensor(np.random.randn(15, 4), dtype=dtype),
          "bias_init": ms.Tensor(np.random.randn(15,), dtype=dtype), "dtype": dtype}),
        (ms.Tensor(np.array([]).reshape(0, 4), dtype=dtype),
         {"in_features": 4, "out_features": 8, "bias": True,
          "weight_init": ms.Tensor(np.random.randn(8, 4), dtype=dtype),
          "bias_init": ms.Tensor(np.random.randn(8), dtype=dtype), "dtype": dtype}),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype),
         {"in_features": 4, "out_features": 6, "bias": True,
          "weight_init": ms.Tensor(np.random.randn(6, 4), dtype=dtype),
          "bias_init": ms.Tensor(np.random.randn(6), dtype=dtype), "dtype": dtype}),
        # continuous
        (ms.Tensor(np.ones((10, 6)), dtype=dtype),
         {"in_features": 6, "out_features": 4, "bias": True,
          "weight_init": ms.Tensor(np.ones((4, 6)), dtype=dtype),
          "bias_init": ms.Tensor(np.ones(4), dtype=dtype), "dtype": dtype}),
        (ms.Tensor(np.full((6, 4), np.finfo(np.float32).max), dtype=dtype),
         {"in_features": 4, "out_features": 8, "bias": True,
          "weight_init": ms.Tensor(np.full((8, 4), 1e-5), dtype=dtype),
          "bias_init": ms.Tensor(np.full(8, 1e-5), dtype=dtype), "dtype": dtype}),
        (ms.Tensor(np.full((4, 3), np.finfo(np.float32).min), dtype=dtype),
         {"in_features": 3, "out_features": 5, "bias": True,
          "weight_init": ms.Tensor(np.full((5, 3), -1e-5), dtype=dtype),
          "bias_init": ms.Tensor(np.full(5, -1e-5), dtype=dtype), "dtype": dtype}),
    )

    for inputs, p_kwargs in case:
        yield OpSampleInput(
            op_input=inputs,
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_nn_linear_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    base_params = {
        'in_features': 3,
        'out_features': 4,
        'bias': True,
        'weight_init': None,
        'bias_init': None,
        'dtype': dtype
    }
    base_input = ms.tensor([[180, 234, 154], [244, 48, 247]], ms.float32)
    error_configs={
        'op_input':{
            'values': ['1'],
        },
        'in_features': {
            'values': ['3', -1],
        },
        'out_features': {
            'values': ['4', -2],
        },
        'bias': {
            'values': ['False', 1],
        },
        'weight_init': {
            'values': ['InvalidInitializer'],
        },
        'bias_init': {
            'values': ['InvalidInitializer'],
        },
        'dtype': {
            'values': ['float32'],
        }
    }
    for param_name, config in error_configs.items():
        for error_value in config['values']:
            test_params = base_params.copy()
            if param_name == "op_input":
                op_input = error_value
            else:
                op_input = base_input
                test_params[param_name] = error_value
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=op_input,
                    op_args=(test_params['in_features'], test_params['out_features'], test_params['bias'],
                             test_params['weight_init'], test_params['bias_init'], test_params['dtype']),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError),
                op_error_info='other ops mint.nn.linear error values'
            )

# sample inputs functions for mint.nn.functional.linear
def basic_sample_inputs_mint_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.nn.functional.linear.
    Reference torch's sample_inputs_linear.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-2, high=2)

    # Features options: (in_features, out_features)
    features_options = [(3, 4), (8, 8)]
    # Batch shape options: no batch, 1D, 2D, etc.
    batch_options = [
        [],  # no batch
        [2],
        [8],
        [2, 3],
    ]

    for has_bias, (in_feat, out_feat), batch_shape in \
            itertools.product([True, False], features_options, batch_options):
        input_tensor = make_arg(batch_shape + [in_feat])
        weight = make_arg([out_feat, in_feat])
        if not has_bias:
            yield OpSampleInput(
                op_input=input_tensor,
                op_args=(weight,),
                op_kwargs={},
                sample_name=op_info.name,
            )
            continue

        bias = make_arg([out_feat])
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(weight,),
            op_kwargs={'bias': bias},
            sample_name=op_info.name,
        )

    yield OpSampleInput(
        op_input=make_arg([2, 1, 2, 1, 2]),
        op_args=(make_arg([4, 2]),),
        op_kwargs={},
        sample_name=op_info.name,
    )
    yield OpSampleInput(
        op_input=make_arg([4,]),
        op_args=(make_arg([9, 4]),),
        op_kwargs={'bias': make_arg([9])},
        sample_name=op_info.name,
    )



# sample inputs functions for mint.nn.functional.conv1d
def basic_sample_inputs_mint_conv1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.nn.functional.conv1d.
    Reference torch's sample_inputs_conv1d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((1, 3, 4), (3, 3, 3), (3,),
            {'stride': (2,), 'padding': 2, 'groups': 1}),
        ((2, 4, 8), (2, 2, 3), (2,),
            {'stride': 3, 'padding': 1, 'groups': 2, 'dilation': 2}),
        ((1, 4, 5), (1, 4, 3), None,
            {'stride': (2,), 'padding': "valid"}),
        ((3, 7), (2, 3, 1), (2,),
            {'stride': (5,), 'padding': (2,), 'groups': 1, 'dilation': (1,)}),
        # With defaults
        ((1, 4, 5), (3, 4, 3), None, {}),
    )

    for input_shape, weight_shape, bias_shape, case_kwargs in cases:
        # Create kwargs dict with proper order: bias, stride, padding, dilation, groups
        # This matches the interface: conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        op_kwargs = {}
        # bias comes first
        if bias_shape is not None:
            op_kwargs['bias'] = make_arg(bias_shape)
        else:
            op_kwargs['bias'] = None
        # Then other parameters in order: stride, padding, dilation, groups
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        # Batched input (3D: N, C, L)
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg(weight_shape),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )


# sample inputs functions for mint.nn.functional.conv2d
def basic_sample_inputs_mint_conv2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.nn.functional.conv2d.
    Reference torch's sample_inputs_conv2d.
    '''
    # conv2d uses single-side precision comparison schema, so we set low=0, high=9 here
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=9)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, groups, dilation)
    cases = (
        ((1, 3, 4, 4), (3, 3, 3, 3), (3,),
            {'stride': (2, 2), 'padding': 2, 'groups': 1}),
        ((1, 2, 4, 3), (4, 2, 3, 4), None,
            {'stride': 2, 'padding': 1, 'groups': 1}),
        ((12, 6, 4, 6), (8, 3, 4, 7), (8,),
            {'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 2}),
        ((1, 4, 5, 5), (3, 4, 3, 3), None, {}),  # all defaults: bias=None, stride=1, padding=0, dilation=1, groups=1
    )

    for input_shape, weight_shape, bias_shape, case_kwargs in cases:
        # Create kwargs dict with proper order: bias, stride, padding, dilation, groups
        # This matches the interface: conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        op_kwargs = {}
        op_kwargs['bias'] = make_arg(bias_shape) if bias_shape is not None else None
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        # Batched input
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg(weight_shape),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )
        # Unbatched input (3D: C, H, W)
        yield OpSampleInput(
            op_input=make_arg(input_shape[1:]),
            op_args=(make_arg(weight_shape),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_conv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.nn.functional.conv2d.
    Covers: various padding modes, dilation, groups, small/large/extremal value
            tensors, discontiguous tensors, type promotion.
    Skips: scalar inputs (conv2d requires tensor inputs).
    '''
    conv_dim = kwargs.get('conv_dim')

    conv_dim = {'conv1d': 1, 'conv2d': 2, 'conv3d': 3}[conv_dim]

    # conv2d uses single-side precision comparison schema, so we set low=-9, high=0 here
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-9, high=0)

    # ---- various padding modes, dilation, groups ----
    cases = (
        ((1, 3, *((8,)*conv_dim)), (6, 3, *((3,)*conv_dim)), (6,),
            {'stride': 1, 'padding': 'same', 'groups': 1}),
        ((2, 4, *((6,)*conv_dim)), (4, 4, *((3,)*conv_dim)), None,
            {'stride': 1, 'padding': 'valid'}),
        ((1, 6, *((8,)*conv_dim)), (2, 3, *((3,)*conv_dim)), (2,),
            {'stride': 2, 'padding': 1, 'dilation': 2, 'groups': 2}),
        ((3, 8, *((10,)*conv_dim)), (4, 2, *((5,)*conv_dim)), (4,),
            {'stride': (2, *((1,)*(conv_dim-1))), 'padding': (2,)* conv_dim, 'dilation': (1, *((2,)*(conv_dim-1))),
             'groups': 4}),
        ((1, 4, *((7,)*conv_dim)), (8, 1, *((3,)*conv_dim)), (8,),
            {'stride': 1, 'padding': 1, 'groups': 4}),
        ((2, 3, *((5,)*conv_dim)), (3, 3, *((1,)*conv_dim)), None,
            {'stride': 1, 'padding': 0, 'groups': 1}),
    )

    for input_shape, weight_shape, bias_shape, case_kwargs in cases:
        op_kwargs = {}
        if bias_shape is not None:
            op_kwargs['bias'] = make_arg(bias_shape)
        else:
            op_kwargs['bias'] = None
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg(weight_shape),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

    for input_shape, weight_shape, bias_shape, case_kwargs in cases:
        op_kwargs = {}
        if bias_shape is not None:
            op_kwargs['bias'] = make_arg(bias_shape)
        else:
            op_kwargs['bias'] = None
        if 'stride' in case_kwargs:
            op_kwargs['stride'] = case_kwargs['stride']
        if 'padding' in case_kwargs:
            op_kwargs['padding'] = case_kwargs['padding']
        if 'dilation' in case_kwargs:
            op_kwargs['dilation'] = case_kwargs['dilation']
        if 'groups' in case_kwargs:
            op_kwargs['groups'] = case_kwargs['groups']
        yield OpSampleInput(
            op_input=make_arg(input_shape[1:]),
            op_args=(make_arg(weight_shape),),
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values), *((1,)*(conv_dim-1)))
        weight_values = np.array(_float_small_values[:1]).reshape(1, 1, *((1,)*conv_dim))
        bias_values = np.array([0.0])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(weight_values, dtype=dtype, device=device),),
            op_kwargs={'bias': make_tensor_with_np_array(bias_values, dtype=dtype, device=device),
                       'stride': 1, 'padding': 0},
            sample_name=op_info.name,
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values), *((1,)*(conv_dim-1)))
        weight_values = np.array(_large_values[:1]).reshape(1, 1, *((1,)*conv_dim))
        bias_values = np.array([0.0])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(weight_values, dtype=dtype, device=device),),
            op_kwargs={'bias': make_tensor_with_np_array(bias_values, dtype=dtype, device=device),
                       'stride': 1, 'padding': 0},
            sample_name=op_info.name,
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, *((3,)*conv_dim)), t)
                weight_values = np.ones((1, 1, *((1,)*conv_dim)))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(weight_values, dtype=dtype, device=device),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 0},
                    sample_name=op_info.name,
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (1, 3, *((8,)*conv_dim))
        weight_shape = (3, 3, *((3,)*conv_dim))
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(make_arg(weight_shape, discontiguous=discontiguous),),
            op_kwargs={'bias': None, 'stride': 1, 'padding': 1},
            sample_name=op_info.name,
        )



def dynamic_sample_inputs_mint_conv2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.nn.functional.conv2d.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    # conv2d uses single-side precision comparison schema, so we set low=0, high=9 here
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(6, 3, 3, 3), dtype=dtype),),
                op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8, 8)),
                    op_args=(make_func(shape=(6, 3, 3, 3)),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 10)),
                    op_args=(make_func(shape=(6, 3, 3, 3)),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8, 8)),
                    op_args=(make_func(shape=(6, 3, 3, 3)),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 10)),
                    op_args=(make_func(shape=(6, 3, 3, 3)),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 2},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 6, 8, 8)),
                    op_args=(make_func(shape=(4, 3, 3, 3)),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 2},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 6, 10, 10)),
                    op_args=(make_func(shape=(4, 3, 5, 5)),),
                    op_kwargs={'bias': None, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 2},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def dynamic_sample_inputs_mint_binary_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0.1, high=0.9)
    reductions = ("mean", "sum", "none")
    for reduction in reductions:
        if not kwargs.get("only_dynamic_rank", False):
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                    op_args=(ms.Tensor(shape=(None, None, None, None), dtype=dtype),),
                    op_kwargs={'weight':None,'reduction':reduction},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(1, 3, 8, 8)),
                        op_args=(make_func(shape=(1, 3, 8, 8)),),
                        op_kwargs={'weight':None,'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 10, 10)),
                        op_args=(make_func(shape=(2, 3, 10, 10)),),
                        op_kwargs={'weight':None,'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                ),
            )

        if not kwargs.get("only_dynamic_shape", False):
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(ms.Tensor(shape=None, dtype=dtype),),
                    op_kwargs={'weight':None,'reduction':reduction},
                    sample_name=f"{op_info.name}_dynamic_rank_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(1, 3, 8, 8)),
                        op_args=(make_func(shape=(1, 3, 8, 8)),),
                        op_kwargs={'weight':None,'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 10, 10)),
                        op_args=(make_func(shape=(2, 3, 10, 10)),),
                        op_kwargs={'weight':None,'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input",
                    ),
                ),
            )

def dynamic_sample_inputs_mint_mse_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=1)
    reductions = ("mean", "sum", "none")
    for reduction in reductions:
        if not kwargs.get("only_dynamic_rank", False):
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                    op_args=(ms.Tensor(shape=(None, None, None, None), dtype=dtype),),
                    op_kwargs={'reduction':reduction},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(1, 3, 8, 8)),
                        op_args=(make_func(shape=(1, 3, 8, 8)),),
                        op_kwargs={'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 10, 10)),
                        op_args=(make_func(shape=(1, 1, 10, 10)),),
                        op_kwargs={'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                ),
            )
        if not kwargs.get("only_dynamic_shape", False):
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(ms.Tensor(shape=None, dtype=dtype),),
                    op_kwargs={'reduction':reduction},
                    sample_name=f"{op_info.name}_dynamic_rank_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(1, 3, 8, 8)),
                        op_args=(make_func(shape=(1, 3, 8, 8)),),
                        op_kwargs={'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 10, 10)),
                        op_args=(make_func(shape=(1, 1, 10, 10)),),
                        op_kwargs={'reduction':reduction},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input",
                    ),
                ),
            )

def dynamic_sample_inputs_mint_layer_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=1)
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, 4, 5), dtype=dtype),
                op_args=(ms.mutable((3, 4, 5)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4, 5)),
                    op_args=(ms.mutable((3, 4, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 4, 5)),
                    op_args=(ms.mutable((3, 4, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.mutable((3, 4, 5)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4, 5)),
                    op_args=(ms.mutable((3, 4, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 4, 5)),
                    op_args=(ms.mutable((3, 4, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def dynamic_sample_inputs_mint_softmax(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={'dim': 2},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4, 5)),
                    op_args=(),
                    op_kwargs={'dim': 2},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 4, 5)),
                    op_args=(),
                    op_kwargs={'dim': 2},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={'dim': 2},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4, 5)),
                    op_args=(),
                    op_kwargs={'dim': 2},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 4, 5)),
                    op_args=(),
                    op_kwargs={'dim': 2},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_inputs_mint_nn_functional_conv2d_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for mint.nn.functional.conv2d.
    Covers semantic and constraint violations from the API doc:
      - input rank not 3 or 4
      - weight rank not 4
      - bias rank not 1 / bias size mismatch
      - stride < 1 / stride tuple length != 2
      - padding < 0 / padding tuple length != 2 / padding="same" with stride!=1
      - dilation < 1 / dilation tuple length != 2
      - groups constraint violations (C_in % groups != 0, C_out % groups != 0)
      - kernel size larger than input spatial dims
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    base_input = make_arg((1, 4, 8, 8))
    base_weight = make_arg((6, 4, 3, 3))
    base_bias = make_arg((6,))

    # input rank not 3 or 4
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 4, 8, 8, 2)),
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='input rank not 3 or 4',
    )

    # weight rank not 4
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(make_arg((6, 4, 3, 3, 1)),),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='weight rank not 4',
    )

    # bias rank not 1
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': make_arg((6, 2)), 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='bias rank not 1',
    )

    # bias size mismatch (C_out=6 but bias has 3 elements)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': make_arg((3,)), 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='bias size != C_out',
    )

    # stride < 1
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': -1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='stride < 1',
    )

    # stride tuple length != 2
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': (1, 1, 1), 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='stride tuple length != 2',
    )

    # padding < 0
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': -1, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='padding < 0',
    )

    # padding tuple length != 2
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': (1, 1, 1), 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='padding tuple length != 2',
    )

    # dilation < 1
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': 0, 'dilation': -1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='dilation < 1',
    )

    # dilation tuple length != 2
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': 0, 'dilation': (1, 1, 1), 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='dilation tuple length != 2',
    )

    # C_in % groups != 0
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=base_input,
            op_args=(base_weight,),
            op_kwargs={'bias': base_bias, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 3},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='C_in % groups != 0',
    )

    # C_out % groups != 0
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 6, 8, 8)),
            op_args=(make_arg((5, 3, 3, 3)),),
            op_kwargs={'bias': make_arg((5,)), 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 2},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='C_out % groups != 0',
    )

    # kernel size larger than input spatial dims
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 4, 3, 3)),
            op_args=(make_arg((6, 4, 5, 5)),),
            op_kwargs={'bias': make_arg((6,)), 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='kernel size larger than input spatial dims',
    )

    # empty tensor (spatial dim is 0)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 4, 0, 8)),
            op_args=(make_arg((6, 4, 3, 3)),),
            op_kwargs={'bias': make_arg((6,)), 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='empty tensor input',
    )


# sample inputs functions for Tensor.repeat (method)
def basic_sample_inputs_tensor_repeat(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for Tensor.repeat (method), aligned to torch's coverage:
      - scalar tensor with 1D repeat
      - 1D tensor with zero and positive repeats
      - 2D/3D tensors with per-dimension repeats
      - keep sizes length equal to input.ndim for MindSpore compatibility
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    # Scalar () with repeat(2) -> shape (2,)
    yield OpSampleInput(
        op_input=make_x(()),
        op_args=(2,),
        op_kwargs={},
        sample_name=f"{op_info.name}_scalar_repeat_2",
    )

    # Scalar () with repeat(0, 0) -> shape (0, 0)
    yield OpSampleInput(
        op_input=make_x(()),
        op_args=([0, 0],),
        op_kwargs={},
        sample_name=f"{op_info.name}_scalar_repeat_0x0",
    )

    # 1D (S,) with repeat(0) -> empty
    yield OpSampleInput(
        op_input=make_x((S,)),
        op_args=(0,),
        op_kwargs={},
        sample_name=f"{op_info.name}_1d_repeat_0",
    )

    # 1D (S,) with repeat(3) -> shape (S*3,)
    yield OpSampleInput(
        op_input=make_x((S,)),
        op_args=(3,),
        op_kwargs={},
        sample_name=f"{op_info.name}_1d_repeat_3",
    )

    # 2D (S, S) with repeat(1, 2) -> (S, 2S)
    yield OpSampleInput(
        op_input=make_x((S, S)),
        op_args=([1, 2],),
        op_kwargs={},
        sample_name=f"{op_info.name}_2d_repeat_1x2",
    )

    # 2D (S, S) with repeat(2, 1) -> (2S, S)
    yield OpSampleInput(
        op_input=make_x((S, S)),
        op_args=([2, 1],),
        op_kwargs={},
        sample_name=f"{op_info.name}_2d_repeat_2x1",
    )

    # 3D (2, 1, S) with repeat(2, 3, 1) -> (4, 3, S)
    yield OpSampleInput(
        op_input=make_x((2, 1, S)),
        op_args=([2, 3, 1],),
        op_kwargs={},
        sample_name=f"{op_info.name}_3d_repeat_2x3x1",
    )

# sample inputs functions for chunk
def basic_sample_inputs_mint_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.chunk op.
    '''
    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S, S), (2,)),     # chunks only
        ((S, S, S), (S, 1)),   # chunks=S, dim=1
        ((S, S, S), (S, -1)),  # chunks=S, dim=-1
    )
    for shape, args in cases:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.chunk op.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # 0D is not supported by mint.chunk, will be added to error cases later
    cases = (
        ((3,), 5, 0),                      # 1D: ragged last (chunks > size), dim=0
        ((3, 6), 3, 1),                    # 2D: equal split, dim=1
        ((2, 3, 4), 3, -1),                # 3D: non-equal split, last dim
        ((2, 2, 2, 2), 2, 0),              # 4D: equal split, dim=0
        ((2, 2, 2, 3, 2), 2, 3),           # 5D: non-equal split, dim=3
        ((2, 2, 2, 2, 2, 3), 4, 5),        # 6D: ragged last (chunks > size), dim=5
        ((2, 2, 2, 2, 2, 2, 3), 2, -1),    # 7D: non-equal split, last dim
        ((2, 2, 2, 2, 2, 2, 2, 4), 2, 7),  # 8D: equal split, dim=7
    )
    for shape, chunks, dim in cases:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(chunks, dim),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic sample inputs for mint.chunk op.
    '''
    # chunk infer does NOT support dynamic rank and does NOT support the case
    # when the dimension specified by `dim` is dynamic. So we ensure the `dim`
    # dimension size is static at compile time, while other dimensions can be dynamic.
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Case A: 2D input, dim=1 static, other dim dynamic
        chunks, dim = 4, 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 6), dtype=dtype),  # dim=1 is static 6
                op_args=(chunks, dim),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 6)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Case B: 3D input, dim=0 static, other dims dynamic
        chunks, dim = 3, 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim=0 is static 6
                op_args=(chunks, dim),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

def error_inputs_other_op_mint_chunk_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #tensor, dim,dtype
        ("1", (1, 0)),
        (make_tensor(shape=(2,), dtype=ms.float32), ('1', 0)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "0"))
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.chunk error values'
        )

def extra_reference_inputs_operator_eq_common_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from extra_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)
    # non-tensor case
    yield OpSampleInput(
        op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
        op_args=([1, 2, 3],),
        op_kwargs={},
        sample_name=op_info.name + "_non_tensor",
    )
    # 0-d float64 tensor equality: guards against scalar-path precision loss
    # where a 0-d tensor is misrouted to EqScalar and truncated to float32.
    scalar_val = np.array(3.1415926, dtype=np.float64)
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(scalar_val, dtype=ms.float64, device=device),
        op_args=(make_tensor_with_np_array(scalar_val, dtype=ms.float64, device=device),),
        op_kwargs={},
        sample_name=op_info.name + "_0d_float64_eq",
    )

def error_inputs_tensor_operator_eq(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """Extra error cases for operator '=='."""
    # 1) Non-broadcastable shapes
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
            op_args=(make_tensor(shape=(4, 5), dtype=ms.float32, device=device),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, TypeError, RuntimeError),
        op_error_info="Shapes (2, 3) and (4, 5) are not broadcastable.",
    )

# sample inputs functions for gather
def basic_sample_inputs_mint_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.gather op.
    Cover 1D/2D common cases.
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    # index helper: default int64, low=0; call with shape=..., high=extent
    make_index = functools.partial(make_tensor, device=device, low=0, dtype=ms.int64)

    # 1D: dim=0, index length <= extent
    x_shape = (S,)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(S,), high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 2D: dim=0 and dim=1
    x_shape = (S, S)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(S, S), high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(1, make_index(shape=(S, S // 2), high=x_shape[1])),
        op_kwargs={},
        sample_name=op_info.name,
    )


def basic_sample_inputs_mint_unique(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.unique op.
    Reference torch's sample_inputs_unique:
      - sizes: (), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S)
      - flags: sorted/return_inverse/return_counts in {False, True}
      - dims: None, -2, -1, 0, 1, 2 with validity checks
      - inputs: all-zeros, mixed 0/1, random values
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    sizes = ((), (S,), (S, S, S), (S, 1, S), (S, 0, S))
    flag_cases = [(sorted_flag, ret_inv, ret_cnt) for sorted_flag in (False, True)
                  for ret_inv in (False, True) for ret_cnt in (False, True)]
    #TODO: mint.unique has different behavior when dim is None or -1, skip it temporarily.
    dim_cases = (-2, 0, 2)

    for shape in sizes:
        rank = len(shape)
        for sorted_flag, ret_inv, ret_cnt in flag_cases:
            for dim in dim_cases:
                # skip invalid dim per rank
                if rank == 0 and dim is not None:
                    continue
                if dim is not None and (dim < -rank or dim >= rank):
                    continue
                # for shapes with a zero dimension, if dim selects a different axis than the zero axis, skip
                if 0 in shape and dim is not None:
                    zero_axis = shape.index(0)
                    dim_norm = dim if dim >= 0 else dim + rank
                    if dim_norm != zero_axis:
                        continue

                # Use positional args as per API (no keyword args for unique)
                # Order: (sorted, return_inverse, return_counts, dim)
                unique_args = (sorted_flag, ret_inv, ret_cnt, dim)

                # (1) all-zero tensor
                zero_np_dtype = dtype_to_np_dtype[dtype]
                zeros_np = np.zeros(shape, dtype=zero_np_dtype)
                zeros_ms = make_tensor_with_np_array(zeros_np, dtype=dtype, device=device)
                yield OpSampleInput(
                    op_input=zeros_ms,
                    op_args=unique_args,
                    op_kwargs={},
                    sample_name=op_info.name,
                )

                # (2) mixed 0/1 values (then cast to dtype)
                ones_zeros_np = np.random.randint(0, 2, size=shape).astype(zero_np_dtype)
                mixed_ms = make_tensor_with_np_array(ones_zeros_np, dtype=dtype, device=device)
                yield OpSampleInput(
                    op_input=mixed_ms,
                    op_args=unique_args,
                    op_kwargs={},
                    sample_name=op_info.name,
                )

                # (3) many random values
                yield OpSampleInput(
                    op_input=make_x(shape),
                    op_args=unique_args,
                    op_kwargs={},
                    sample_name=op_info.name,
                )

def extra_sample_inputs_mint_unique(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.unique
    Reference torch's module tests for unique.
    '''
    cases = (
        (ms.Tensor(np.random.randn(50, 7, 17, 1, 5, 2, 4, 5), dtype=dtype), True, False, False, 3),
        (ms.Tensor(np.full((8, 8), np.nan), dtype=dtype), True, False, False, 0),
        (ms.Tensor(np.full((8, 8), np.inf), dtype=dtype), True, False, False, 0),
        (ms.Tensor(np.random.randn(4094, 64), dtype=dtype), True, False, False, 0),
        (ms.Tensor(np.array([]).reshape(0, 5), dtype=dtype), True, False, False, 0),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype), True, False, False, 0),
        # continuous
        (ms.Tensor(np.ones((10, 6)), dtype=dtype), True, True, True, 0),
        (ms.Tensor(np.full((10, 4), np.finfo(np.float32).max), dtype=dtype), True, True, False, 0),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).min), dtype=dtype), True, True, False, 0),
    )

    for inputs, sorted, return_inverse, return_counts, dims in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(sorted, return_inverse, return_counts, dims),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_unique(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.unique.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, None, None, None), dtype=dtype),
                op_args=(True, False, False, 0),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8, 2, 5)),
                    op_args=(True, False, False, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 6, 9)),
                    op_args=(True, False, False, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(True, False, False, 2),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 3, 10, 2, 3, 6, 7)),
                    op_args=(True, False, False, 5),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 10)),
                    op_args=(True, False, False, 2),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(True, False, False, 0),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 8)),
                    op_args=(True, False, False, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2,)),
                    op_args=(True, False, False, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def error_inputs_other_op_mint_unique_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    base_params = {
        'sorted': True,
        'return_inverse': False,
        'return_counts': False,
        'dim': 0}
    base_input=make_tensor(shape=(2,), dtype=ms.float32)
    error_configs={
        'op_input': 1,
        'sorted': "true",
        'return_inverse': "false",
        'return_counts': "false",
        'dim': '0'
    }

    for param_name, error_value in error_configs.items():
        test_params = base_params.copy()
        if param_name == "op_input":
            op_input = error_value
        else:
            op_input = base_input
            test_params[param_name] = error_value
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(test_params['sorted'], test_params['return_inverse'], test_params['return_counts'],
                         test_params['dim']),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.unique error values',
        )

def basic_sample_inputs_mint_clamp(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.clamp op.
    Follow torch's sample_inputs_clamp: use (S, M, S) and five representative cases.
    '''
    S, M = SMALL_DIM_SIZE, MEDIUM_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=None, high=None)

    shape = (S, M, S)
    # case 1: min and max are tensors with same shape as input
    yield OpSampleInput(
        op_input=make_arg(shape),
        op_args=(make_arg(shape), make_arg(shape)),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # case 2: min and max broadcastable (drop first dim)
    yield OpSampleInput(
        op_input=make_arg(shape),
        op_args=(make_arg(shape[1:]), make_arg(shape[1:])),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # case 3: only min is tensor with broadcastable shape
    yield OpSampleInput(
        op_input=make_arg(shape),
        op_args=(make_arg((S, 1, S)),),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # case 4: min is None, max is tensor
    yield OpSampleInput(
        op_input=make_arg(shape),
        op_args=(None, make_arg(shape)),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # case 5: min is tensor, max is None
    yield OpSampleInput(
        op_input=make_arg(shape),
        op_args=(make_arg(shape), None),
        op_kwargs={},
        sample_name=op_info.name,
    )
    # case 6: 0D
    # case 7: 8D
    cases = [
        (ms.Tensor(np.random.randn(), ms.float32), ms.Tensor(np.random.randn(), ms.float32),
         ms.Tensor(np.random.randn(), ms.float32)),
        (ms.Tensor(np.random.randn(7, 7, 5, 5, 3, 9, 9, 2), ms.float64), 0.4, 6.4),
    ]
    for input_x, input_min, input_max in cases:
        yield OpSampleInput(
            op_input=input_x,
            op_args=(input_min, input_max),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_clamp_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        ("1", 0, 1),
        (make_tensor(shape=(2,), dtype=ms.float32), None, None),
        (make_tensor(shape=(2,), dtype=ms.float32), "None", None),
        (make_tensor(shape=(2,), dtype=ms.float32), 0, "None")]
    for op_input, min_val, max_val in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(min_val, max_val),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.clamp error values'
        )

def extra_sample_inputs_mint_clamp(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.clamp
    Reference torch's module tests for clamp.
    '''
    cases = (
        (ms.Tensor(np.full((3, 8, 4, 5, 3), np.nan), dtype=dtype),
         ms.Tensor(np.full((3, 8, 4, 5, 3), np.nan), dtype=dtype),
         ms.Tensor(np.full((3, 8, 4, 5, 3), np.nan), dtype=dtype)),
        (ms.Tensor(np.full((4, 8, 5, 7), np.inf), dtype=dtype),
         ms.Tensor(np.full((4, 8, 5, 7), np.inf), dtype=dtype),
         ms.Tensor(np.full((4, 8, 5, 7), np.inf), dtype=dtype)),
        (ms.Tensor(np.random.randint(-128, 127, (7, 3, 9, 3, 9, 5, 7)), dtype=dtype), -2.2, 4.5),
        (ms.Tensor(np.random.randint(-32768, 32767, (3, 4, 5, 7)), dtype=dtype), 0, 20),
        (ms.Tensor(np.array([]).reshape(3,0), dtype=dtype), 0.0, 1.0),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype), -5.0, 15.0),
        # continuous
        (ms.Tensor(np.ones((10, 6)), dtype=dtype),0.5, 1.5),
        (ms.Tensor(np.full((10, 4), np.finfo(np.float32).max), dtype=ms.float32), np.finfo(np.float32).max/2,
         np.finfo(np.float32).max),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).min), dtype=ms.float32), np.finfo(np.float32).min / 2,
         np.finfo(np.float32).min),
    )

    for inputs, min_values, max_values in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(min_values, max_values),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_clamp(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.clamp.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 8, None, None), dtype=dtype),
                op_kwargs={'min': ms.Tensor(shape=(None, 8, None, None), dtype=dtype),
                           'max': ms.Tensor(shape=(None, 8, None, None), dtype=dtype)},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 8, 4, 5)),
                    op_kwargs={'min': make_func(shape=(1, 8, 4, 5)),
                               'max': make_func(shape=(1, 8, 4, 5))},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 5, 7)),
                    op_kwargs={'min': make_func(shape=(4, 8, 5, 7)),
                               'max': make_func(shape=(4, 8, 5, 7))},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_kwargs={'min': ms.Tensor(shape=None, dtype=dtype),
                           'max': ms.Tensor(shape=None, dtype=dtype)},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 8, 4, 5)),
                    op_kwargs={'min': make_func(shape=(1, 8, 4, 5)),
                               'max': make_func(shape=(1, 8, 4, 5))},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 5, 7)),
                    op_kwargs={'min': make_func(shape=(4, 8, 5, 7)),
                               'max': make_func(shape=(4, 8, 5, 7))},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_kwargs={'min': ms.Tensor(shape=None, dtype=dtype),
                           'max': ms.Tensor(shape=None, dtype=dtype)},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(7, 7, 5, 5)),
                    op_kwargs={'min': make_func(shape=(7, 7, 5, 5)),
                               'max': make_func(shape=(7, 7, 5, 5))},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 4, 9, 8)),
                    op_kwargs={'min': make_func(shape=(1, 4, 9, 8)),
                               'max': make_func(shape=(1, 4, 9, 8))},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def basic_sample_inputs_mint_stack(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.stack op.
    Follow torch cases: different shapes and number of tensors, multiple dims.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # shape x number of tensors (kept small)
    cases = (
        ((3, 4), 1),
        ((1, 2, 1, 4), 3),
        ((0, 1, 0), 2),
    )

    for shape, num_tensors in cases:
        tensors = [make_arg(shape) for _ in range(num_tensors)]
        # dim range similar to torch sample: from -1 to len(shape)-1
        for dim in range(-1, len(shape)):
            yield OpSampleInput(
                op_input=tensors,
                op_args=(dim,),
                op_kwargs={},
                sample_name=op_info.name,
            )

def error_inputs_other_op_mint_stack_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic error inputs for mint.stack op.
    Follow torch cases: error values tensors, dims.
    '''
    case = (
        ("1", -1),
        ([ms.Tensor(np.array([[0, 1], [2, 1]]), dtype=ms.float32),
          ms.Tensor(np.array([[0, 1], [2, 1]]), dtype=ms.float32)], 3),
        ([ms.Tensor(np.array([[0, 1], [2, 1]]), dtype=ms.float32),
          ms.Tensor(np.array([[0, 1], [2, 1]]), dtype=ms.float32)], 10)
    )
    for op_inputs, dims in case:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(dims,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info="other ops mint.stack/cat error values"
        )

def extra_sample_inputs_mint_stack(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.stack op where input tensor dtypes differ.
    This tests mixed dtypes inside the input sequence.
    '''
    S = SMALL_DIM_SIZE
    shape = (S,)
    # Build a list of tensors with different dtypes while keeping same shape
    # Choose a small, representative set of dtypes
    mixed_dtypes = [ms.float32, ms.int64] if dtype != ms.float32 else [ms.float16, ms.int64]
    tensors = [make_tensor(shape=shape, dtype=dt, device=device) for dt in mixed_dtypes]

    case = (
        (tensors, -1),
        ([ms.Tensor([[1, 2, 3]], dtype=dtype), ms.Tensor([[4, 5, 6]], dtype=dtype)], 1),
        ([ms.Tensor([[1, 2], [3, 4]], dtype=dtype), ms.Tensor([[5, 6], [7, 8]], dtype=dtype)], 0),
        ([ms.Tensor(np.random.randint(0, 255, size=(7, 17, 1, 5, 2, 4, 5)), dtype=dtype) for _ in range(2)], 5),
        ([ms.Tensor([float('nan'), float('inf'), float('-inf')], dtype=dtype), ms.Tensor([0.0, 1.0, -1.0],
                                                                                         dtype=dtype)], 1),
        ([ms.Tensor(np.random.randn(4096, 64), dtype=dtype) for _ in range(2)], 1),
        ([ms.Tensor(np.array([]).reshape(0, 5), dtype=dtype),
          ms.Tensor(np.array([]).reshape(0, 5), dtype=dtype)], 0),
        # discontinuous
        ([ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype),
          ms.Tensor(np.arange(24, 48).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype)], 0),
        # continuous
        ([ms.Tensor(np.arange(30).reshape(5, 6), dtype=dtype) for _ in range(2)], -1),
        ([ms.Tensor(np.full((5, 3), np.finfo(np.float32).max), dtype=dtype) for _ in range(2)], 0),
        ([ms.Tensor(np.full((4, 2), np.finfo(np.float32).min), dtype=dtype) for _ in range(2)], 1),
    )
    # Use a valid dim for 1D inputs: dim can be 0 or -1
    for inputs, dims in case:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(dims,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def mint_stack_func_with_flatten_inputs(*args):
    """A kwargs-free wrapper for mint.stack used by ms.grad in tests.

    Note:
        OpSampleInput.convert_to_args() expands op_input lists/tuples into multiple
        positional arguments. For mint.stack, the first argument should be a single
        sequence of tensors, so this wrapper repacks positional tensor args into
        a list and treats an optional trailing int as dim.
    """
    if not args:
        raise ValueError("mint_stack_func_with_flatten_inputs expects at least one argument.")
    dim = 0
    tensors = args
    # In our tests, dim is passed as a Python int when present.
    if isinstance(args[-1], int):
        dim = args[-1]
        tensors = args[:-1]
    return mint.stack(list(tensors), dim=dim)


def extra_sample_inputs_mint_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.gather op.
    Requirements:
    - cover 0D and 3D..8D (1D/2D already included in basic).
    - include an empty index case.
    Keep shapes small for resource efficiency.
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)
    make_index = functools.partial(make_tensor, device=device, low=0, dtype=ms.int64)

    # 0D scalar input, dim=0, index: scalar 0
    x_shape = ()
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(), high=1)),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Empty index tensor case (1D input). Although 1D was in basic, this is a distinct edge case.
    x_shape = (S,)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(0,), high=1, dtype=ms.int32)),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 3D: gather along middle dim (dim=1)
    x_shape = (2, 3, 4)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(1, make_index(shape=(2, 2, 4), high=x_shape[1])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 4D: negative dim (-1)
    x_shape = (2, 2, 3, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(-1, make_index(shape=(2, 2, 3, 1), high=x_shape[-1])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 5D: dim=3, non-dim axes of index <= input
    x_shape = (2, 2, 2, 3, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(3, make_index(shape=(2, 2, 2, 2, 2), high=x_shape[3])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 6D: dim=0
    x_shape = (3, 2, 2, 2, 2, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(2, 2, 2, 2, 2, 2), high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 7D: last dim
    x_shape = (2, 2, 2, 2, 2, 2, 3)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(-1, make_index(shape=(2, 2, 2, 2, 2, 2, 2), high=x_shape[-1])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 8D: dim=5
    x_shape = (2, 2, 2, 2, 2, 4, 2, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(5, make_index(shape=(2, 2, 2, 2, 2, 2, 2, 2), high=x_shape[5])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # empty tensor
    x_shape = (2, 0)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=x_shape, high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=1)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values), 1)
        index = make_arg(shape=(1, 2, 1), dtype=ms.int64)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(0, index),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_small_value',
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values), 1)
        index = make_arg(shape=(1, len(_large_values), 1), dtype=ms.int64)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1, index),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_large_value',
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((7, 8), t)
                index = ms.Tensor(np.random.randint(0, 8, (7, 8)), dtype=ms.int64)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(-1, index),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_x = make_arg(shape=(2, 2), discontiguous=discontiguous)
        index = make_arg(shape=(2, 2), dtype=ms.int64, discontiguous=discontiguous)
        yield OpSampleInput(
            op_input=input_x,
            op_args=(0, index),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
        )


def dynamic_sample_inputs_mint_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic sample inputs for mint.gather op.
    Consider both dynamic_shape and dynamic_rank.
    Notes per infer logic (gather_d.cc):
      - dim should be a scalar constant.
      - index.rank must equal input.rank.
      - For non-dim axes, dynamic shapes lead to retry; so keep non-dim axes static at compile time.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input, dim=1 static length, axis 0 static at compile time
        dim = 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),  # non-dim axis (0) static
                op_args=(dim, ms.Tensor(shape=(5, None), dtype=ms.int64)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(dim, make_tensor(shape=(5, 3), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8)),
                    op_args=(dim, make_tensor(shape=(5, 4), dtype=ms.int64, device=device, low=0, high=8)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Dynamic shape case 2: 3D input, dim=0 (static), other axes dynamic only on input
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim axis static
                op_args=(dim, ms.Tensor(shape=(None, 2, 2), dtype=ms.int64)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(dim, make_tensor(shape=(3, 2, 2), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(dim, make_tensor(shape=(4, 2, 2), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(dim, ms.Tensor(shape=None, dtype=ms.int64)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(dim, make_tensor(shape=(2,), dtype=ms.int64, device=device, low=0, high=3)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(dim, make_tensor(shape=(2, 3), dtype=ms.int64, device=device, low=0, high=2)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_inputs_other_op_mint_gather_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, dim,index(tensor)
        ("1", (1, make_tensor(shape=(1,), dtype=ms.int32))),
        (make_tensor(shape=(2,), dtype=ms.float32), ("1", make_tensor(shape=(1,), dtype=ms.int32))),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "1")),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, make_tensor(shape=(1,), dtype=ms.float32))),
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.gather error values'
        )

# sample inputs functions for mint.nn.functional.interpolate
def _normalize_mode_and_ranks(mode: str):
    # Map the op_db mode to runtime interpolate mode and supported ranks
    if mode in ("nearest1d", "nearest2d", "nearest3d"):
        internal_mode = "nearest"
        ranks = {"nearest1d": [1], "nearest2d": [2], "nearest3d": [3]}[mode]
    else:
        internal_mode = mode
        ranks = {
            "nearest": [1, 2, 3],
            "linear": [1],
            "bilinear": [2],
            "bicubic": [2],
            "trilinear": [3],
        }[mode]
    return internal_mode, ranks


def basic_sample_inputs_mint_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.nn.functional.interpolate.
    Reference torch's sample_inputs_interpolate:
      - cover size and scale_factor usages
      - align_corners for linear-family: True/False/None; for nearest: None only
    '''
    mode = kwargs.get("mode")
    internal_mode, ranks = _normalize_mode_and_ranks(mode)

    if internal_mode in ("linear", "bilinear", "bicubic", "trilinear"):
        align_corners_options = (True, False, None)
    else:
        align_corners_options = (None,)

    N, C = 1, 1
    D = 2
    S_small = 2
    S_large = 3

    def shape_with_nc(side: int, rank: int):
        return tuple([N, C] + [side] * rank)

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    for align_corners in align_corners_options:
        for rank in ranks:
            # Using size
            size_small = tuple([S_small] * rank)
            size_large = tuple([S_large] * rank)
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": size_small,
                    "scale_factor": None,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": None,
                },
                sample_name=op_info.name,
            )
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": size_large,
                    "scale_factor": None,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": None,
                },
                sample_name=op_info.name,
            )

            # Using scale_factor and varying recompute_scale_factor
            for recompute in (False, True):
                for scale in (1.7, 0.6):
                    yield OpSampleInput(
                        op_input=make_arg(shape_with_nc(D, rank)),
                        op_args=(),
                        op_kwargs={
                            "size": None,
                            "scale_factor": scale,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": recompute,
                        },
                        sample_name=op_info.name,
                    )

def extra_sample_inputs_mint_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):

    mode = kwargs.get("mode")
    internal_mode, ranks = _normalize_mode_and_ranks(mode)
    ranks = ranks[0]

    if internal_mode in ("linear", "bilinear", "bicubic", "trilinear"):
        align_corners_options = (True, False, None)
    else:
        align_corners_options = (None,)

    def _extra_sample_inputs_mint_interpolate(input_values):
        if isinstance(input_values, np.ndarray):
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input=input_values[0]
        for align_corners in align_corners_options:
            yield OpSampleInput(
                op_input=op_input,
                op_kwargs={"size": (2,) * ranks,
                           "scale_factor": None,
                           "mode": internal_mode,
                           "align_corners": align_corners,
                           "recompute_scale_factor": None, },
                sample_name=op_info.name,
            )
            for recompute in (False, True):
                yield OpSampleInput(
                    op_input=op_input,
                    op_kwargs={"size": None,
                               "scale_factor": (1.7,) * ranks,
                               "mode": internal_mode,
                               "align_corners": align_corners,
                               "recompute_scale_factor": recompute, },
                    sample_name=op_info.name,
                )

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if mode not in ['nearest3d','nearest2d','nearest1d']:
            _float_small_values = tuple(d for d in dtypes_extra_values['_float_small_values'] if d not in(-math.pi,
                math.pi,math.pi / 2,-math.pi / 2) )
            input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values), *((1,)*(ranks-1)))
            yield from  _extra_sample_inputs_mint_interpolate(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values), *((1,)*(ranks-1)))
        if not (dtype == ms.float16 and mode in ['bilinear','trilinear','linear']):
            yield from _extra_sample_inputs_mint_interpolate(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if mode not in ['bilinear','trilinear','linear']:
        if dtype is not None and dtype.is_floating_point:
            if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
                for t in [np.inf, -np.inf, np.nan]:
                    input_values = np.full((1, 1, *((3,)*ranks)), t)
                    yield from _extra_sample_inputs_mint_interpolate(input_values)

    # ---- discontiguous tensors ----
    if mode not in ['nearest1d','nearest2d']:
        for discontiguous in (True,):
            input_shape = (1, 3, *((8,) * ranks))
            input_values = make_arg(input_shape, discontiguous=discontiguous)
            yield from _extra_sample_inputs_mint_interpolate(input_values)



def extra_sample_inputs_mint_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):

    def _extra_sample_inputs_mint_linear(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        yield OpSampleInput(
            op_input=op_input,
            op_args=(make_arg((op_input.shape[-1],)),),
            op_kwargs={},
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            op_input=op_input,
            op_args=(make_arg((2, op_input.shape[-1],)),),
            op_kwargs={'bias': make_arg((2,))},
            sample_name=op_info.name,
        )
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values))
        yield from  _extra_sample_inputs_mint_linear(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values))
        yield from _extra_sample_inputs_mint_linear(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 2), t)
                yield from _extra_sample_inputs_mint_linear(input_values)

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_values = make_arg((1,3), discontiguous=discontiguous)
        yield from _extra_sample_inputs_mint_linear(input_values)

def extra_sample_inputs_mint_batch_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-9, high=0)

    def _extra_sample_inputs_mint_batch_norm(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        yield OpSampleInput(
            op_input=op_input,
            op_args=(make_arg((op_input.shape[1],)),make_arg((op_input.shape[1],)),make_arg((op_input.shape[1],)),
                     make_arg((op_input.shape[1],))),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values))
        yield from  _extra_sample_inputs_mint_batch_norm(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        yield from _extra_sample_inputs_mint_batch_norm(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 2), t)
                yield from _extra_sample_inputs_mint_batch_norm(input_values)

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_values = make_arg((1,2), discontiguous=discontiguous)
        yield from _extra_sample_inputs_mint_batch_norm(input_values)

def basic_sample_inputs_reduction_count_nonzero(op_info, dtype, device=None, **kwargs):
    """count_nonzero does not have keepdim parameter"""
    for sample_input in basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs):
        sample_input.op_kwargs.pop('keepdim', None)
        yield sample_input


def extra_sample_inputs_mint_layer_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def _extra_sample_inputs_mint_layer_norm(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        yield OpSampleInput(
            op_input=op_input,
            op_args=(op_input.shape[1:],),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values))
        yield from  _extra_sample_inputs_mint_layer_norm(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        yield from _extra_sample_inputs_mint_layer_norm(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 2), t)
                yield from _extra_sample_inputs_mint_layer_norm(input_values)

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_values = make_arg((1,2), discontiguous=discontiguous)
        yield from _extra_sample_inputs_mint_layer_norm(input_values)

def extra_sample_inputs_mint_log_softmax(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def _extra_sample_inputs_mint_log_softmax(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        yield OpSampleInput(
            op_input=op_input,
            op_args=(),
            op_kwargs={'dim': 1},
            sample_name=op_info.name,
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values))
        yield from  _extra_sample_inputs_mint_log_softmax(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        yield from _extra_sample_inputs_mint_log_softmax(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 2), t)
                yield from _extra_sample_inputs_mint_log_softmax(input_values)

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_values = make_arg((1, 2), discontiguous=discontiguous)
        yield from _extra_sample_inputs_mint_log_softmax(input_values)

def extra_sample_inputs_mint_AdaptiveAvgPool(op_info: OpInfo, dtype=None, device=None, **kwargs):

    pool_dim = kwargs.get('pool_dim')

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    rank = {'Pool1d': 1, 'Pool2d': 2, 'Pool3d': 3,
            'functional_Pool1d': 1, 'functional_Pool2d': 2, 'functional_Pool3d': 3}[pool_dim]

    def _extra_sample_inputs_mint_AdaptiveAvgPool(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        yield OpSampleInput(
            op_input=op_input,
            op_args=(tuple(np.random.randint(2,9,size=rank)),),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, *((1,)*(rank-1)), len(_float_small_values))
        yield from  _extra_sample_inputs_mint_AdaptiveAvgPool(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, *((1,)*(rank-1)), len(_large_values))
        yield from _extra_sample_inputs_mint_AdaptiveAvgPool(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, *(2,)*rank), t)
                yield from _extra_sample_inputs_mint_AdaptiveAvgPool(input_values)

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_values = make_arg((1, 1, *(2,)*rank), discontiguous=discontiguous)
        yield from _extra_sample_inputs_mint_AdaptiveAvgPool(input_values)

def extra_sample_inputs_mint_GroupNorm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def _extra_sample_inputs_mint_GroupNorm(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        yield OpSampleInput(
            op_input=op_input,
            op_args=(1, 1, ),
            op_kwargs={},
            sample_name=op_info.name,
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values))
        yield from  _extra_sample_inputs_mint_GroupNorm(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values))
        yield from _extra_sample_inputs_mint_GroupNorm(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 2), t)
                yield from _extra_sample_inputs_mint_GroupNorm(input_values)

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_values = make_arg((1, 1, 2), discontiguous=discontiguous)
        yield from _extra_sample_inputs_mint_GroupNorm(input_values)

def extra_sample_inputs_mint_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    def _extra_sample_inputs_mint_dropout(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device,)
        else:
            op_input = input_values
        for p in [0.0, 1.0]:
            yield OpSampleInput(
                op_input=op_input,
                op_args=(),
                op_kwargs={'p': p, 'training': True},
                sample_name=op_info.name,
            )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1,  len(_float_small_values))
        yield from  _extra_sample_inputs_mint_dropout(input_values)

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        yield from _extra_sample_inputs_mint_dropout(input_values)

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((2, 4), t)
                yield from _extra_sample_inputs_mint_dropout(input_values)

    # case fail at kbk mode while tensor is discontiguous skip...


def extra_sample_inputs_mint_pad(op_info: OpInfo, dtype=None, device=None, **kwargs):

    mode = kwargs.get('mode')

    input_shape_length_convert_to_pad_shape_length = {2: [1], 3: [1, 2], 4: [2, 3], 5: [3]}

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    value = 2. if mode == 'constant' else None

    def _extra_sample_inputs_mint_pad(input_values):
        if isinstance(input_values, np.ndarray):
            op_input = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        else:
            op_input = input_values
        for dim in input_shape_length_convert_to_pad_shape_length[op_input.dim()]:
            yield OpSampleInput(
                op_input=op_input,
                op_args=((1,)*dim*2, ),
                op_kwargs={'mode': mode, 'value': value},
                sample_name=op_info.name,
            )
    for dim in [2, 3, 4, 5]:
        # ---- small value tensors ----
        if dtype is not None and dtype.is_floating_point:
            _float_small_values = dtypes_extra_values['_float_small_values']+(0.0, -0.0,)

            dim_to_shape={2: (18,), 3: (2,9), 4: (3,3,2), 5:(3,3,2)}

            if dim != 5:
                input_values = np.array(_float_small_values).reshape(1, *dim_to_shape[dim]  )
            else:
                input_values = np.array(_float_small_values).reshape(1, 1, *dim_to_shape[dim] )
            yield from  _extra_sample_inputs_mint_pad(input_values)

        # ---- large value tensors ----
        if dtype is not None and dtype.is_floating_point:
            if dtype == ms.float16:
                dim_to_shape = {2: (8,), 3: (2, 4), 4: (2, 2, 2), 5: (2, 2, 2)}
                _large_values = dtypes_extra_values['_fp16_large_values']+(-521, 521,)
            else:
                dim_to_shape = {2: (12,), 3: (2, 6), 4: (2, 3, 2), 5: (2, 3, 2)}
                _large_values = dtypes_extra_values['_fp32_large_values']+(-521, 521)
            if dim != 5:
                input_values = np.array(_large_values).reshape(1,
                            *dim_to_shape[dim])
            else:
                input_values = np.array(_large_values).reshape(1, 1,
                            *dim_to_shape[dim])
            yield from _extra_sample_inputs_mint_pad(input_values)

        # ---- extremal value tensors (inf, -inf, nan) ----
        if dtype is not None and dtype.is_floating_point:
            if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
                for t in [np.inf, -np.inf, np.nan]:
                    input_values = np.full((2,)*dim, t)
                    yield from _extra_sample_inputs_mint_pad(input_values)

        # ---- discontiguous tensors ----
        for discontiguous in (True,):
            input_values = make_arg((2,)*dim, discontiguous=discontiguous)
            yield from _extra_sample_inputs_mint_pad(input_values)

def dynamic_sample_inputs_mint_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic sample inputs for mint.nn.functional.interpolate.
    Guideline based on upsample_forward_base inference:
      - exactly one of size or scale_factor must be provided
      - dynamic shape: spatial dims can be None at compile time
      - dynamic rank: compile with shape=None and run with specific ranks
    '''
    mode = kwargs.get("mode")
    internal_mode, ranks = _normalize_mode_and_ranks(mode)

    if internal_mode in ("linear", "bilinear", "bicubic", "trilinear"):
        dyn_align_corners = (True, False)
    else:
        dyn_align_corners = (None,)

    N, C = 1, 1
    D1 = 2
    D2 = 3
    S_target = 2

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=-1, high=1)

    def nc_shape(rank: int, side: int):
        return tuple([N, C] + [side] * rank)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape with size fixed at compile time
        for rank in ranks:
            size_tuple = tuple([S_target] * rank)
            for align_corners in dyn_align_corners:
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=tuple([N, C] + [None] * rank), dtype=dtype),
                        op_args=(),
                        op_kwargs={
                            "size": mutable(size_tuple),
                            "scale_factor": None,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": None,
                        },
                        sample_name=f'{op_info.name}_dynamic_shape_compile_input_size_r{rank}',
                    ),
                    op_running_inputs=(
                        OpSampleInput(
                            op_input=make_func(shape=nc_shape(rank, D1)),
                            op_args=(),
                            op_kwargs={
                                "size": mutable(size_tuple),
                                "scale_factor": None,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": None,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_running_input_size_r{rank}',
                        ),
                        OpSampleInput(
                            op_input=make_func(shape=nc_shape(rank, D2)),
                            op_args=(),
                            op_kwargs={
                                "size": mutable(size_tuple),
                                "scale_factor": None,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": None,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_running_input_size_r{rank}',
                        ),
                    ),
                )

        # Dynamic shape with scale_factor fixed at compile time
        for rank in ranks:
            for align_corners in dyn_align_corners:
                for scale in (1.7, 0.6):
                    yield OpDynamicInput(
                        op_compile_input=OpSampleInput(
                            op_input=ms.Tensor(shape=tuple([N, C] + [None] * rank), dtype=dtype),
                            op_args=(),
                            op_kwargs={
                                "size": None,
                                "scale_factor": scale,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": None,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_compile_input_scale_r{rank}',
                        ),
                        op_running_inputs=(
                            OpSampleInput(
                                op_input=make_func(shape=nc_shape(rank, D1)),
                                op_args=(),
                                op_kwargs={
                                    "size": None,
                                    "scale_factor": scale,
                                    "mode": internal_mode,
                                    "align_corners": align_corners,
                                    "recompute_scale_factor": None,
                                },
                                sample_name=f'{op_info.name}_dynamic_shape_running_input_scale_r{rank}',
                            ),
                            OpSampleInput(
                                op_input=make_func(shape=nc_shape(rank, D2)),
                                op_args=(),
                                op_kwargs={
                                    "size": None,
                                    "scale_factor": scale,
                                    "mode": internal_mode,
                                    "align_corners": align_corners,
                                    "recompute_scale_factor": None,
                                },
                                sample_name=f'{op_info.name}_dynamic_shape_running_input_scale_r{rank}',
                            ),
                        ),
                    )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank with size specified
        for rank in ranks:
            size_tuple = tuple([S_target] * rank)
            align_corners = dyn_align_corners[0]
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(),
                    op_kwargs={
                        "size": mutable(size_tuple),
                        "scale_factor": None,
                        "mode": internal_mode,
                        "align_corners": align_corners,
                        "recompute_scale_factor": None,
                    },
                    sample_name=f'{op_info.name}_dynamic_rank_compile_input_r{rank}',
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=nc_shape(rank, D1)),
                        op_args=(),
                        op_kwargs={
                            "size": mutable(size_tuple),
                            "scale_factor": None,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": None,
                        },
                        sample_name=f'{op_info.name}_dynamic_rank_running_input_r{rank}',
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=nc_shape(rank, D2)),
                        op_args=(),
                        op_kwargs={
                            "size": mutable(size_tuple),
                            "scale_factor": None,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": None,
                        },
                        sample_name=f'{op_info.name}_dynamic_rank_running_input_r{rank}',
                    ),
                ),
            )


# basic op_basic_reference_inputs_func for prelu
def basic_sample_inputs_prelu_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield typical shape cases for unary ops.

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

    # 1D shape is not supported in grad of mint.nn.PReLU
    shapes = (
        (S,),
        (M, S),
        (S, S, L),
        (3, 0, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for input_shape in shapes:
        _input = make_func(input_shape)
        #prelu: type promoting not support,got float and half
        if _input.dtype != ms.float16:
            yield OpSampleInput(
                _input,
                op_args=(),
                op_kwargs={'num_parameters': 1, 'init': random.choice([0.25, -0.25]), 'dtype': _input.dtype},
                sample_name=op_info.name,
            )

#tensor.sum has keyword parameter dtype
def tensor_sum_ms(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.sum(dim, keepdim, dtype=dtype)

#tensor.sum has keyword parameter dtype
def tensor_sum_torch(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.sum(dim, keepdim, dtype=dtype)

#tensor.mean has keyword parameter dtype
def tensor_mean_ms(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.mean(dim, keepdim, dtype=dtype)

#tensor.mean has keyword parameter dtype
def tensor_mean_torch(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.mean(dim, keepdim, dtype=dtype)

def tensor_argmax_ms(op_input, *op_args, **op_kwargs):
    return op_input.argmax(*op_args, **op_kwargs)

def tensor_argmax_torch(op_input, *op_args, **op_kwargs):
    return op_input.argmax(*op_args, **op_kwargs)

def tensor_argmin_ms(op_input, *op_args, **op_kwargs):
    return op_input.argmin(*op_args, **op_kwargs)

def tensor_argmin_torch(op_input, *op_args, **op_kwargs):
    return op_input.argmin(*op_args, **op_kwargs)

def tensor_max_ms(op_input, *op_args, **op_kwargs):
    return op_input.max(*op_args, **op_kwargs)

def tensor_max_torch(op_input, *op_args, **op_kwargs):
    return op_input.max(*op_args, **op_kwargs)

def tensor_view_dtype_ms(op_input, dtype):
    return op_input.view(dtype)

def tensor_sign_to_sgn(op_input, *op_args, **op_kwargs):
    if isinstance(op_input, torch.Tensor) and op_input.dtype.is_complex:
        return torch.sgn(op_input, *op_args, **op_kwargs)
    return torch.sign(op_input, *op_args, **op_kwargs)

def basic_sample_inputs_binary_logical_or(op_info, dtype, device=None, **kwargs):
    """add  bool type inputs"""
    yield from basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)

    yield OpSampleInput(
        op_input=ms.Tensor([True, False, True], dtype=ms.bool_),
        op_args=(ms.Tensor([True, False, True], dtype=ms.bool_),),
        sample_name=op_info.name,
    )

    yield OpSampleInput(
        op_input=make_tensor(
            (3,),
            dtype=dtype, device=device,
        ),
        op_args=(ms.Tensor([True, False, True], dtype=ms.bool_),),
        sample_name=op_info.name,
    )

    yield OpSampleInput(
        op_input=ms.Tensor([True, False, True], dtype=ms.bool_),
        op_args=(make_tensor(
            (3,),
            dtype=dtype, device=device,
        ),),
        sample_name=op_info.name,
    )

def extra_sample_inputs_reduction_count_nonzero(op_info, dtype, device=None, **kwargs):
    """count_nonzero does not have keepdim parameter"""
    for sample_input in extra_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs):
        sample_input.op_kwargs.pop('keepdim', None)
        yield sample_input

def basic_sample_inputs_reduction_with_correction(op_info, dtype, device=None, **kwargs):
    """count_nonzero does not have keepdim parameter"""
    yield from basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs)
    for i in [-1,0,2,6]:
        yield OpSampleInput(
            op_input=make_tensor(
                (5,),
                dtype=dtype,device=device,
            ),
            op_kwargs={'dim':0, 'keepdim':True, 'correction':i},
            sample_name=f'{op_info.name}_{dtype}_correction{i}_dim1'
        )
        yield OpSampleInput(
            op_input=make_tensor(
                (5,5),
                dtype=dtype, device=device,
            ),
            op_kwargs={'dim': (0,1), 'keepdim': True, 'correction': i},
            sample_name=f'{op_info.name}_{dtype}_correction{i}_dim2'
        )

def basic_sample_inputs_mint_pow(op_info: BinaryOpInfo, dtype=None, device=None, **kwargs):
    XS = EXTRA_SMALL_DIM_SIZE
    S = EXTRA_SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE

    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )

    shapes = (
        ((), ()),
        ((S,), ()),
        ((S, 1), (S,)),
        ((M, S), ()),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, 1, S), (M, S)),
        ((M, 1, S), (1, M, S)),
        ((0, 1, XS), (0, M, XS)),
    )

    for input_shape, other_shape in shapes:
        #mint.pow supports all values in 910B, but 910A will provide incorrect values while dtype in uint8/int8
        if dtype in [ms.int8, ms.uint8] and MSContext.get_instance().get_ascend_soc_version() == "ascend910":
            _input = make_func(input_shape, low=op_info.input_low, high = op_info.input_high)
            _other = make_func(other_shape, low=op_info.other_low, high = op_info.other_high)
        else:
            _input = make_func(input_shape)
            _other = make_func(other_shape)
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_pow(op_info: BinaryOpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    shapes = (
        ((2, 2, 3, 2), (2, 2, 3, 2)),
        ((2, 2, 2, 3, 2), (2, 2, 2, 3, 2)),
        ((3, 2, 2, 2, 2, 2), (3, 2, 2, 2, 2, 2)),
        ((2, 2, 2, 2, 2, 2, 3), (2, 2, 2, 2, 2, 2, 3)),
        ((2, 2, 2, 2, 2, 4, 2, 2), (2, 2, 2, 2, 2, 4, 2, 2)),
    )
    for input_shape, other_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low, high = op_info.input_high)
        _other = make_func(other_shape, low=op_info.other_low, high = op_info.other_high)

        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            op_input=ms.Tensor(random.choice([True, False])),
            op_args=(_other,),
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            op_input=_input,
            op_args=(ms.Tensor(random.choice([True, False])),),
            sample_name=op_info.name,
        )

    if not dtype.is_complex:
        yield from _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func(
            op_info, dtype, device, **kwargs)
        yield from _generate_binary_op_scalar_inputs_func(op_info, dtype, device, **kwargs)
    if dtype.is_floating_point or dtype.is_complex:
        yield from _generate_binary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)

def dynamic_sample_inputs_mint_pow(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input, dim=1 static length, axis 0 static at compile time
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),  # non-dim axis (0) static
                op_args=(ms.Tensor(shape=(5, None), dtype=dtype),),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6), random_method='randn'),
                    op_args=(make_func(shape=(5, 6), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8), random_method='randn'),
                    op_args=(make_func(shape=(5, 8), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Dynamic shape case 2: 3D input, dim=0 (static), other axes dynamic only on input
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim axis static
                op_args=(ms.Tensor(shape=(6, None, 2), dtype=dtype),),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2), random_method='randn'),
                    op_args=(make_func(shape=(6, 3, 2), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2), random_method='randn'),
                    op_args=(make_func(shape=(6, 5, 2), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,), random_method='randn'),
                    op_args=(make_func(shape=(3,), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), random_method='randn'),
                    op_args=(make_func(shape=(2, 3), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


def basic_sample_inputs_mint_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (((S, S, S), (1, 2)),
             ((S, S, S), (-1, 2)),
             ((S, S, S), (-1, -1)),
             ((S, S, S), (1, -1)),
             ((S, S), (-1, 2)),
             ((S,), (0, 2))
             )
    for shape, (dim, index) in cases:
        yield OpSampleInput(
            op_input=make_x(shape),
            op_args=(dim, index),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((2, 3, 4), 1, -2),
        ((2, 2, 3, 2), -1, 1),
        ((2, 2, 2, 3, 2), 3, 1),
        ((3, 2, 2, 2, 2, 2), 0, 1),
        ((2, 2, 2, 2, 2, 2, 3), -1, 1),
        ((2, 2, 2, 2, 2, 4, 2, 2), 5, 3),
    )
    for shape, dim, index in cases:
        yield OpSampleInput(
            op_input=make_x(shape),
            op_args=(dim, index),
            op_kwargs={},
            sample_name=op_info.name,
        )


def dynamic_sample_inputs_mint_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input, dim=1 static length, axis 0 static at compile time
        dim = 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),  # non-dim axis (0) static
                op_args=(dim, mutable(0)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(dim, mutable(1)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8)),
                    op_args=(dim, mutable(2)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Dynamic shape case 2: 3D input, dim=0 (static), other axes dynamic only on input
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim axis static
                op_args=(dim, mutable(1)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(dim, mutable(2)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(dim, mutable(3)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(dim, mutable(2)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(dim, mutable(0)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(dim, mutable(1)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_inputs_other_op_mint_select_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, dim,index
        ("1", (1, 0)),
        (make_tensor(shape=(2,), dtype=ms.float32), ("1", 0)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "1")),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, -1))
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.select error values'
        )

def basic_sample_inputs_mint_isclose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=op_info.exclude_value)
    cases = [
        (ms.Tensor(np.random.randint(-50, 50, (3, 9, 8, 9)), dtype=dtype),
         ms.Tensor(np.random.randint(-50, 50, (3, 9, 8, 9)), dtype=dtype), 1e-05, 1e-05, False),
        (ms.Tensor(np.random.randn(), dtype=dtype), ms.Tensor(np.random.randn(), dtype=dtype), 1e-05, 1e-05, True),
        (make_arg(shape=(7, 8)), make_arg(shape=(7, 8)), False, False, False),
        (make_arg(shape=(9, 8)), make_arg(shape=(9, 8)), 1, 1, False)
    ]
    for op_inputs, other, rtol, atol, equal_nan in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(other, rtol, atol, equal_nan),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_isclose(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype),
         ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype), 1e-05, 1e-05, False),
        (ms.Tensor(np.full((4, 8), np.finfo(np.float32).min), dtype=dtype),
         ms.Tensor(np.full((4, 8), np.finfo(np.float32).min), dtype=dtype), 1e-05, 1e-05, False),
        # discontinuous
        (ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32)[:, ::2], dtype=dtype),
         ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32)[:, ::2], dtype=dtype), 0.0, 0.0, False),
        (ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), dtype=dtype),
         ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), dtype=dtype), 1.0, 0.0, False),
        (ms.Tensor(np.zeros((0, 5)), dtype=dtype), ms.Tensor(np.zeros((0, 5)), dtype=dtype), True, True, False),
        #broadcast
        (ms.Tensor(np.arange(24).reshape(2, 3, 4).astype(np.float32), dtype=dtype),
         ms.Tensor(np.array([[[1.0], [2.0], [3.0]]]), dtype=dtype), 1e-05, 1e-05, False),

    ]
    for op_inputs, other, rtol, atol, equal_nan in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(other, rtol, atol, equal_nan),
            op_kwargs={},
            sample_name=op_info.name,
        )

    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((7, 9, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(input_values, dtype=dtype, device=device), 1e-05, 1e-05, False),
                    op_kwargs={},
                    sample_name=op_info.name,
                )

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
                op_args=(make_tensor_with_np_array(np_array, dtype=dtype), 1e-05, 1e-05, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

def error_inputs_other_op_inputs_mint_isclose(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        ("0", ms.Tensor(np.random.randint(-50, 50, (3, 9, 8, 9)), dtype=ms.float32), 1e-05, 1e-05, False),
        (ms.Tensor(np.random.randn(), dtype=ms.float32), 0, 1e-05, 1e-05, True),
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32),
         ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), "invalid_rtol", 1e-05, False),
        (ms.Tensor(np.random.randn(9, 8), dtype=ms.float32), ms.Tensor(np.random.randn(9, 8), dtype=ms.float32),
         1.0, "invalid_rtol", False),
        (ms.Tensor(np.random.randn(9, 8), dtype=ms.float32),
         ms.Tensor(np.random.randn(9, 8), dtype=ms.float32), 1, True, "False")
    ]
    for op_input, other, rtol, atol, equal_nan in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(other, rtol, atol, equal_nan),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.select error values'
        )

def dynamic_sample_inputs_mint_isclose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 8, None, None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, None, None, None, None), dtype=dtype), True, False, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 8, 4, 5, 7, 5, 3)),
                    op_args=(make_func(shape=(1, 8, 4, 5, 7, 5, 3)), True, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 8, 5, 9, 4, 8, 5)),
                    op_args=(make_func(shape=(9, 1, 5, 9, 4, 8, 5)), True, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None), dtype=dtype), True, False, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 3, 3, 3, 9)),
                    op_args=(make_func(shape=(1, 3, 3, 3, 9)), True, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 8, 4, 3)),
                    op_args=(make_func(shape=(7, 8, 4, 1)), True, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None), dtype=dtype), True, False, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 3)),
                    op_args=(make_func(shape=(1, 3)), True, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9)),
                    op_args=(make_func(shape=(1, 9)), True, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def basic_sample_inputs_mint_argsort(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        (make_tensor(shape=(2, 2, 1), dtype=dtype), {'dim': -1, "descending": False, "stable": False}),
        (make_tensor(shape=(2,), dtype=dtype), {'dim': -1, "descending": True, "stable": False}),
        (make_tensor(shape=(2, 3), dtype=dtype), {'dim': 0, "descending": False, "stable": False}),
    ]
    for op_inputs, p_kwargs in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

def error_inputs_other_op_inputs_mint_argsort_func(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        (0, {'dim': -1, "descending": False, "stable": False}),
        (make_tensor(shape=(2,), dtype=ms.float32), {'dim': "-1", "descending": True, "stable": False}),
        (make_tensor(shape=(2, 3), dtype=ms.float32), {'dim': 0, "descending": "False", "stable": False}),
        (make_tensor(shape=(2, 3), dtype=ms.float32), {'dim': 0, "descending": False, "stable": "False"}),
    ]
    for op_input, p_kwargs in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(),
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.select error values'
        )

def dynamic_sample_inputs_mint_argsort(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def extra_sample_inputs_mint_argsort(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        (make_tensor(shape=(3, 2, 2, 2, 2, 2), dtype=ms.float32), {'dim': -1, "descending": False, "stable": False}),
        (make_tensor(shape=(2, 3), dtype=ms.float32), {'dim': 0, "descending": False, "stable": False}),
        #broadcast
        (make_tensor(shape=(1, 4, 3), dtype=ms.float32), {'dim': 1, "descending": True, "stable": True}),
    ]
    for op_inputs, p_kwargs in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )

    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(),
            op_kwargs={'dim': 0, "descending": False, "stable": False},
            sample_name=op_info.name,
        )

    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((7, 9, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(),
                    op_kwargs={'dim': 0, "descending": False, "stable": False},
                    sample_name=op_info.name,
                )

def basic_sample_inputs_mint_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    def make_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high)

    shapes = ((), (S,), (L, M, S))
    num_classess = (-1, 10)

    for shape, num_classes in itertools.product(shapes, num_classess):
        yield OpSampleInput(
            op_input=make_input(
                shape,
                low=0,
                high=10 if num_classes == -1 else num_classes // 2,
            ),
            op_args=(),
            op_kwargs={"num_classes": num_classes},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    def make_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high)

    shapes = (
        ((2, 2, 3, 2), -1),
        ((2, 2, 2, 3, 2), 10),
        ((3, 2, 2, 2, 2, 2), -1),
        ((2, 2, 2, 2, 2, 2, 3), 10),
    )
    for shape, num_classes in shapes:
        yield OpSampleInput(
            op_input=make_input(
                shape,
                low=0,
                high=10 if num_classes == -1 else num_classes // 2,
            ),
            op_args=(),
            op_kwargs={"num_classes": num_classes},
            sample_name=op_info.name,
        )

    # ---- discontiguous tensors ----
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=1)
    for discontiguous in (True,):
        input_shape = (5, 50)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(),
            op_kwargs={"num_classes": 62},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
        )


def basic_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    shapes = ((M, S, L), (M, S), (S,), (),)
    make_tensor_partial = functools.partial(make_tensor, dtype=dtype, device=device)
    for shape in shapes:
        yield OpSampleInput(make_tensor_partial(shape))
        if len(shape) > 1:
            yield OpSampleInput(
                op_input=make_tensor_partial(shape),
                op_args=(),
                op_kwargs={"start_dim": 1, "end_dim": -1},
                sample_name=op_info.name,
            )
            yield OpSampleInput(
                op_input=make_tensor_partial(shape),
                op_args=(),
                op_kwargs={"start_dim": 0, "end_dim": len(shape)-1},
                sample_name=op_info.name,
            )
        else:
            yield OpSampleInput(
                op_input=make_tensor_partial(shape),
                op_args=(),
                op_kwargs={"start_dim": -1, "end_dim": -1},
                sample_name=op_info.name,
            )


def extra_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # shape x start_dim x end_dim
    cases = (
        ((5, 4, 0, 1, 3, 7), 1, 3),
        ((5, 4, 0, 1, 3, 7), 4, 5),
        ((5, 4, 1, 1, 3, 7), 2, 3),
        ((), 0, 0),
        ((1,), 0, -1),
        ((3, 7, 5), 1, 2),
        ((4, 5), 1, 1),
        ((1, 5, 5, 1, 5, 1, 5, 1), 0, 2),
        ((1, 5, 5, 1, 5, 1, 5, 1), 3, -1),
        ((1, 5, 5, 1, 5, 1, 5, 1), -8, -1),
        ((1, 5, 5, 1, 5, 7, 5, 1), -2, -1),
        ((2, 4, 2), 0, 1),
        ((4, 2, 2), 1, 2),
        ((0, 3, 4, 5), 1, 3),
    )

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    for shape, start, end in cases:
        yield OpSampleInput(
            make_arg(shape),
            op_args=(start, end,),
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            make_arg(shape, discontiguous=True).transpose(0, -1),
            op_args=(start, end,),
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            make_arg(shape).transpose(0, -1),
            op_args=(start, end,),
            sample_name=op_info.name
        )

def error_inputs_other_op_mint_flatten_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, start_dim, end_dim,
        ("1", (0, -1)),
        (make_tensor(shape=(2,), dtype=ms.float32), ("0", -1)),
        (make_tensor(shape=(2,), dtype=ms.float32), (0, "1")),
        (make_tensor(shape=(2,), dtype=ms.float32), (5, -1))]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.flatten error values'
        )

def dynamic_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.flatten.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 7, None, None), dtype=dtype),
                op_args=(0, -1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 7, 5, 9)),
                    op_args=(0, -1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 7, 8, 10)),
                    op_args=(0, -1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0, -1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 4, 8, 4)),
                    op_args=(0, -1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 4, 9, 7, 4, 4, 9)),
                    op_args=(0, -1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0, -1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(7, 5, 7)),
                    op_args=(0, -1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 9)),
                    op_args=(0, -1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def basic_sample_inputs_mint_reshape(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    cases = (
        ((S, S, S), (S * S, S)),
        ((S * S, S), (S, S, S)),
        ((S * S, S), (S, -1, S)),  # neg index
        ((S * S * 2, S), (S, -1)),  # neg index
        ((S,), (S,)),
        ((), ()),  # empty
        ((), (1,)),
    )

    for a, b in cases:
        yield OpSampleInput(
            op_input=make_arg(a),
            op_args=(b,),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_reshape(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (
        ((125,), (25, 5)),
        ((25, 25), (1, 5, 5, 1, 5, 1, 5, 1)),
        ((16, 32), (2, 4, 1, 4, 4, 1, 4)),
        ((16, 12), (12, 16)),
        ((1, 16, 12), (12, 16)),
        ((1, 5, 1, 5), (25, 1)),
        ((2, 4, 2), (4, 4)),
        ((1, 4), (1, 1, 2, 1, 2)),
        ((3, 5, 7), (7, 5, 3)),
        ((1,), ()),  # empty
        ((5, 0, 2, 3), (5, 0, 2, 3)),
        ((2, 1, 0, 3, 1), (5, 0)),
        ((1,), ()),  # empty
        ((4, 5, 6), (4, 5, 6, 1, 1, 1)),
        ((), (1, 1, 1, 1)),  # empty
    )

    irreversible_cases = (
        ((), (-1,)),  # neg index, empty
        ((4, 7, 9, 1, 1), (1, 4, 3, -1, 1)),  # neg index
    )

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    for a, b in cases:
        yield OpSampleInput(
            op_input=make_arg(a),
            op_args=(b,),
            op_kwargs={},
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            op_input=make_arg(b),
            op_args=(a,),
            op_kwargs={},
            sample_name=op_info.name,
        )

    for a, b in irreversible_cases:
        yield OpSampleInput(
            op_input=make_arg(a),
            op_args=(b,),
            op_kwargs={},
            sample_name=op_info.name,
        )
def dynamic_sample_inputs_mint_reshape(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
            op_args=(mutable(input_data=(2, 3), dynamic_len=True),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(3, 3, 3, 9), dtype=dtype),
                op_args=(mutable(input_data=(3, 3, 1, -1, 1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(7, 9, 5, 7), dtype=dtype),
                op_args=(mutable(input_data=(5, 3, 3, 7, 7), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(5, 5, 2, 4), dtype=dtype),
                op_args=(mutable(input_data=(8, 1, 1, -1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )
    # dynamic rank
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=dtype),
            op_args=(mutable(input_data=(2, 6), dynamic_len=True),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_rank_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(7, 3)),
                op_args=(mutable(input_data=(21, ), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(9, 9, 3, 5, 8)),
                op_args=(mutable(input_data=(2, 2, 3, 5, 9, -1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(4,)),
                op_args=(mutable(input_data=(2, 2), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
        ),
    )

def error_sample_inputs_mint_reshape(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.Tensor(np.random.randn(2,3,4)),
            op_args=((5,6),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='proposed_shape is invalid for input size'
    )

def basic_sample_inputs_mint_pad(op_info: OpInfo, dtype=None, device=None, mode=None, **kwargs):
    assert mode in ('constant', 'reflect', 'replicate', 'circular')
    if mode in ['reflect', 'replicate']:
        cases: tuple = (  # ignore
            ((1, 3), (1, 2)),
            ((1, 3), (0, 1)),
            ((0, 3, 3), (1, 2)),
            ((0, 3, 3), (0, 1)),
            ((1, 3, 3), (1, 2)),
            ((1, 3, 3), (0, 1)),
            ((1, 3, 3), (0, 2, 0, 1)),
            ((0, 3, 3, 3), (0, 2, 0, 1)),
            ((3, 3, 5, 5), (0, 2, 0, 1)),
            ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
            # ((1, 3, 4, 4), (-1, 1, -2, 1)),
        )
    elif mode == 'constant':
        cases = (
            ((1, 3), (1, 2)),
            ((1, 3), (0, 1)),
            ((1, 3), (0, 2, 0, 1)),
            ((5, 4), (-1, -2, 1, 1)),
            ((0, 3, 3), (1, 2)),
            ((0, 3, 3), (0, 1)),
            ((0, 3, 3), (0, 2, 0, 1)),
            ((0, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 3), (1, 2)),
            ((1, 3, 3), (0, 1)),
            ((1, 3, 3), (0, 2, 0, 1)),
            ((1, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((0, 3, 3, 3), (1, 2)),
            ((0, 3, 3, 3), (0, 1)),
            ((0, 3, 3, 3), (0, 2, 0, 1)),
            ((0, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((3, 3, 5, 5), (1, 2)),
            ((3, 3, 5, 5), (0, 1)),
            ((3, 3, 5, 5), (0, 2, 0, 1)),
            ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 3, 3, 3), (1, 2)),
            ((1, 3, 3, 3, 3), (0, 1)),
            ((1, 3, 3, 3, 3), (0, 2, 0, 1)),
            ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 4, 4), (-1, 1, -2, 1)),
        )
    else:  # mode == 'circular'
        if dtype == ms.bool_:
            cases = (
                ((2, 3, 3), (1, 2)),
                ((1, 3, 3), (1, 2)),
            )
        else:
            cases = (
                ((1, 3, 3), (1, 2)),
                ((1, 3, 3), (0, 1)),
                ((1, 3, 3), (1, 2)),
                ((1, 3, 3), (0, 1)),
                ((1, 3, 3, 3), (0, 2, 0, 1)),
                ((3, 3, 5, 5), (0, 2, 0, 1)),
                ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
                # ((1, 3, 4, 4), (-1, 1, -2, 1)),
            )

    make_inp = functools.partial(make_tensor, device=device, dtype=dtype)

    if mode == 'constant':
        # Default args
        yield OpSampleInput(
            op_input=make_inp((1, 3, 3)),
            op_args=((2, 2),),
            op_kwargs={},
            sample_name=op_info.name,
        )

    if mode in ['reflect', 'replicate', 'circular']:
        for shape, pad in cases:
            yield OpSampleInput(
                op_input=make_inp(shape),
                op_args=(pad, mode),
                op_kwargs={},
                sample_name=op_info.name,
            )
    else:  # mode == 'constant'
        for pad_value in (1., 2.):
            for shape, pad in cases:
                yield OpSampleInput(
                    op_input=make_inp(shape),
                    op_args=(pad, mode, pad_value),
                    op_kwargs={},
                    sample_name=op_info.name,
                )


def gather_variable(shape, index_dim, max_indices, duplicate=False):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = mint.zeros(shape, dtype=ms.int64)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            mint.randperm(max_indices)[:shape[batch_dim]])
    if duplicate:
        index.select(batch_dim, 0).copy_(index.select(batch_dim, 1))
    return index


def basic_sample_inputs_mint_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE

    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices)

    zero = ms.tensor(0, dtype=ms.int64)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((S, S), 1, M).to(ms.int64), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.copy(), _tensor(()))),
        (_tensor(()), (0, zero.copy(), ms.tensor(2.5).to(dtype))),
    )

    for tensor, args in test_cases:
        yield OpSampleInput(
            op_input=tensor,
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    def _gather_variable(shape, max_indices):
        return mint.randint(0, max_indices, shape)

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((2, 3, 4), 0, _gather_variable((3,)*3, 2), (3,)*3),
        ((3, 3, 4, 5), 1, _gather_variable((3,)*4, 3), (3, 4, 4, 3)),
        ((3, 3, 4, 5, 6), -1, _gather_variable((3,)*5,  6), (3,)*5),
    )
    for input_shape, dim, index, src in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(dim, index, make_arg(src)),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_dim{len(input_shape)}_{dim}d",
        )
    def _generate_sample_values(values):
        if len(values) % 2 == 0:
            np_array = np.array(values).reshape(2, -1)
        elif len(values) % 3 == 0:
            np_array = np.array(values).reshape(3, -1)
        else:
            np_array = np.array(values).reshape(-1, 1)
        index = _gather_variable(np_array.shape, np_array.shape[0])
        return OpSampleInput(
            op_input=make_tensor_with_np_array(np_array, dtype=dtype),
            op_args=(0, index, make_arg(np_array.shape)),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_dim0_{len(np_array.shape)}d",
        )
    #for small values:
    if dtype == ms.uint8:
        #for small values
        uint_small_values=dtypes_extra_values['_uint_small_values']
        yield _generate_sample_values(uint_small_values )
    elif dtype in [ms.int16, ms.int8, ms.int32, ms.int64]:
        # for small values
        yield _generate_sample_values(dtypes_extra_values['_int_small_values'])
        #for large values
        yield _generate_sample_values(dtypes_extra_values['_int_large_values'])
    elif dtype.is_floating_point:
        # for small values
        yield _generate_sample_values(dtypes_extra_values['_float_small_values'])
        # for large values
        yield _generate_sample_values(dtypes_extra_values['_fp16_large_values']
                                      if dtype==ms.float16 else dtypes_extra_values['_fp32_large_values'])
    if dtype.is_floating_point or dtype.is_complex:
        #for extremal value
        for t in [np.inf, -np.inf, np.nan]:
            input_values = np.full((3,4), t)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                op_args=(1, _gather_variable((2,5), 4), make_arg((2, 5))),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_extremal_dim1_2d",
            )
    for discontiguous in (True, False):
        yield OpSampleInput(
            op_input=make_arg((3,4,5), low=0, high=10, discontiguous=discontiguous),
            op_args=(0, _gather_variable((5,3,5), 3),
                     make_arg((5,3,5), discontiguous=discontiguous)),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_dim0_3d",
        )

def error_sample_inputs_mint_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    def _gather_variable(shape, max_indices):
        return mint.randint(0, max_indices, shape)
    input = make_arg((3, 4, 5))
    src=ms.Tensor(np.random.uniform(0,10,size=(1,1,1)), dtype=ms.float32)
    #index is either not int32 nor int64
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=input,
            op_args=(1 , make_tensor((3, 4, 5), dtype = ms.float32), make_arg((3, 4, 5))),
            op_kwargs={},
            sample_name=f"{op_info.name}_float32_dim1_index_float_3d",
        ),
        op_error_type=RuntimeError,
        op_error_info='index cannot be float'
    )
    # input is empty shape
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.Tensor(shape=(3, 0, 5), dtype=ms.float32),
            op_args=(1, _gather_variable((1, 0, 1), 0), ms.Tensor((1, 0, 1), dtype=ms.float32)),
            op_kwargs={},
            sample_name=f"{op_info.name}_dim1_shape_empty_3d",
        ),
        op_error_type=RuntimeError,
        op_error_info='input shape can not be empty'
    )
    # input and src does not have same shape-dim
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=input,
            op_args=(1, _gather_variable((3, 4), 4), make_arg((3, 4))),
            op_kwargs={},
            sample_name=f"{op_info.name}_shape_src_not_equal_input",
        ),
        op_error_type=RuntimeError,
        op_error_info='input is empty shape'
    )
    # src do not have the same dtype with input
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=input,
            op_args=(0, make_tensor((1, 1, 1), dtype=ms.int32, low=-2, high=0),
                     ms.Tensor(np.random.randint(0,10,size=(1,1,1)), dtype=ms.int32)),
            op_kwargs={},
            sample_name=f"{op_info.name}_dtype_src_input_not_match",
        ),
        op_error_type=TypeError,
        op_error_info='input has not match dtype with src'
    )
    # src has bigger shape than index
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=input,
            op_args=(0, make_tensor((1, 2, 1), dtype=ms.int32, low=-2, high=0), src),
            op_kwargs={},
            sample_name=f"{op_info.name}_index_bigger_shape",
        ),
        op_error_type=RuntimeError,
        op_error_info='src has bigger shape than index'
    )

def basic_sample_inputs_mint_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=(0,))
    cases = ((10, 10), (10,), ())
    # Other values will bring random number of output, skip it.
    p_vals = [0.5, 0.0, 1.0]
    # This is to handle special case for feature_alpha_dropout which has different
    # supported dtypes depending on `train` parameter
    training_vals = [True, False]

    for case, p, training in itertools.product(cases, p_vals, training_vals):
        if not case and dtype.is_floating_point:
            continue
        yield OpSampleInput(
            op_input=make_arg(case),
            op_args=(p, training),
            op_kwargs={},
            sample_name=op_info.name,
        )

def basic_sample_inputs_mint_expand_as(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, 1, 1), (S, S, S)),
             ((), ()),
             ((), (1, 1)),
             ((6, 7, 8, 9), (2, 3, 4, 5, 6, 7, 8, 9)),
             )

    for shape, shape_other in cases:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(make_arg(shape_other),),
            op_kwargs={},
            sample_name=op_info.name,
        )


# sample inputs functions for chunk
def basic_sample_inputs_mint_cumsum(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.cumsum op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S, S), (0,)),
        ((S, S, S), (1,)),
        ((), (0,)),
    )
    for shape, args in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_cumsum(op_info: OpInfo, dtype=None, device=None):
    """
    Generate extra sample inputs for mint.cumsum op.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((4,), (0,)),
        ((5, 9), (-1,)),
        ((3, 4, 8), (1,)),
        ((8, 7, 9, 7), (-2,)),
        ((7, 8, 5, 9, 3), (-2,)),
        ((9, 3, 7, 3, 4, 7, 3), (-6,)),
        ((2, 2, 2, 2, 2, 2, 3), (-1,)),
    )
    for shape, args in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_cumsum_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #tensor, dim,dtype
        ("1", (1, None)),
        (make_tensor(shape=(2,), dtype=ms.float32), ('1', None)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "float32"))
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.cumsum error values'
        )


def dynamic_sample_inputs_mint_cumsum(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.cumsum.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 5, None, None), dtype=dtype),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 4, 9)),
                    op_args=(1,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 5, 9, 7)),
                    op_args=(2,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4, 7, 7)),
                    op_args=(3,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 4, 7, 3, 4, 9)),
                    op_args=(1,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 5)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

# sample inputs functions for index_select
def basic_sample_inputs_mint_index_select(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.index_select op.
    """
    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S, S), 0),
        ((S, S, S), 1),
    )

    for shape, dim in cases:
        make_index = functools.partial(make_tensor, device=device, dtype=ms.int32, low=0, high=shape[dim]-1)
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(dim, make_index((shape[dim]-1,))),
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_index_select(op_info: OpInfo, dtype=None, device=None):
    """
    Generate extra sample inputs for mint.index_select op.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((4,), 0),
        ((5, 9), -1),
        ((3, 4, 8), 1),
        ((8, 7, 9, 7), 2),
        ((7, 8, 5, 9, 3), 2),
        ((9, 3, 7, 3, 4, 7, 3), -6),
        ((2, 2, 2, 2, 2, 2, 3,), -1),
        ((2, 2, 2, 2, 2, 2, 2, 4), 7),
        # empty tensor
        ((7, 0, 2, 8), -1),
    )
    for shape, dim in cases:
        make_index = functools.partial(make_tensor, device=device, dtype=ms.int32, low=0, high=shape[dim]-1)
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(dim, make_index((shape[dim]-1,))),
            sample_name=op_info.name,
        )
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values), 1)
        index_values = np.array(_float_small_values[:1])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(0, make_tensor_with_np_array(index_values, dtype=ms.int32, device=device)),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_small_value',
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        index_values = np.array(len(_large_values)-1).reshape(1,)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1, make_tensor_with_np_array(index_values, dtype=ms.int64, device=device)),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_large_value',
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((7, 9, 3), t)
                index = ms.Tensor(np.random.randint(0, 3, (4,)), dtype=ms.int32)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(2, index),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (9, 3)
        index_shape = (3,)
        dim = -1
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(
                dim,
                make_arg(index_shape, dtype=ms.int32, low=0, high=input_shape[dim]-1, discontiguous=discontiguous)
            ),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
        )


def error_inputs_other_op_mint_index_select_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, dim,index(tensor)
        ("1", (1, make_tensor(shape=(1,), dtype=ms.float32))),
        (make_tensor(shape=(2,), dtype=ms.float32), ("1", make_tensor(shape=(1,), dtype=ms.float32))),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "1"))
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.index_select error values'
        )

def error_inputs_other_op_tensor_index_select_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        (make_tensor(shape=(2,), dtype=ms.float32), ("1", make_tensor(shape=(1,), dtype=ms.float32))),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "1")),
        (make_tensor(shape=(2,), dtype=ms.float32), (2, make_tensor(shape=(1,), dtype=ms.float32))),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, make_tensor(shape=(1,2), dtype=ms.float32))),
    ]

    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops tensor.index_select error values'
        )

def dynamic_sample_inputs_mint_index_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.index_select.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 5, None, None), dtype=dtype),
                op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 4, 9)),
                    op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 5, 9, 7)),
                    op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4, 7, 7)),
                    op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 4, 7, 3, 4, 9)),
                    op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 5)),
                    op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 5, 3, 4, 1)),
                    op_args=(1, ms.Tensor(shape=(0,), dtype=ms.int32),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

# sample inputs functions for chunk
def basic_sample_inputs_mint_masked_select(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.masked_select op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)
    make_mask = functools.partial(make_tensor, device=device, dtype=ms.bool_)

    cases = (
        ((S, S), (S, S)),
        ((S, S), (S,)),
        ((), (S, S)),
    )
    for shape, args in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(make_mask(args), ),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_masked_select(op_info: OpInfo, dtype=None, device=None):
    """
    Generate extra sample inputs for mint.masked_select op.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)
    make_mask = functools.partial(make_tensor, device=device, dtype=ms.bool_)

    cases = (
        (3,),
        (3, 6),
        (2, 3, 4),
        (2, 2, 2, 2),
        (2, 2, 2, 3, 2)
    )
    for shape in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(make_mask(shape),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_masked_select_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, mask
        ("1", ms.tensor([1, 0, 1, 0], ms.bool_)),
        (ms.tensor([1, 2, 3, 4], ms.int8), 1),
        (ms.tensor([1, 2, 3, 4], ms.int8), ms.tensor([1, 0, 1, 0], ms.int8))
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.masked_select error values'
        )

def dynamic_sample_inputs_mint_masked_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.masked_select.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 8, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=ms.bool_),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(7, 8, 5)),
                    op_args=(ms.Tensor(shape=(7, 1, 5), dtype=ms.bool_),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 8, 9)),
                    op_args=(ms.Tensor(shape=(3, 1, 9), dtype=ms.bool_),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=ms.bool_),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4, 7, 7)),
                    op_args=(ms.Tensor(shape=(9, 4, 7, 1), dtype=ms.bool_),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 4, 7, 3, 4, 9)),
                    op_args=(ms.Tensor(shape=(4, 8, 1, 7, 3, 4, 9), dtype=ms.bool_),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=ms.bool_), ),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 5)),
                    op_args=(ms.Tensor(shape=(4, 1), dtype=ms.bool_),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(ms.Tensor(shape=(1,), dtype=ms.bool_),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

# sample inputs functions for chunk
def basic_sample_inputs_mint_nonzero(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.nonzero op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        (S,),
        (S, S),
        (S, S, S),
        (S, 1, S),
        (S, 0, S),
    )
    for shape in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_nonzero(op_info: OpInfo, dtype=None, device=None):
    """
    Generate extra sample inputs for mint.nonzero op.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        (3,),
        (3, 6),
        (2, 3, 4),
        (2, 2, 2, 2),
        (2, 2, 2, 3, 2),
        (2, 2, 2, 2, 2, 3),
        (2, 2, 2, 2, 2, 2, 3),
        (2, 2, 2, 2, 2, 2, 2, 4),
    )
    for shape in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_nonzero_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, as_tuple
        ("1", {"as_tuple": False}),
        (make_tensor(shape=(2,), dtype=ms.float32), {"as_tuple": "False"})
    ]
    for op_input, p_kwargs in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(),
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.nonzero error values'
        )

def dynamic_sample_inputs_mint_nonzero(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.nonzero.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 5, None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 2, 4, 7)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 5, 6, 9, 3)),
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
                    op_input=make_func(shape=(5, 4, 1, 1)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5, 2, 1, 4, 1)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 1)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

# sample inputs functions for chunk
def basic_sample_inputs_mint_split(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.split op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)])),
        ((S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)])),
        ((S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)]))
    )
    for shape, args in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_kwargs={'split_size_or_sections': args, 'dim': 0},
            sample_name=op_info.name,
        )


def error_inputs_other_op_mint_split_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #tensor, split_size_or_sections,dim
        ("1", (1, 0)),
        (make_tensor(shape=(2,), dtype=ms.float32), ('1', 0)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "0")),
        (make_tensor(shape=(2,), dtype=ms.float32), ((-1, -2), "0")),
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.split error values'
        )

def extra_sample_inputs_mint_split(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.split
    Reference torch's module tests for split.
    '''
    #not support 0d
    cases = (
        (ms.Tensor(np.full((5, 8, 4, 3, 5, 1, 1), np.inf), dtype=dtype), 5, -3),
        (ms.Tensor(np.full((1, 39, 16, 1, 1, 44), np.nan), dtype=dtype), 1, -6),
        (ms.Tensor(np.random.randint(-328, 3767, (1, 1, 5, 5, 1, 3, 5, 4)), dtype=dtype), 1, 0),
        (ms.Tensor(np.array([]).reshape(0, 5), dtype=dtype), [2, 3], 1),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype), 2, 0),
        (ms.Tensor(np.ones((10, 6)), dtype=dtype), 2, 0),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype), 2, 0),
        (ms.Tensor(np.full((4, 8), np.finfo(np.float32).min), dtype=dtype), 4, 1)
    )

    for inputs, split_size_or_sections, dims in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(split_size_or_sections, dims),
            op_kwargs={},
            sample_name=op_info.name,
        )

# sample inputs functions for chunk
def basic_sample_inputs_tensor_split(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.split op.
    """
    S = SMALL_DIM_SIZE
    M = MEDIUM_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)]), -2),
        ((S, M, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)]), 0),
        ((S, M, M), ([int(M / 2), M - int(M / 2) * 2, int(M / 2)]), 1)
    )
    for shape, args, dim in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(args, ),
            op_kwargs={'dim':dim},
            sample_name=op_info.name,
        )

def _is_ascend910(device):
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        return ascend_name in ('ascend910', 'ascend910b', 'ascend910_93', "ascend950")
    return False

def basic_reference_inputs_mint_mean(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs)
    #loss error: when input tensor dtype is flat64, set output dtype float16
    if dtype == ms.float64:
        for i in [ms.float32, ms.float64]:
            yield OpSampleInput(
                op_input=make_tensor((9, 3, 4, 5,), dtype=dtype, device=device, ),
                op_kwargs={'dim': (-2, -3), 'keepdim': True, 'dtype': i},
                sample_name=f'{op_info.name}_{dtype}out-dtype{i}_minus-dim_2_3'
            )
    #not support int/bool/complex dtype
    elif dtype in [ms.float32, ms.float16, ms.bfloat16]:
        for i in [ms.float32, ms.float64, ms.float16]:
            yield OpSampleInput(
                op_input=make_tensor((5, 8,), dtype=dtype, device=device,),
                op_kwargs={'dim': -2, 'keepdim': True, 'dtype': i},
                sample_name=f'{op_info.name}_{dtype}out-dtype{i}_minus-dim_2'
            )
    #only support complex dtype
    elif dtype in [ms.complex64, ms.complex128]:
        for i in [ms.complex64, ms.complex128]:
            yield OpSampleInput(
                op_input=make_tensor((4, 4, 5, 7, 3, 9, 9, 7,), dtype=dtype, device=device,),
                op_kwargs={'dim': -3, 'keepdim': True, 'dtype': i},
                sample_name=f'{op_info.name}_{dtype}out-dtype{i}_minus-dim_3'
            )
    else:
        for i in [ms.float32, ms.float16, ms.float64, ms.int8, ms.uint8, ms.int32, ms.uint32,
                  ms.int64, ms.uint64, ms.int16, ms.uint16, ms.bool_]:
            yield OpSampleInput(
                op_input=make_tensor((3,), dtype=dtype, device=device, ),
                op_kwargs={'dim': 0, 'keepdim': True, 'dtype': i},
                sample_name=f'{op_info.name}_{dtype}out-dtype{i}_minus-dim_0'
            )

def basic_reference_inputs_gelu(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    for i in ['none', 'tanh']:
        #gelu:negative dimensions are not allowed
        yield OpSampleInput(
            #add approximate test cases
            op_input=make_tensor((2, 2,), dtype=dtype, device=device, ),
            op_kwargs={'approximate': i},
            sample_name=f'{op_info.name}_{dtype}_minus-dim_2_2',
        )

def basic_reference_inputs_leaky_relu(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    yield OpSampleInput(
        op_input=make_tensor((2,), dtype=dtype, device=device,),
        op_kwargs={'negative_slope': random.choice([0.01, 1, -0.01])},
        sample_name=f'{op_info.name}_{dtype}_minus-dim_2'
    )

def basic_reference_inputs_tensor_astype(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    #loss error, dtype is uint8
    for i in [ms.float32, ms.float64, ms.float16]:
        #add copy and output dtype cases
        yield OpSampleInput(
            op_input=make_tensor((1, 6,), dtype=dtype, device=device, ),
            op_kwargs={'dtype': i,  'copy': random.choice([True, False])},
            sample_name=f'{op_info.name}_{dtype}out-dtype{i}_minus-dim_1_6'
        )

def extra_reference_inputs_tensor_astype(op_info, dtype=None, device=None, **kwargs):
    yield from extra_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    #bool dtype not support uint output dtype
    for i in [ms.int8, ms.int32, ms.int64, ms.int16, ms.bool_, ms.complex64, ms.complex128]:
        yield OpSampleInput(
            op_input=make_tensor((1, 6,), dtype=dtype, device=device, ),
            op_kwargs={'dtype': i, 'copy': random.choice([True, False])},
            sample_name=f'{op_info.name}_{dtype}out-dtype{i}_minus-dim_1_6'
        )

def basic_sample_inputs_repeat_tile(op_info, dtype, device=None, **kwargs):
    """Generate sample inputs for tile operations.

    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device to create tensors on.
        **kwargs: Additional keyword arguments.

    Yields:
        OpSampleInput: Sample input with tensor and dims tuple.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    test_cases = [
        ((2, 3), (2, 1)),
        ((2, 3), (1, 2)),
        ((2, 3), (2, 2)),
        ((), (2, 3)),
        ((4,), (3,)),
        ((2, 1, 3), (1, 2, 1)),
        ((100,), (3, 8, 2, 1)),
        ((2, 3), (1, 1)),
        ((2, 3, 4), (2, 1, 3)),
    ]

    for shape, dims in test_cases:
        yield OpSampleInput(
            op_input=make_func(shape),
            op_args=(dims,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_shape{shape}_dims{dims}"
        )

def extra_sample_inputs_repeat_tile(op_info, dtype, device=None, **kwargs):
    """Generate extra sample inputs for tile operations.

        Args:
            op_info: OpInfo object.
            dtype: Data type of the tensors.
            device: Device to create tensors on.
            **kwargs: Additional keyword arguments.

        Yields:
            OpSampleInput: Sample input with tensor and dims tuple.
        """
    for sample_input in extra_sample_inputs_for_shape_func(op_info, dtype, device):
        sample_input.op_args = tuple(random.sample(range(0,5), len(sample_input.op_input.shape)))
        yield sample_input

def extra_sample_inputs_tensor_repeat(op_info, dtype, device=None, **kwargs):
    # large mix small values
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
                op_args=([1, 2],),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_args=([1, 2],),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )
    # empty shape
    yield OpSampleInput(
        op_input=make_tensor(shape=(1, 0, 3), dtype=dtype, device=device),
        op_args=([1, 1, 2],),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_shape",
    )

def dynamic_sample_inputs_repeat_tile(op_info, dtype, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
            op_args=(mutable(input_data=(3, 0, 4, 5), dynamic_len=False),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(2, 3, 2, 4), dtype=dtype),
                op_args=(mutable(input_data=(4, 0, 4, 5), dynamic_len=False),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 3, 2, 3), dtype=dtype),
                op_args=(mutable(input_data=(4, 0, 4, 5), dynamic_len=False),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 3, 2, 2), dtype=dtype),
                op_args=(mutable(input_data=(4, 0, 4, 5), dynamic_len=False),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )
    # dynamic rank
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=dtype),
            op_args=(mutable(input_data=(3,1,1,1), dynamic_len=True),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_rank_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(3, 3)),
                op_args=(mutable(input_data=(1,1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(1, 3)),
                op_args=(mutable(input_data=(1, 1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(3,)),
                op_args=(mutable(input_data=(1, 1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
        ),
    )

def dynamic_sample_inputs_repeat(op_info, dtype, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
            op_args=(mutable(input_data=[3, 0, 4, 5], dynamic_len=False),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_shape_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(2, 3, 2, 4), dtype=dtype),
                op_args=(mutable(input_data=[4, 0, 4, 5], dynamic_len=False),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 3, 2, 3), dtype=dtype),
                op_args=(mutable(input_data=[4, 0, 4, 5], dynamic_len=False),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 3, 2, 2), dtype=dtype),
                op_args=(mutable(input_data=[4, 0, 4, 5], dynamic_len=False),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )
    # dynamic rank
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=dtype),
            op_args=(mutable(input_data=[3, 1, 1, 1], dynamic_len=True),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_rank_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(3, 3)),
                op_args=(mutable(input_data=[1, 1], dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(1, 3)),
                op_args=(mutable(input_data=[1, 1], dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(3,)),
                op_args=(mutable(input_data=[1, 1], dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
        ),
    )

def error_sample_inputs_repeat(op_info, dtype, device=None, **kwargs):
    #
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,3), dtype=ms.int32, device=device),
            op_args=(3,),
            op_kwargs={},
            sample_name=f"{op_info.name}_repeats_out_of_dimension",
        ),
        op_error_type=ValueError,
        op_error_info='number of items of repeats cannot be smaller than the number if dimensions of tensor',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.int32, device=device),
            op_args=((2, -1),),
            op_kwargs={},
            sample_name=f"{op_info.name}_repeats_neg_value",
        ),
        op_error_type=RuntimeError,
        op_error_info='repeats has value smaller than 0',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.int32, device=device),
            op_args=(np.array([2, 3]),),
            op_kwargs={},
            sample_name=f"{op_info.name}_repeats_wrong_type",
        ),
        op_error_type=TypeError,
        op_error_info='repeats has wrong type',
    )

#convert tensor for op
def convert_tensor_for_op(op_sample_input, op_info):
    if isinstance(op_info, BinaryOpInfo):
        op_sample_input.op_args = (op_sample_input.op_input,)
    elif isinstance(op_info, ReductionOpInfo):
        op_sample_input.op_kwargs = {"dim": None, "keepdim": False} | op_sample_input.op_kwargs
        op_sample_input.sample_name = f"{op_sample_input.sample_name}_default"
    return  op_sample_input

#basic func with keyword parameter dtype
def structure_basic_op_func_with_dtype_arg(op_info, dtype, device=None, **kwargs):
    dtypes_dict = next((dtypes for name, dtypes in op_dtype_need_convert.items()
                        if op_info.name.startswith(name)), {})
    if dtype in dtypes_dict:
        dtype_list = [dtypes_dict[dtype]] if not isinstance(dtypes_dict[dtype],list) else dtypes_dict[dtype]
        dtype_list = list(set(dtype_list) & set(op_info.get_supported_dtypes(device)))
        for to_dtype in dtype_list:
            op_sample_input = OpSampleInput(
                op_input=make_tensor((5, 5), dtype=dtype, device=device),
                op_args=(),
                op_kwargs={"dtype": to_dtype},
                sample_name=f"{op_info.name}_{dtype}_to_{to_dtype}",
                grad_position=() if dtype.is_floating_point and not to_dtype.is_floating_point else None
            )
            yield convert_tensor_for_op(op_sample_input, op_info)

#extra func with keyword parameter dtype
def structure_extra_op_func_with_dtype_arg(op_info, dtype, device=None, **kwargs):
    dtypes_dict = next((dtypes for name, dtypes in op_dtype_need_convert.items()
                        if op_info.name.startswith(name)), {})
    _uint_values = dtypes_extra_values['_uint_small_values']
    _int_values = dtypes_extra_values['_int_small_values']
    _float_vals = dtypes_extra_values['_float_small_values']
    tensor = None
    if dtype in dtypes_dict:
        dtype_list = [dtypes_dict[dtype]] if not isinstance(dtypes_dict[dtype],list) else dtypes_dict[dtype]
        dtype_list = list(set(dtype_list) & set(op_info.get_supported_dtypes(device)))
        for to_dtype in dtype_list:
            if dtype == ms.uint8 :
                tensor = make_tensor_with_np_array(np.array(_uint_values), dtype=dtype, device=device)
            elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
                tensor = make_tensor_with_np_array(np.array(_int_values) , dtype=dtype, device=device)
            elif dtype.is_floating_point:
                tensor = make_tensor_with_np_array(np.array(_float_vals), dtype=dtype, device=device)
            if tensor is not None:
                op_sample_input = OpSampleInput(
                        op_input=tensor,
                        op_args=(),
                        op_kwargs={"dtype": to_dtype},
                        sample_name=f"{op_info.name}_extra_{dtype}_convert_to_{to_dtype}",
                    )
                yield convert_tensor_for_op(op_sample_input, op_info)

            if dtype.is_floating_point and device=="ascend":
                op_sample_input = OpSampleInput( op_input=ms.Tensor([np.inf, -np.inf, np.nan]),
                        op_args=(),
                        op_kwargs={"dtype": to_dtype},
                        sample_name=f"{op_info.name}_extra_inf_{dtype}_convert_to_{to_dtype}"
                    )
                yield convert_tensor_for_op(op_sample_input, op_info)

#basic reduction func with keyword parameter dtype
def basic_reduction_op_func_with_dtype_arg(op_info, dtype, device, **kwargs):
    yield from basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs)
    yield from structure_basic_op_func_with_dtype_arg(op_info, dtype, device, **kwargs)

#extra reduction func with keyword parameter dtype
def extra_reduction_op_func_with_dtype_arg(op_info, dtype, device, **kwargs):
    yield from extra_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs)
    yield from structure_extra_op_func_with_dtype_arg(op_info, dtype, device, **kwargs)

#basic unary func with keyword parameter dtype
def basic_unary_op_func_with_dtype_arg(op_info, dtype, device, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    yield from structure_basic_op_func_with_dtype_arg(op_info, dtype, device, **kwargs)

#extra unary func with keyword parameter dtype
def extra_unary_op_func_with_dtype_arg(op_info, dtype, device, **kwargs):
    yield from extra_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    yield from structure_extra_op_func_with_dtype_arg(op_info, dtype, device, **kwargs)

#basic binary func with keyword parameter dtype
def basic_binary_op_func_with_dtype_arg(op_info, dtype, device, **kwargs):
    yield from basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)
    yield from structure_basic_op_func_with_dtype_arg(op_info, dtype, device, **kwargs)

#extra binary func with keyword parameter dtype
def extra_binary_op_func_with_dtype_arg(op_info, dtype, device, **kwargs):
    yield from extra_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)
    yield from structure_extra_op_func_with_dtype_arg(op_info, dtype, device, **kwargs)

def basic_sample_inputs_mint_unique_consecutive(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.unique op.
    Reference torch's sample_inputs_unique:
      - sizes: (), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S)
      - flags: sorted/return_inverse/return_counts in {False, True}
      - dims: None, -2, -1, 0, 1, 2 with validity checks
      - inputs: all-zeros, mixed 0/1, random values
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    sizes = ((), (S,), (S, S, S), (S, 1, S), (S, 0, S))
    flag_cases = [(ret_inv, ret_cnt) for ret_inv in (False, True) for ret_cnt in (False, True)]
    #TODO: mint.unique has different behavior when dim is None or -1, skip it temporarily.
    dim_cases = (-2, 0, 2)

    for shape in sizes:
        rank = len(shape)
        for ret_inv, ret_cnt in flag_cases:
            for dim in dim_cases:
                # skip invalid dim per rank
                if rank == 0 and dim is not None:
                    continue
                if dim is not None and (dim < -rank or dim >= rank):
                    continue
                # for shapes with a zero dimension, if dim selects a different axis than the zero axis, skip
                if 0 in shape and dim is not None:
                    zero_axis = shape.index(0)
                    dim_norm = dim if dim >= 0 else dim + rank
                    if dim_norm != zero_axis:
                        continue

                # Use positional args as per API (no keyword args for unique)
                # Order: (sorted, return_inverse, return_counts, dim)
                unique_kwargs = {'return_inverse':ret_inv, 'return_counts':ret_cnt, 'dim':dim}

                # (1) all-zero tensor
                zero_np_dtype=dtype_to_np_dtype[dtype]
                zeros_np = np.zeros(shape, dtype=zero_np_dtype)
                zeros_ms = make_tensor_with_np_array(zeros_np, dtype=dtype, device=device)
                yield OpSampleInput(
                    op_input=zeros_ms,
                    op_args=(),
                    op_kwargs=unique_kwargs,
                    sample_name=op_info.name,
                )

                # (2) mixed 0/1 values (then cast to dtype)
                ones_zeros_np = np.random.randint(0, 2, size=shape).astype(zero_np_dtype)
                mixed_ms = make_tensor_with_np_array(ones_zeros_np, dtype=dtype, device=device)
                yield OpSampleInput(
                    op_input=mixed_ms,
                    op_args=(),
                    op_kwargs=unique_kwargs,
                    sample_name=op_info.name,
                )

                # (3) many random values
                yield OpSampleInput(
                    op_input=make_x(shape),
                    op_args=(),
                    op_kwargs=unique_kwargs,
                    sample_name=op_info.name,
                )
def basic_sample_inputs_GroupNorm(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.nn.GroupNorm.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Test cases: (input_shape, num_groups, num_channels, eps, affine, desc)
    test_cases = [
        ((4, 6), 2, 6, 2.0, None, 'redline test case'),
        ((4, 6, 5), 3, 6, 1e-3, None, '1d_affine'),
        ((4, 12), 3, 12, 1e-3, None, '1d_affine_GN'),
        ((150, 6), 1, 6, 1e-3, None, '1d_affine_large_batch'),
        ((4, 5, 5), 5, 5, 1e-3, False, '1d_no_affine_IN'),
        ((4, 10), 1, 10, 1e-3, False, '1d_no_affine_LN'),
        ((4, 6, 2, 3), 3, 6, 1e-3, None, '2d_affine'),
        ((4, 3, 2, 3), 3, 3, 1e-3, False, '2d_no_affine_IN'),
        ((4, 3, 2, 3), 1, 3, 1e-3, False, '2d_no_affine_LN'),
    ]

    for input_shape, num_groups, num_channels, eps, affine, desc in test_cases:
        op_kwargs = {'eps': eps}
        if affine is not None:
            op_kwargs['affine'] = affine
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(num_groups, num_channels),
            op_kwargs=op_kwargs,
            sample_name=f'{op_info.name}_{desc}'
        )


def basic_sample_inputs_LayerNorm(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.nn.LayerNorm.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Test cases: (input_shape, normalized_shape, eps, elementwise_affine, bias, desc)
    test_cases = [
        ((9,), [9], 3.7642600490219508e-06, False, True, 'redline test case'),
        ((4, 5, 5), [5], 1e-3, None, None, '1d_elementwise_affine'),
        ((128, 5, 5), [5], 1e-3, None, None, '1d_elementwise_affine_large_batch'),
        ((4, 5, 5), [5], 1e-3, False, False, '1d_no_elementwise_affine'),
        ((4, 2, 2, 5), [2, 2, 5], 1e-3, None, None, '3d_elementwise_affine'),
        ((4, 2, 2, 5), [2, 2, 5], 1e-3, False, False, '3d_no_elementwise_affine'),
        ((0, 5), [5], 1e-3, None, None, '1d_empty_elementwise_affine'),
        ((4, 2, 2, 5), [2, 2, 5], 1e-3, True, False, '3d_elementwise_affine_no_bias'),
    ]

    for input_shape, normalized_shape, eps, elementwise_affine, bias, desc in test_cases:
        op_kwargs = {'eps': eps}
        if elementwise_affine is not None:
            op_kwargs['elementwise_affine'] = elementwise_affine
        if bias is not None:
            op_kwargs['bias'] = bias
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(normalized_shape,),
            op_kwargs=op_kwargs,
            sample_name=f'{op_info.name}_{desc}'
        )


def basic_sample_inputs_layer_norm(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.nn.functional.layer_norm.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Basic shape test cases: (input_shape, normalized_shape, eps, desc)
    shape_cases = [
        ((1, 2, 3), (1, 2, 3), 0.5, 'eps_0.5'),
        ((2, 2, 3), (2, 3), -0.5, 'eps_neg0.5'),
        # ((1,), (1,), None, '1d_default'), # Todo: to fix forward precision issue
        ((7,), (7,), 3.7642600490219508e-06, 'redline_eps'),
        ((1, 2), (2,), None, '2d_default'),
        ((0, 1), (1,), None, 'empty_batch'),
    ]

    for input_shape, normalized_shape, eps, desc in shape_cases:
        # Test 1: no weight, no bias
        if eps is not None:
            op_args = (normalized_shape, None, None, eps)
        else:
            op_args = (normalized_shape,)  # Use all defaults
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=op_args,
            op_kwargs={},
            sample_name=f'{op_info.name}_{desc}_no_weight_bias'
        )

        # Test 2: with only weight
        if eps is not None:
            op_args = (normalized_shape, make_arg(normalized_shape), None, eps)
        else:
            op_args = (normalized_shape, make_arg(normalized_shape))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=op_args,
            op_kwargs={},
            sample_name=f'{op_info.name}_{desc}_weight_only'
        )

        # Test 3: with only bias
        if eps is not None:
            op_args = (normalized_shape, None, make_arg(normalized_shape), eps)
        else:
            op_args = (normalized_shape, None, make_arg(normalized_shape))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=op_args,
            op_kwargs={},
            sample_name=f'{op_info.name}_{desc}_bias_only'
        )

        # Test 4: with both weight and bias
        if eps is not None:
            op_args = (normalized_shape, make_arg(normalized_shape), make_arg(normalized_shape), eps)
        else:
            op_args = (normalized_shape, make_arg(normalized_shape), make_arg(normalized_shape))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=op_args,
            op_kwargs={},
            sample_name=f'{op_info.name}_{desc}_weight_bias'
        )

    # Minimal test case without optional args
    yield OpSampleInput(
        op_input=make_arg((1, 2)),
        op_args=((2,),),
        op_kwargs={},
        sample_name=f'{op_info.name}_minimal'
    )


def basic_sample_inputs_mint_cat(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.cat op.
    """
    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S), (S, S), {'dim': -1}),
        ((S, S), (S, S), {'dim':  1}),
        ((S, S), (S, S), {'dim': 0}),
        ((1, 2, 3), (1, 2, 3), {'dim': -2}),
        ((1,), (1,), {})
    )
    for input_shape_1, input_shape_2, p_kwargs in cases:
        yield OpSampleInput(
            op_input=tuple([make_arg(input_shape_1), make_arg(input_shape_2)]),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )


def mint_cat_func_with_flatten_inputs(*args):
    """A kwargs-free wrapper for mint.cat used by ms.grad in tests.

    Note:
        OpSampleInput.convert_to_args() expands op_input tuples/lists into multiple
        positional arguments. For mint.cat, the first argument should be a single
        sequence of tensors, so this wrapper repacks positional tensor args into
        a tuple and treats an optional trailing int as dim.
    """
    if not args:
        raise ValueError("mint_cat_func_with_flatten_inputs expects at least one argument.")
    dim = 0
    tensors = args
    # In our tests, dim is passed as a Python int when present.
    if isinstance(args[-1], int):
        dim = args[-1]
        tensors = args[:-1]
    return mint.cat(tuple(tensors), dim=dim)


def extra_sample_inputs_mint_cat(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate extra sample inputs for mint.cat op.
    """

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3, 7, 6, 8), (3, 7, 6, 8), {'dim': 1}),
        ((3, 7, 6, 8, 2, 3, 4), (3, 7, 6, 8, 2, 3, 4), {'dim': 0}),
        ((3, 7, 6, 8, 2, 3, 4), (3, 7, 6, 8, 2, 3, 4), {'dim': 2}),
        ((50, 41), (50, 41), {'dim': -1}),
    )
    for input_shape_1, input_shape_2, p_kwargs in cases:
        yield OpSampleInput(
            op_input=tuple([make_arg(input_shape_1), make_arg(input_shape_2)]),
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
        )


def sample_inputs_permute(op_info, device, dtype, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = [((1, 2, 3, 4), (0, 2, 3, 1)),
             ((1, 2, 3, 4), (0, -2, -1, 1)),
             ((), ()),
             ((1, 2, 3, 4), (2, 1, 3, 0))]

    for shape, args in cases:
        yield OpSampleInput(make_arg(shape), op_args=(args,), op_kwargs={}, sample_name=op_info.name)


def basic_sample_inputs_mint_permute(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from sample_inputs_permute(op_info, device, dtype, **kwargs)

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), ()),
        ((1,), (0,)),
        ((2, 2), (1, 0)),
        ((2, 2), (0, 1)),
        ((2, 0, 1), (0, 2, 1)),
        ((3, 4, 2), (2, 1, 0)),
        ((3, 4, 2), (1, 0, 2)),
        ((3, 4, 2), (0, 1, 2)),
    )

    # Adds tricky permutations and permutations with noncontiguity
    for shape, permutation in cases:
        a = make_arg(shape)
        yield OpSampleInput(a, op_args=(permutation,), op_kwargs={}, sample_name=op_info.name)


def error_inputs_other_op_mint_permute_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #tensor, dim(tuple)
        ("1", ((0, 1),)),
        (make_tensor(shape=(2,), dtype=ms.float32), ('1',)),
        (make_tensor(shape=(2,), dtype=ms.float32), ((-1, -2),)),
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.permute error values'
        )

def extra_sample_inputs_mint_permute(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.permute
    Reference torch's module tests for permute.
    '''
    cases = (
        (ms.Tensor(np.full((4,), np.inf), dtype=dtype), ((0,),),),
        (ms.Tensor(np.full((9, 3, 5, 9), np.nan), dtype=dtype), ((3, 1, 2, 0),),),
        (ms.Tensor(np.random.randint(0, 10, size=(9, 4, 5, 9, 8, 5, 4, 7)), dtype=dtype),
         ((7, 3, 2, 5, 6, 0, 4, 1),),),
        (ms.Tensor(np.array([]).reshape(0, 5), dtype=dtype), ((1, 0),),),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype), ((1, 0),),),
        (ms.Tensor(np.zeros((8, 4)), dtype=dtype), ((0, 1),),),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype), ((1, 0),),),
        (ms.Tensor(np.full((4, 8), np.finfo(np.float32).min), dtype=dtype), ((0, 1),),)
    )

    for inputs, dims in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=dims,
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_permute(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.permute.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 8, None, None), dtype=dtype),
                op_args=(ms.mutable((3, 2, 1, 0)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 8, 4, 3)),
                    op_args=(ms.mutable((3, 2, 1, 0)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 8, 2, 1)),
                    op_args=(ms.mutable((3, 2, 1, 0)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.mutable((3, 2, 1, 0)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4, 1, 1)),
                    op_args=(ms.mutable((3, 2, 1, 0)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5, 2, 1)),
                    op_args=(ms.mutable((3, 2, 1, 0)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.mutable((1, 0)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4)),
                    op_args=(ms.mutable((1, 0)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 1)),
                    op_args=(ms.mutable((1, 0)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def basic_sample_inputs_mint_unsqueeze(op_info: OpInfo, dtype=None, device=None, **kwargs):
    shapes_and_axes = [
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), 3),
        ((3, 4, 5), -1),
        ((3, 4, 5), -3),
        ((), 0),
        ((), -1),
        ((1,), 0),
        ((1,), -1),
    ]

    for shape, axis in shapes_and_axes:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None,)
        yield OpSampleInput(tensor, op_args=(axis, ))

def error_inputs_other_op_mint_unsqueeze_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, dim,
        ("1", (1,)),
        (make_tensor(shape=(2,), dtype=ms.float32), ('0',)),
        (make_tensor(shape=(2,), dtype=ms.float32), (True,))]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.unsqueeze error values'
        )

def extra_sample_inputs_mint_unsqueeze(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.unsqueeze
    Reference torch's module tests for unsqueeze.
    '''
    cases = (
        (ms.Tensor(np.full((9, 7, 8, 4, 5, 5), np.inf), dtype=dtype), 3),
        (ms.Tensor(np.full((7, 9, 3, 3, 4), np.nan), dtype=dtype), 4),
        (ms.Tensor(np.random.randn(8, 5, 9, 9, 9, 7, 4), dtype=dtype), 6),
        (ms.Tensor(np.array([]).reshape(3, 0), dtype=dtype), 2),
        # discontinuous
        (ms.Tensor(np.arange(32).reshape(8, 4).astype(np.float32)[:, ::2], dtype=dtype), 2,),
        (ms.Tensor(np.zeros((8, 4)), dtype=dtype), 0),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype), 2),
        (ms.Tensor(np.full((4, 8), np.finfo(np.float32).min), dtype=dtype), 0)
    )

    for inputs, dims in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(dims,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_unsqueeze(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.unsqueeze.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, None, None, None, None, None), dtype=dtype),
                op_args=(5,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 3, 5, 7, 9, 8, 10)),
                    op_args=(5,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 3, 7, 5, 9, 8, 4)),
                    op_args=(5,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 5, 7, 8)),
                    op_args=(1,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(13, 8, 20, 16, 1)),
                    op_args=(1,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 3)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2,)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def basic_sample_inputs_mint_transpose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE
    M = MEDIUM_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    cases = (((1, 2, 3), (-1, -2)),
             ((1, 2, 3), (-1, 2)),
             ((1, 2, 3), (1, -2)),
             ((1, 2, 3), (1, 2)),
             ((), (0, 0)),
             ((1, ), (0, 0)),
             ((M, M), (0, 1)),
             ((S, S, S), (2, 0)), )

    for shape, args in cases:
        yield OpSampleInput(make_arg(shape), op_args=args, op_kwargs={}, sample_name=op_info.name)


def error_inputs_other_op_mint_transpose_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #tensor, dim0,dim1,dtype
        ("1", (0, 1), {}),
        (make_tensor(shape=(2,), dtype=ms.float32), (-1, 1), {}),
        (make_tensor(shape=(2,), dtype=ms.float32), (0, "1"), {}),
        (make_tensor(shape=(2,), dtype=ms.float32), (4, 1), {}),
        (make_tensor(shape=(2,), dtype=ms.float32), (0, 1), {"dtype": "float32"}),
    ]
    for op_input, args, p_kwargs in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs=p_kwargs,
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.transpose error values'
        )

def extra_sample_inputs_mint_transpose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.transpose
    Reference torch's module tests for transpose.
    '''
    cases = (
        (ms.Tensor(np.full((3, 9, 3, 7), np.inf), dtype=dtype), -3, -1),
        (ms.Tensor(np.full((3, 9), np.nan), dtype=dtype), -1, -1),
        (ms.Tensor(np.random.randint(0, 255, (8, 5, 7, 9, 4, 9, 5, 5)), dtype=dtype), -8, 4),
        (ms.Tensor(np.array([]).reshape(3, 0), dtype=dtype), 0, 1),
        # discontinuous
        (ms.Tensor(np.arange(32).reshape(8, 4).astype(np.float32)[:, ::2], dtype=dtype), 0, 1),
        (ms.Tensor(np.zeros((8, 4)), dtype=dtype), 0, 1),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype), 0, 1),
        (ms.Tensor(np.full((4, 8), np.finfo(np.float32).min), dtype=dtype), 0, 1)
    )

    for inputs, dim0, dim1 in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(dim0, dim1),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_transpose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.transpose.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 8, None, None), dtype=dtype),
                op_args=(1, 0),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 8, 4, 3)),
                    op_args=(1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 8, 2, 1)),
                    op_args=(1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, 0),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4, 1, 1)),
                    op_args=(1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5, 2, 1, 4, 1)),
                    op_args=(1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, 0),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4)),
                    op_args=(1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 1)),
                    op_args=(1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )
def basic_sample_inputs_mint_topk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE
    M = MEDIUM_DIM_SIZE

    def get_tensor_input(size):
        # get no repeat value
        input_sum = 1
        for num in size:
            input_sum *= num
        input_np = np.arange(0, input_sum)
        np.random.shuffle(input_np)
        return ms.tensor(input_np.reshape(*size), dtype=dtype)

    yield OpSampleInput(get_tensor_input((S, )), op_args=(1, 0, True, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, ))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1, True, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2, True, True))


def error_inputs_other_op_mint_topk_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, k,dim,largest,sorted
        ("1", (1, 1, True, True)),
        (make_tensor(shape=(2,), dtype=ms.float32), ("1", 1, True, True)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, "1", True, True)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, 1, "True", True)),
        (make_tensor(shape=(2,), dtype=ms.float32), (1, 1, True, "True"))
    ]
    for op_input, args in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.topk error values'
        )

def extra_sample_inputs_mint_topk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.topk.
    Reference torch's module tests for topk.
    '''
    cases = (
        (ms.Tensor(np.full((3, 9, 3, 7), np.inf), dtype=dtype), 1, 0, True, True),
        (ms.Tensor(np.full((3, 9), np.nan), dtype=dtype), 1, 0, True, True),
        # discontinuous
        (ms.Tensor(np.arange(32).reshape(8, 4).astype(np.float32)[:, ::2], dtype=dtype), 1, 1, False, False),
        (ms.Tensor(np.array([]).reshape(0, 5), dtype=dtype), 0, 0, True, False),
    )
    #fix the random seed,Prevent random inputs from having a probability of causing misalignment with the benchmark
    np.random.seed(42)
    random_cases = (ms.Tensor(np.random.randint(-128, 127, (2, 2, 3, 2, 2)), dtype=dtype),
                    1, 2, False, False)
    all_cases = cases[:2] + (random_cases,) + cases[2:]

    for inputs, k, dims, largest, sorted in all_cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(k, dims, largest, sorted),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_topk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.topk.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 8, None, None), dtype=dtype),
                op_args=(1, 0, True, True),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 8, 4, 3)),
                    op_args=(1, 0, True, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 8, 2, 1)),
                    op_args=(1, 0, True, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, 0, True, True),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4, 1, 1)),
                    op_args=(1, 0, True, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 4, 7, 8, 2)),
                    op_args=(1, 0, True, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, 0, True, True),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4)),
                    op_args=(1, 0, True, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 1)),
                    op_args=(1, 0, True, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def basic_sample_inputs_mint_bernoulli(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        (),
        (5, 4, 8),
        (9, 5, 4, 3),
        (4, 7, 9, 5, 5, 4),
        (7, 3, 3, 4, 7, 7, 4),
        (8, 9, 7, 8, 3, 3, 9, 5),
    )
    for shape in cases:
        yield OpSampleInput(make_arg(shape, low=0, high=1),
                            op_args=(),
                            sample_name=f"{op_info.name}_{len(shape)}d")

def extra_sample_inputs_mint_bernoulli(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    # value out of range [0,1] will raise exception, so  skip large/inf values.
    yield OpSampleInput(make_arg(shape=(3, 0, 5), low=0, high=1),
                        op_args=(),
                        sample_name=f"{op_info.name}_{dtype}_empty")
    #for disncontiguous values:
    for discontiguous in [True, False]:
        yield OpSampleInput(make_arg(shape=(3, 4, 5), low=0, high=1, discontiguous=discontiguous),
                            op_args=(),
                            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}")

def basic_sample_inputs_adaptive_avg_pool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as (input shape, output size)
    cases = (
        #((0, 8, 8), (5,)), ms doesn't support this case
        ((8, 4, 9), 6),
        ((3, 8, 8), 5),
        ((3, 8, 8), 1)
    )

    for input_shape, output_size in cases:
        # Batched
        yield OpSampleInput(make_arg(input_shape), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)
        # Unbatched
        yield OpSampleInput(make_arg(input_shape[1:]), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)


def basic_sample_inputs_adaptive_avg_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as (input shape, output size)
    cases = (
        ((1, 7, 3, 8), (1)),
        ((1, 8, 8, 8), (5, 7)),
        ((2, 8, 8, 8), (None, 7)),
        ((1, 8, 4, 3), (5, None)),
        ((1, 8, 4, 3), (None, None)),
        ((1, 8, 4, 3), (5)),
    )

    for input_shape, output_size in cases:
        # Batched
        yield OpSampleInput(make_arg(input_shape), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)
        # Unbatched
        yield OpSampleInput(make_arg(input_shape[1:]), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)

def basic_sample_inputs_adaptive_avg_pool3d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as (input shape, output size)
    cases = (
        ((1, 38, 5, 1, 12), (50, 12, 26)),
        #((0, 8, 8, 8, 8), (5, 7, 4)),
        # ms don't support
        ((1, 8, 4, 3, 7), (None, None, None)),
        ((1, 8, 4, 3, 7), (1, 1, 1)),
        ((3, 3, 8, 8, 6), (5, 7, None)),
        ((1, 3, 8, 8, 6), (5, None, 2)),
        ((3, 3, 8, 8, 6), (None, 3, 2)),
    )

    for input_shape, output_size in cases:
        # Batched
        yield OpSampleInput(make_arg(input_shape), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)
        # Unbatched
        yield OpSampleInput(make_arg(input_shape[1:]), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)

def basic_sample_inputs_avgpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Order: input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    cases = (((5, 5, 8, 8), 4, 1, 2, False, False, 18),
             ((1, 3, 9, 9), 3, 1, 1, True, False, 2),
             ((1, 3, 9, 9), (4, 4), (2, 3), 1, True, False, 2),
             ((1, 3, 9, 9), (6, 6), (3, 3), (2, 3), True, True, 2),
             ((2, 3, 9, 9), (3, 3), (1, 1), (1, ), True, False, 2),
             #((1, 1, 4, 4), (2, 2), (), (0, ), False, True, -2),
             ((1, 2, 6, 6), (4, 4), (2, 2), (2, ), True, True, None))

    for input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override in cases:
        yield OpSampleInput(make_arg(input_shape),
                            op_args=(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override),
                            op_kwargs={},
                            sample_name=op_info.name)
    # Case with just input_shape and kernel_size
    yield OpSampleInput(make_arg((1, 3, 9, 9)), op_args=((3, 3),), op_kwargs={}, sample_name=op_info.name)


def basic_sample_inputs_max_pool(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (((9, 8, 4, 3), 2, 1, 0, 1, False, False),)

    for input_shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices in cases:
        yield OpSampleInput(make_arg(input_shape),
                            op_args=(kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                            op_kwargs={},
                            sample_name=op_info.name)


def basic_sample_inputs_adaptivegpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as (input shape, output size)
    cases = (
        ((1, 7, 5, 9), (48)),
        ((1, 8, 8, 8), (5, 7)),
        ((2, 8, 8, 8), (None, 7)),
        ((1, 8, 4, 3), (5, None)),
        ((1, 8, 4, 3), (None, None)),
        ((1, 8, 4, 3), (5)),
    )

    for input_shape, output_size in cases:
        # Batched
        yield OpSampleInput(make_arg(input_shape), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)
        # Unbatched
        yield OpSampleInput(make_arg(input_shape[1:]), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)


def basic_sample_inputs_adaptivegpool3d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as (input shape, output size)
    cases = (
        ((1, 18, 6, 1, 32), (None, None, 3)),
        ((1, 8, 4, 3, 7), (None, None, None)),
        ((1, 8, 4, 3, 7), (1, 1, 1)),
        ((3, 3, 8, 8, 6), (5, 7, None)),
        ((1, 3, 8, 8, 6), (5, None, 2)),
        ((3, 3, 8, 8, 6), (None, 3, 2)),
    )

    for input_shape, output_size in cases:
        # Batched
        yield OpSampleInput(make_arg(input_shape), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)
        # Unbatched
        yield OpSampleInput(make_arg(input_shape[1:]), op_args=(output_size,), op_kwargs={}, sample_name=op_info.name)


def basic_sample_inputs_mint_div(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)

    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    _input = make_arg((S, S), low=op_info.input_low,
                       high=op_info.input_high, exclude_value=op_info.input_exclude_value)
    _other = make_arg((S, S), low=op_info.other_low,
                       high=op_info.other_high, exclude_value=op_info.other_exclude_value)

    # Add kwarg rounding_mode
    rounding_mode = ['floor', 'trunc']
    if dtype  is not ms.bool_:
        for value in rounding_mode:
            yield OpSampleInput(
                op_input=_input,
                op_args=(_other,),
                op_kwargs={'rounding_mode': value},
                sample_name=op_info.name
            )


def extra_sample_inputs_mint_div(op_info: OpInfo, dtype=None, device=None, **kwargs):
    if dtype is not ms.bool_:
        for input in extra_reference_inputs_binary_op_common_func(op_info, dtype=dtype, device=device, **kwargs):
            for value in [None, 'floor', 'trunc']:
                input.op_kwargs.update({'rounding_mode': value})
                yield input

# op error sense for convolutional operators
def error_inputs_other_op_with_convolution(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    conv_dim=kwargs.get('conv_dim')

    conv_dim={'conv1d':1,'conv2d':2,'conv3d':3}[conv_dim]

    def shape_with_nc(N: int, C: int, side: int, rank: int):
        return tuple([N, C] + [side] * rank)

    N_in = 2
    C_in = 4
    C_out = 1
    feature_map = 4
    kernel_size = 3
    group = 2
    padding = 4
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim )),
            op_args=(make_arg(shape_with_nc(C_out, C_in, kernel_size, conv_dim )),),
            op_kwargs={'bias':make_arg((2,))},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='bias does not equal output channel',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim )),
            op_args=(make_arg(shape_with_nc(C_out, int(C_in/group+1), kernel_size, conv_dim )),),
            op_kwargs={'groups': group},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='C_in/group must be equal weight[1]',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim)),
            op_args=(make_arg(shape_with_nc(C_out, int(C_in/group-1), kernel_size, conv_dim)),),
            op_kwargs={'groups': group},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='C_in/group must be equal weight[1]',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim)),
            op_args=(make_arg(shape_with_nc(group+1, int(C_in/group), kernel_size, conv_dim)),),
            op_kwargs={'groups': group},
            sample_name=op_info.name,
        ),
        # conv1d raise RuntimeError
        op_error_type=(ValueError, RuntimeError),
        op_error_info='C_out should be divisible by group and C_out is not equal 0',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim )),
            op_args=(make_arg(shape_with_nc(C_out, C_in, feature_map+1, conv_dim)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='The convolution kernel should smaller than the feature map',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim+1)),
            op_args=(make_arg(shape_with_nc(C_out, C_in, kernel_size, conv_dim)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        #conv1d raise RuntimeError
        op_error_type=(ValueError, RuntimeError),
        op_error_info=f'input size must be <= {conv_dim+2}',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim)),
            op_args=(make_arg(shape_with_nc(C_out, C_in, kernel_size, conv_dim+1)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        #conv1d raise ValueError
        op_error_type=(ValueError, RuntimeError),
        op_error_info='convolution kernel dim should equal feature map dim',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim)),
            op_args=(make_arg(shape_with_nc(C_out, C_in, kernel_size, conv_dim - 1)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        # conv1d raise ValueError
        op_error_type=(ValueError, RuntimeError),
        op_error_info='convolution kernel dim should equal feature map dim',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_in, C_in, feature_map, conv_dim)),
            op_args=(make_arg(shape_with_nc(C_out, C_in, kernel_size, conv_dim)),),
            op_kwargs={'padding': padding,
                       'dilation': math.ceil((feature_map + 2 * padding - 1)/(kernel_size - 1))},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='padding and dilation should satisfy the formula: feature_map + 2 * padding >= '
                      'dilation x (kernel_size -1) +1 ',
    )

# op error sense for mint.nn.functional.linear operators
def error_inputs_other_op_mint_nn_functional_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((3,)),
            op_args=(make_arg((3, 4)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='weight shape does not match',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((3,)),
            op_args=(make_arg((4, 3)),),
            op_kwargs={'bias': make_arg((3,))},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='bias shape does not match',
    )

# the error inputs scene of op with bce loss
def error_inputs_other_op_with_loss_function(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    args = inspect.signature(op_info.op).parameters.keys()
    # bce loss
    if 'weight' in args:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((3, 4)),
                op_args=(make_arg((4,)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='target shape should equal input shape',
        )
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((3, 4)),
                op_args=(make_arg((3, 4)),),
                op_kwargs={'weight': make_arg((5,)), },
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='the weight shape should equal  or can be broadcast to input/target shape',
        )
    elif 'pos_weight' in args:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((3, 4)),
                op_args=(make_arg((3, 4)),),
                op_kwargs={'pos_weight': make_arg((5,)), },
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='the pos_weight shape should equal  or can be broadcast to input/target shape',
        )
        # l1 or mse loss
    else:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((3, 4)),
                op_args=(make_arg((5,)),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='the target shape should equal  or can be broadcast to input shape',
        )

def error_inputs_other_op_mint_nn_functional_batch_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(make_arg((4,)), make_arg((4,)), make_arg((4,)), make_arg((4,)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='running_mean/running_var/weight/bias shape should be (C,)',
    )


def error_inputs_other_op_group_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(3, 6,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='num_channels should equal ''C'' ',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(2, 3,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='num_groups should be divided with no remainder by  num_channels',
    )

def error_inputs_other_op_layer_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=((3,),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='normalizd_shape should equal input_shape[begin_norm_axis:]',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(3,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='''if normalizd_shape type is int,it should be equal input's last dim ''',
    )

def error_inputs_other_op_with_AdaptiveAvgPool(op_info: OpInfo, dtype=None, device=None, **kwargs):
    pool_dim = kwargs.get('pool_dim')

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    rank = {'Pool1d': 1, 'Pool2d': 2, 'Pool3d': 3,
            'functional_Pool1d': 1, 'functional_Pool2d': 2, 'functional_Pool3d': 3}[pool_dim]

    def shape_with_nc(side: int, rank: int):
        return tuple([side] * rank)

    N_correct = 2
    N_error = 0

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_correct, rank)),
            op_args=(shape_with_nc(N_correct, rank),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'''the input's dim should be {rank + 1}d or {rank + 2}d ''',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_correct, rank + 3)),
            op_args=(shape_with_nc(N_correct, rank),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'''the input's dim should be {rank + 1}d or {rank + 2}d ''',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_correct, rank + 1)),
            op_args=(shape_with_nc(N_error, rank),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the value of output_size is not a positive integer ',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(N_correct, rank + 1)),
            op_args=(shape_with_nc(N_correct, rank - 1),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, IndexError),
        op_error_info=f'the length of output_size must be equal to {rank} ',
    )
    if pool_dim != 'Pool2d':
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape_with_nc(N_correct, rank + 1)),
                op_args=(shape_with_nc(N_correct, rank + 1),),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'the length of output_size must be equal to {rank} ',
        )

def error_inputs_other_op_with_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(2, ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=''' 'p' must be in [0,1]  ''',
    )

def basic_sample_inputs_adaptivegpool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as (input shape, output size)
    cases = (
        ((5, 9), 1),
        ((5, 9), 3),
        ((5, 8, 10), 7),
        ((5, 8, 10), 4),
        ((5, 8, 10), 10),
        ((5, 8, 7), 12),
    )

    for input_shape, output_size in cases:
        # Batched
        yield OpSampleInput(make_arg(input_shape), op_args=(output_size,),
                            op_kwargs={}, sample_name=f"{op_info.name}_{dtype}")

def extra_sample_inputs_adaptivegpool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    #big shape
    yield OpSampleInput(
        op_input=make_arg((1024, 817)),
        op_args=(4,),
        op_kwargs={}, sample_name=f"{op_info.name}_{dtype}_big_shape")
    #for large/small values
    float_vals = dtypes_extra_values['_float_small_values']
    if dtype==ms.float16:
        float_vals = float_vals + dtypes_extra_values['_fp16_large_values']
    else:
        float_vals = float_vals + dtypes_extra_values['_fp32_large_values']
    yield OpSampleInput(
            op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8)), dtype=dtype),
            op_args=(4,),
            op_kwargs={}, sample_name=f"{op_info.name}_{dtype}_large_small_values")

    #for inf/nan values
    for t in [np.inf, np.nan, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((7, 7, 8), t), dtype=dtype),
            op_args=(3,),
            op_kwargs={}, sample_name=f"{op_info.name}_{dtype}_")

    #for discontiguous values
    for discontiguous in [False, True]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8), discontiguous=discontiguous),
            op_args=(3,),
            op_kwargs={}, sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}")

def error_sample_inputs_adaptivegpool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    #input not tensor
    yield OpErrorInput(
        OpSampleInput(
            op_input=np.random.uniform(0,10, (3, 4)),
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_not_tensor",
        ),
        op_error_type=TypeError,
        op_error_info="input is not tensor",
    )
    # empty shape
    yield OpErrorInput(
        OpSampleInput(
            op_input=make_tensor(shape=(0, 3), dtype=ms.float32, device=device),
            op_args=(1, ),
            op_kwargs={},
            sample_name=f"{op_info.name}_empty",
        ),
        op_error_type=ValueError,
        op_error_info="input is empty shape",
    )
    #outputsize is zero
    yield OpErrorInput(
        OpSampleInput(
            op_input=make_tensor(shape=(4, 3), dtype=ms.float32, device=device),
            op_args=(0,),
            op_kwargs={},
            sample_name=f"{op_info.name}_output_zero",
        ),
        op_error_type=RuntimeError,
        op_error_info="outputsize is zero",
    )

def error_inputs_other_op_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    mode = kwargs.get("mode")
    internal_mode, rank = _normalize_mode_and_ranks(mode)
    rank = rank[0]
    N, C = 1, 1
    D = 2
    size = 3
    scale_factor = 1.7

    def shape_with_nc(side: int, rank: int):
        return tuple([N, C] + [side] * rank)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, rank)),
            op_args=(),
            op_kwargs={'scale_factor': tuple([scale_factor] * (rank + 1)), 'mode': internal_mode},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='if scale_factor is a tuple or list,its length must be the same as the dimension of the input'
                      'tensor with the N and C dimensions removed.',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, rank)),
            op_args=(),
            op_kwargs={'size': tuple([size] * (rank + 1)), 'mode': internal_mode},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='if size is a tuple or list,its length must be the same as the dimension of the input'
                      'tensor with the N and C dimensions removed.',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, rank)),
            op_args=(),
            op_kwargs={'size': tuple([float(size)] * rank), 'mode': internal_mode},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='the data type of size mast be int',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, rank)),
            op_args=(),
            op_kwargs={'size': tuple([size] * rank),
                       'scale_factor': tuple([scale_factor] * rank), 'mode': internal_mode},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=''' 'size' and 'scale_factor' can not be set simultaneously.''',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, rank)),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=''' 'size' and 'scale_factor' can not be both empty.''',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, 4 if internal_mode == 'nearest' else rank + 1)),
            op_args=(),
            op_kwargs={'size': size, 'mode': internal_mode},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.dim not match mode. ',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, 0 if internal_mode == 'nearest' else rank - 1)),
            op_args=(),
            op_kwargs={'size': size, 'mode': internal_mode},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.dim not match mode. ',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape_with_nc(D, rank)),
            op_args=(),
            op_kwargs={'size': size, 'recompute_scale_factor': True},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=' when set recompute_scale_factor to true,size should be None',
    )
    if internal_mode == 'nearest':
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={'size': size, 'align_corners': True, 'mode': internal_mode},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=' align_corners not support mode nearest',
        )

def error_inputs_other_op_pad(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    mode = kwargs.get("mode")
    input_shape_length_not_support_convert_to_pad_shape_length={2: [2, 3], 3: [3], 4: [1], 5: [2, 1]}
    input_shape_length_convert_to_pad_shape_length = {2: [1], 3: [1,2], 4: [2,3], 5: [3]}
    for shape_length in (2, 3, 4, 5):
        input_shape = tuple(np.random.randint(2, 10, size=shape_length))
        if mode in ['reflect', 'replicate']:
            for pad_shape_lenght in input_shape_length_not_support_convert_to_pad_shape_length[shape_length]:
                pad_shape = tuple(np.ones(pad_shape_lenght, int))
                yield OpErrorInput(
                    op_sample_input=OpSampleInput(
                        op_input=make_arg(input_shape),
                        op_args=(pad_shape*2,),
                        op_kwargs={'mode': mode},
                        sample_name=op_info.name,
                    ),
                    op_error_type=ValueError,
                    op_error_info='pad shape does not match input shape',
                )
        if mode in ['reflect']:
            for pad_shape_lenght in input_shape_length_convert_to_pad_shape_length[shape_length]:
                pad_shape = sum( ((x, x) for x in input_shape[-pad_shape_lenght:][::-1]), ())
                yield OpErrorInput(
                    op_sample_input=OpSampleInput(
                        op_input=make_arg(input_shape),
                        op_args=(pad_shape,),
                        op_kwargs={'mode': mode},
                        sample_name=op_info.name,
                    ),
                    op_error_type=RuntimeError,
                    op_error_info='padding size should be less than the corresponding self dimension',
                )
        if mode != 'circular':
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_arg(input_shape),
                    op_args=((1,1,1,1,1,1,1),),
                    op_kwargs={'mode': mode},
                    sample_name=op_info.name,
                ),
                op_error_type=ValueError,
                op_error_info='pad shape length should no more than 6',
            )
        else:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_arg(input_shape),
                    op_args=((1,)*(len(input_shape)*2+1),),
                    op_kwargs={'mode': mode},
                    sample_name=op_info.name,
                ),
                op_error_type=ValueError,
                op_error_info='pad shape length should no more than input shape *2',
            )
def error_inputs_other_op_with_activation(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=((3,)),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the dim value out range',
    )

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

def error_inputs_other_op_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = ((1.0, -3), (-3, 1.0))
    for dim, index in cases:
        yield OpErrorInput(
            OpSampleInput(
                op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
                op_args=(dim, index),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=TypeError,
            op_error_info="dim or index error",
        )


def error_inputs_other_op_split(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = ((1.0, -1), (19, 1), (1 - 1, -1), ([1, 0, 0, 0], -1),)
    for split_size, dim in cases:
        yield OpErrorInput(
            OpSampleInput(
                op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
                op_kwargs={'split_size': split_size, 'dim': dim},
                sample_name=op_info.name,
            ),
            op_error_type=TypeError,
            op_error_info="split_size or dim error",
        )


def error_inputs_other_op_tile(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield OpErrorInput(
        OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
            op_args=(),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info="dims error",
    )

    yield OpErrorInput(
        OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
            op_args=(1.0,),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info="dims error",
    )


def error_inputs_other_op_topk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield OpErrorInput(
        OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
            op_args=('1', 0, True, True),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info="k error",
    )

    yield OpErrorInput(
        OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
            op_args=(1, 0, True, 2),
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info="sorted error",
    )


def error_inputs_other_op_tensor_transpose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (('1', 1), (0, '0'), ([1], 1), (1, -3), (-3, 0))
    for args1, args2 in cases:
        yield OpErrorInput(
            OpSampleInput(
                op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
                op_args=(args1, args2),
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info="dim error",
        )


def error_inputs_other_op_tensor_unique(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = ((1.0, True, False, 0),
             (True, 1.0, False, 0),
             (True, False, 1.0, 0),
             (True, True, False, 1.0),
             )
    for args1, args2, args3, args4 in cases:
        yield OpErrorInput(
            OpSampleInput(
                op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
                op_args=(args1, args2, args3, args4),
                sample_name=op_info.name,
            ),
            op_error_type=TypeError,
            op_error_info="dim error",
        )


def error_inputs_other_op_tensor_unsqueeze(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (1.0, -4)
    for args in cases:
        yield OpErrorInput(
            OpSampleInput(
                op_input=make_tensor(shape=(2, 3), dtype=ms.float32, device=device),
                op_args=(args,),
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info="dim error",
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


def extra_sample_inputs_tensor_unique(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)
    extremal_cases = [
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ]

    for extremal_val, name_suffix in extremal_cases:
        t = make_x((5,))
        t[2] = extremal_val
        yield OpSampleInput(
            op_input=t,
            op_args=(True, True, True, 0),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}')


def extra_sample_inputs_tensor_split(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )

    # Single element tensors
    yield OpSampleInput(
        op_input=make_func((1,)),
        op_args=(1,),
        op_kwargs={'dim': 0},
        sample_name=f'{op_info.name}_{dtype}_single_element_1d_default'
    )

    # Duplicate values test - numerical stability
    t = make_func((4, 4))
    yield OpSampleInput(
        op_input=t,
        op_args=(1,),
        op_kwargs={'dim': 0},
        sample_name=f'{op_info.name}_{dtype}_duplicate_values_default'
    )

    # Extremal values test - only for floating point types
    if dtype.is_floating_point:
        # inf and nan is unsupported on Ascend910 devices.
        skip_inf_nan = False
        if device == 'ascend':
            ascend_name = MSContext.get_instance().get_ascend_soc_version()
            if ascend_name == 'ascend910':
                warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
                skip_inf_nan = True

        # Test with different extremal values
        extremal_cases = [
            (0.0, 'zero'),
            (1.0, 'one'),
        ]

        # Only add inf and nan cases if not on Ascend910
        if not skip_inf_nan:
            extremal_cases.extend([
                (float('nan'), 'nan'),
                (float('inf'), 'inf'),
                (float('-inf'), 'neg_inf'),
            ])

        for extremal_val, name_suffix in extremal_cases:
            t = make_func((5,))
            t[2] = extremal_val
            yield OpSampleInput(
                op_input=t,
                op_args=(1,),
                op_kwargs={'dim': 0},
                sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
            )

    # Large input tests - numerical stability
    LARGE_1D_SIZE = 2 ** 8
    yield OpSampleInput(
        op_input=make_func((LARGE_1D_SIZE,)),
        op_args=(1,),
        op_kwargs={'dim': 0},
        sample_name=f'{op_info.name}_{dtype}_default_large_1d_stability'
    )


def extra_sample_inputs_tile(op_info, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if dtype.is_floating_point:
        # inf and nan is unsupported on Ascend910 devices.
        skip_inf_nan = False
        if device == 'ascend':
            ascend_name = MSContext.get_instance().get_ascend_soc_version()
            if ascend_name == 'ascend910':
                warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
                skip_inf_nan = True

        # Test with different extremal values
        extremal_cases = [
            (0.0, 'zero'),
            (1.0, 'one'),
        ]

        # Only add inf and nan cases if not on Ascend910
        if not skip_inf_nan:
            extremal_cases.extend([
                (float('nan'), 'nan'),
                (float('inf'), 'inf'),
                (float('-inf'), 'neg_inf'),
            ])

        for extremal_val, name_suffix in extremal_cases:
            t = make_func((5,))
            t[2] = extremal_val
            yield OpSampleInput(
                op_input=t,
                op_args=((1, 1),),
                sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
            )


def extra_sample_inputs_tensor_unsqueeze(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    extremal_cases = [
        (0.0, 'zero'),
        (1.0, 'one'),
    ]
    extremal_cases.extend([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])

    for extremal_val, name_suffix in extremal_cases:
        t = make_func((5,))
        t[2] = extremal_val
        yield OpSampleInput(
            op_input=t,
            op_args=(0,),
            sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
        )

    # Large input tests - numerical stability
    LARGE_1D_SIZE = 2 ** 10
    yield OpSampleInput(
        op_input=make_func((LARGE_1D_SIZE,)),
        op_args=(0,),
        sample_name=f'{op_info.name}_{dtype}_default_large_1d_stability'
    )


def extra_sample_inputs_tensor_transpose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    extremal_cases = [
        (0.0, 'zero'),
        (1.0, 'one'),
    ]
    extremal_cases.extend([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])

    for extremal_val, name_suffix in extremal_cases:
        t = make_func((5,))
        t[2] = extremal_val
        yield OpSampleInput(
            op_input=t,
            op_args=(-1, 0),
            sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
        )

    # Large input tests - numerical stability
    LARGE_1D_SIZE = 2 ** 8
    yield OpSampleInput(
        op_input=make_func((LARGE_1D_SIZE,)),
        op_args=(-1, 0),
        sample_name=f'{op_info.name}_{dtype}_default_large_1d_stability'
    )

def extra_sample_inputs_tensor_topk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    extremal_cases = [
     (float('nan'), 'nan'),
     (float('inf'), 'inf'),
     (float('-inf'), 'neg_inf'),
    ]
    if dtype is ms.float32:
        for extremal_val, name_suffix in extremal_cases:
            t = make_func((5,))
            t[2] = extremal_val
            yield OpSampleInput(
                op_input=t,
                op_args=(1, 0, True, True),
                sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
            )

def extra_reference_inputs_op_functional_pool_func( op_info: OpInfo, dtype, device=None, **kwargs ):
    """
    Generate additional edge-case test inputs for other operators,
    mint.nn.functional.adaptive_avg_pool2d, mint.nn.functional.adaptive_avg_pool3d,
    mint.nn.functional.avg_pool2d, mint.nn.functional.max_pool2d.
    Args:
        op_info (OpInfo): object for the other operator.
        dtype (type): Data type of the tensors.
        device (str, optional): Device to create tensors on, Default: ``None``.
        **kwargs (dict, optional): Additional keyword arguments.

    Yields:
        OpSampleInput, Sample input configurations for edge cases.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-9, high=0)

    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values), 1)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1,),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_small_value',
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values), 1)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1,),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_large_value',
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 3, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(1,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (1, 3, 8, 8)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(3,),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
        )


def extra_reference_inputs_tensor_clamp(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate additional edge-case test inputs for Tensor.clamp.
    Args:
        op_info (OpInfo): object for the other operator.
        dtype (type): Data type of the tensors.
        device (str, optional): Device to create tensors on, Default: ``None``.
        **kwargs (dict, optional): Additional keyword arguments.

    Yields:
        OpSampleInput, Sample input configurations for edge cases.
    """
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values), 1)
        min_values = np.array([0.0])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(),
            op_kwargs={'min': make_tensor_with_np_array(min_values, dtype=dtype, device=device)},
            sample_name=op_info.name,
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values), 1)
        min_values = np.array([1.0])
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(),
            op_kwargs={'min': make_tensor_with_np_array(min_values, dtype=dtype, device=device)},
            sample_name=op_info.name,
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 3, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(),
                    op_kwargs={'min': 0},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )
    # empty tensor, min and max are tensors with same shape as input
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    empty_tensor = make_arg(shape=(2, 0))
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(empty_tensor, empty_tensor),
        op_kwargs={},
        sample_name=f'{op_info.name}_{dtype}_empty_tensor',
    )

def extra_reference_inputs_tensor_expand_as(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate additional edge-case test inputs for Tensor.expand_as.
    Args:
        op_info (OpInfo): object for the other operator.
        dtype (type): Data type of the tensors.
        device (str, optional): Device to create tensors on, Default: ``None``.
        **kwargs (dict, optional): Additional keyword arguments.

    Yields:
        OpSampleInput, Sample input configurations for edge cases.
    """
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values), 1)
        input_x = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        yield OpSampleInput(
            op_input=input_x,
            op_args=(input_x,),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_small_values',
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, 1, len(_large_values), 1)
        input_x = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
        yield OpSampleInput(
            op_input=input_x,
            op_args=(input_x,),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_large_values',
        )
        # ---- extremal value tensors (inf, -inf, nan) ----
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((4, 4, 9, 3), t)
                other_values = np.ones((4, 4, 9, 3))
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(other_values, dtype=dtype, device=device),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )

def basic_sample_inputs_mint_normal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    samples = (
        ((), ()),
        ((S,), (S,)),
        ((S, M, L), (S, M, L)),
        ((5, 3, 4, 2), (5, 3, 4, 2)),
    )
    for mean_shape, std_shape in samples:
        yield OpSampleInput(make_arg(mean_shape, random_method='randn'),
                            op_args=(make_arg(std_shape, high=100, low=0, random_method='uniform'), ))

    yield OpSampleInput(-2.0, op_args=(make_arg((4, ), high=100, low=0, random_method='uniform'), ))

    std_values = [0, 1, 1.0, 12.5, 312.1199124773063]
    for std_vaule in std_values:
        yield OpSampleInput(make_arg((9, ), random_method='randn'), op_args=(std_vaule, ))

    std = 929.6270736417483
    sizes = ((2, 5, 13), (50, 42), (), (1,), (9, 6, 8, 7, 8, 9, 4),)
    for size in sizes:
        yield OpSampleInput(-2.0, op_args=(std, size))

def extra_sample_inputs_mint_normal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    #inf and nan are unsupported on Ascend910 devices.
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")

    # test inf nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(make_tensor_with_np_array(np.full((S, S), t), dtype=dtype, device=device),
                            op_args=(2,),
                            sample_name=f"{op_info.name}_{dtype}_{t}_mean_2d")
    yield OpSampleInput(-2.0,
                        op_args=(make_tensor_with_np_array(np.full((S, S), np.inf), dtype=dtype, device=device),),
                        sample_name=f"{op_info.name}_{dtype}_inf_2d")

    # test large shape
    large_shapes =  (((S,S,S,S,S,S,S,S), (S,S,S,S,S,S,S,S)),)
    for mean_shape, std_shape in large_shapes:
        yield OpSampleInput(make_arg(mean_shape, random_method='randn'),
                            op_args=(make_arg(std_shape, high=100, low=0, random_method='uniform'), ))

    # test discontiguous tensor
    for discontiguous in (True,):
        input_shape = (1,)
        weight_shape = (3, 1)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(make_arg(weight_shape, high=100, low=0, discontiguous=discontiguous),),
            sample_name=f"{op_info.name}_discontiguous",
        )

    # test std very large and small vaulue
    values =  (10000.0, 1e-6)
    for value in values:
        yield OpSampleInput(make_arg((2, 3), random_method='randn'),
                            op_args=(value, ),
                            sample_name=f"{op_info.name}_std_large_small_value",)

def extra_reference_inputs_tensor_masked_scatter(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate additional edge-case test inputs for Tensor.masked_scatter.
    Args:
        op_info (OpInfo): object for the other operator.
        dtype (type): Data type of the tensors.
        device (str, optional): Device to create tensors on, Default: ``None``.
        **kwargs (dict, optional): Additional keyword arguments.

    Yields:
        OpSampleInput, Sample input configurations for edge cases.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, 1, len(_float_small_values), 1)
        mask = ms.Tensor(np.random.randint(0, 2, (1, 1, 1, 1)), dtype=ms.bool_)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(mask, make_tensor_with_np_array(input_values, dtype=dtype, device=device),),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_small_value',
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values), 1)
        mask = ms.Tensor(np.random.randint(0, 2, (1, 1, 1)), dtype=ms.bool_)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(mask, make_tensor_with_np_array(input_values, dtype=dtype, device=device),),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_large_value',
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((5, 3, 8, 3), t)
                input_x = make_tensor_with_np_array(input_values, dtype=dtype, device=device)
                mask = ms.Tensor(np.random.randint(0, 2, (5, 1, 8, 3)), dtype=ms.bool_)
                yield OpSampleInput(
                    op_input=input_x,
                    op_args=(mask, input_x),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_x = make_arg(shape=(3, 9), discontiguous=discontiguous)
        mask = ms.Tensor(np.random.randint(0, 2, (3, 1)), dtype=ms.bool_)
        yield OpSampleInput(
            op_input=input_x,
            op_args=(mask, input_x),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
        )

    # empty tensor
    empty_tensor = make_arg((2, 0))
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(make_arg(shape=(2, 0), dtype=ms.bool_), empty_tensor),
        op_kwargs={},
        sample_name=op_info.name,
    )


def extra_reference_inputs_tensor_masked_scatter_(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate extra inputs for Tensor.masked_scatter_.
    """
    # Reuse the same sample inputs as masked_scatter
    yield from extra_reference_inputs_tensor_masked_scatter(op_info, dtype, device, **kwargs)


def dynamic_sample_inputs_mint_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic-shape/rank inputs for mint.nn.functional.one_hot.

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
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(),
                op_kwargs={'num_classes': 51},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(50, 47)),
                    op_args=(),
                    op_kwargs={'num_classes': 51},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(43, 46)),
                    op_args=(),
                    op_kwargs={'num_classes': 52},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={'num_classes': 100},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(50, 5)),
                    op_args=(),
                    op_kwargs={'num_classes': 100},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(40,)),
                    op_args=(),
                    op_kwargs={'num_classes': 51},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={'num_classes': 100},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(50, 21, 6, 25, 50)),
                    op_args=(),
                    op_kwargs={'num_classes': 100},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 1)),
                    op_args=(),
                    op_kwargs={'num_classes': 51},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def dynamic_sample_inputs_tensor_index_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic sample inputs for Tensor.index_select.
    Consider both dynamic_shape and dynamic_rank.
    Notes per infer logic (gather_d.cc):
      - dim should be a scalar constant.
      - index must be 1D.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 5D input, dim=1
        dim = 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),
                op_args=(dim, ms.Tensor(shape=(None,), dtype=ms.int64)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_5D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 5, 4 ,4, 9)),
                    op_args=(dim, make_tensor(shape=(9,), dtype=ms.int64, device=device, low=0, high=4)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_5D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 8 ,9, 9)),
                    op_args=(dim, make_tensor(shape=(7,), dtype=ms.int64, device=device, low=0, high=9)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_5D",
                ),
            ),
        )

        # Dynamic shape case 2: 8D input, dim=0 (static), other axes dynamic only on input
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None, None, None, None), dtype=dtype),
                op_args=(dim, ms.Tensor(shape=(None,), dtype=ms.int32)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_8D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4, 7, 8, 9, 3, 7, 4)),
                    op_args=(dim, make_tensor(shape=(5,), dtype=ms.int32, device=device, low=0, high=3)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_8D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 7, 7, 7, 9, 7, 8, 9)),
                    op_args=(dim, make_tensor(shape=(4,), dtype=ms.int32, device=device, low=0, high=7)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_8D",
                ),
            ),
        )


def dynamic_sample_inputs_tensor_clamp(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic sample inputs for Tensor.clamp.
    Consider both dynamic_shape and dynamic_rank.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape
        min = 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(min,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_2D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(min,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_2D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4 ,5)),
                    op_args=(min,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_2D",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        min = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(min,),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(min,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input_1'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4)),
                    op_args=(min,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input_2'
                ),
            )
        )


def dynamic_sample_inputs_tensor_masked_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic sample inputs for Tensor.masked_scatter.
    Consider both dynamic_shape and dynamic_rank.
    Notes per infer logic (gather_d.cc):
      - dim should be a scalar constant.
      - index must be 1D.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=ms.bool_), ms.Tensor(shape=(None, None), dtype=dtype)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_2D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 5)),
                    op_args=(ms.Tensor(np.random.randint(0, 2, (1, 5)), dtype=ms.bool_),
                             make_func(shape=(9, 5))),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_2D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8 ,5)),
                    op_args=(ms.Tensor(np.random.randint(0, 2, (8, 5)), dtype=ms.bool_),
                             make_func(shape=(8, 5))),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_2D",
                ),
            ),
        )

        # Dynamic shape case 2: 7D input
        mask = ms.Tensor(shape=(None, None, None, None, None, None, None), dtype=ms.bool_)
        input_x = ms.Tensor(shape=(None, None, None, None, None, None, None), dtype=dtype)
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=input_x,
                op_args=(mask, input_x),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_7D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 7, 8, 8, 4, 5, 3)),
                    op_args=(
                        ms.Tensor(np.random.randint(0, 2, (5, 7, 8, 8, 1, 5, 3)), dtype=ms.bool_),
                        make_func(shape=(5, 7, 8, 8, 4, 5, 3))
                    ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_7D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 8, 3, 4, 7, 7)),
                    op_args=(
                        ms.Tensor(np.random.randint(0, 2, (3, 5, 8, 3, 4, 1, 7)), dtype=ms.bool_),
                        make_func(shape=(3, 5, 8, 3, 4, 7, 7))
                    ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_7D",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=ms.bool_), ms.Tensor(shape=None, dtype=dtype)
                ),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 5)),
                    op_args=(ms.Tensor(np.random.randint(0, 2, (9, 1)), dtype=ms.bool_),
                             make_func(shape=(9, 5))),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input_1'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(ms.Tensor(np.random.randint(0, 2, (1,)), dtype=ms.bool_),
                             make_func(shape=(3,))),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input_2'
                ),
            )
        )


def dynamic_sample_inputs_tensor_masked_scatter_(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate dynamic sample inputs for Tensor.masked_scatter_.
    """
    # Reuse the same sample inputs as masked_scatter
    yield from dynamic_sample_inputs_tensor_masked_scatter(op_info, dtype, device, **kwargs)

def error_inputs_other_op_adaptive_avg_pool2d(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.functional.adaptive_avg_pool2d.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((1, 7, 3, 8), (1.0)),
        ((5, 7, 9), (3,)),
        ((5, 7), (3, 4)),
        ((1, 8, 4, 3), -1),
        ((1, 7, 0, 8), (1)),
    )
    # output_size is float
    # the length of output_size isnot 2
    # input isnot 3D or 4D
    # output_size is out of range
    # empty tensor
    for input_shape, output_size in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(output_size,),
                op_kwargs={},
                sample_name=f"{op_info.name}_output_size_value_error",
            ),
            op_error_type=(TypeError, ValueError, IndexError, RuntimeError),
            op_error_info=f'output_size error values is not supported for {op_info.name}',
        )


def error_inputs_other_op_adaptive_avg_pool3d(op_info: OpInfo, dtype, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.functional.adaptive_avg_pool3d.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((1, 7, 3, 8), (1.0)),
        ((1, 8, 4, 3, 7), (1, 1, 1, 1)),
        ((1, 8, 4, 3), (0, -2, -3)),
        ((1, 8, 0, 3, 7), (1, 1, 1)),
    )
    # input isnot 4D or 5D
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 7, 3)),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_error_input_3D",
        ),
        op_error_type=TypeError,
        op_error_info=f'input error value is not supported by {op_info.name}.',
    )
    # output_size is float
    # the length of output_size isnot 3
    # output_size must >=0
    # empty tensor
    for input_shape, output_size in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(output_size,),
                op_kwargs={},
                sample_name=f"{op_info.name}_output_size_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'output_size error values is not supported for {op_info.name}',
        )


def error_inputs_other_op_avg_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.functional.avg_pool2d.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # Order: input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    cases = (((9, 9, 5, 4), 4.0, 1, 2, False, False, 18),
             ((9, 9, 5, 4), 4, 1.0, 2, False, False, 18),
             ((1, 3, 9, 9), 3, 1, 1.0, True, False, 2),
             ((1, 3, 9, 9), 3, 1, 1, 1, False, 2),
             ((1, 3, 9, 9), (4, 4), (2, 3), 1, True, 1, 2),
             ((9, 9, 5, 4), 4, 1, 2, False, False, 1.8),
             ((9, 9, 5, 4), 0, 1, 2, False, False, 0),
             ((9, 9, 5, 4), 4, 0, 2, False, False, 0),
             ((9, 9, 5, 4), (1, 1, 1), 1, 4, False, False, 0),
             ((9, 9, 5, 4), 4, (1, 1, 1), 1, False, False, 0),
             ((9, 9, 5, 4), 4, 1, (1, 1, 1), False, False, 0),
             ((9, 9, 5, 4), 4, 1, -1, False, False, 0),
             ((1, 3, 9, 0), 3, 1, 1, True, False, 2))
    # input isnot 3D or 4D
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 7)),
            op_args=(4, 1, 2, False, False, 18),
            op_kwargs={},
            sample_name=f"{op_info.name}_error_input_2D",
        ),
        op_error_type=ValueError,
        op_error_info=f'input error value is not supported by {op_info.name}.',
    )
    # kernel_size is float
    # stride is float
    # padding is float
    # ceil_mode is float
    # count_include_pad is float
    # divisor_override is float
    # kernel_size is less than 1
    # stride is less than 1
    # the length of kernel_size isnot 1 or 2
    # the length of stride isnot 1 or 2
    # the length of padding isnot 1 or 2
    # padding must >=0
    # empty tensor
    for input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )


def error_inputs_other_op_max_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.functional.max_pool2d.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # Order: input_shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices
    cases = (((9, 8, 4, 3), 2.0, 1, 0, 1, False, False),
             ((9, 8, 4, 3), 2, 1.0, 0, 1, False, False),
             ((9, 8, 4, 3), 2, 1, 0.1, 1, False, False),
             ((9, 8, 4, 3), 2, 1, 0, 1.0, False, False),
             ((9, 8, 4, 3), 2, 1, 0, 1, 1.0, False),
             ((9, 8, 4, 3), 2, 1, 0, 1, False, 1.0),
             ((9, 8, 4, 3), -2, 1, 0, 1, False, False),
             ((9, 8, 4, 3), 2, 0, 0, 1, False, False),
             ((9, 8, 4, 3), 2, 1, 0, 0, False, False),
             ((9, 8, 4, 3), 2, 1, 0, (1, 2), False, False),
             ((9, 8, 4, 3), 2, 1, 2, 1, False, False),
             ((9, 8, 4, 3), 2, 1, -1, 1, False, False),
             ((1, 3, 0, 9), 2, 1, 0, 1, False, False))
    # input isnot 4D
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 7)),
            op_args=(2, 1, 0, 1, False, False),
            op_kwargs={},
            sample_name=f"{op_info.name}_error_input_2D",
        ),
        op_error_type=ValueError,
        op_error_info=f'input error value is not supported by {op_info.name}.',
    )
    # kernel_size is float
    # stride is float
    # padding is float
    # dilation is float
    # ceil_mode is float
    # return_indices is float
    # kernel_size is less than 1
    # stride is less than 1
    # dilation is less than 1
    # dilation isnot all 1
    # padding is larger than kernel_size/2
    # padding must >=0
    # empty tensor
    for input_shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def error_inputs_other_op_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.functional.one_hot.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((2, 2, 3, 2), 1.0),
        ((2, 2, 3, 2), -2),
        ((2, 0), 10),
    )
    # num_classes is float
    # num_classes is less than -1
    # empty tensor
    for input_shape, num_classes in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(),
                op_kwargs={"num_classes": num_classes},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )


def error_inputs_tensor_clamp(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for Tensor.clamp.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # min/max are all None
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((2, 9)),
            op_args=(None, None),
            op_kwargs={},
            sample_name=f"{op_info.name}_params_value_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'params error values is not supported for {op_info.name}',
    )
    # the shape of min is larger than input
    input_min = ms.Tensor(np.random.randint(-500, 500, (4, 9)), dtype=ms.float32)
    input_max = ms.Tensor(np.random.randint(-7, 7, (7,)), dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((7,)),
            op_args=(input_min, input_max),
            op_kwargs={},
            sample_name=f"{op_info.name}_min_value_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'min error values is not supported for {op_info.name}',
    )


def error_inputs_tensor_expand_as(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for Tensor.expand_as.
    """
    cases = (((3, 4, 5), (4, 5)),
             ((3, 4), (4, 3)),
             ((3, 0), ()),
             )
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # other is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((2, 9)),
            op_args=(2,),
            op_kwargs={},
            sample_name=f"{op_info.name}_other_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'params error values is not supported for {op_info.name}',
    )
    # the shape of self and other is incompatible.
    # input empty tensor
    for shape, shape_other in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_input(shape),
                op_args=(make_input(shape_other),),
                op_kwargs={},
                sample_name=f"{op_info.name}_shape_error",
            ),
            op_error_type=ValueError,
            op_error_info=f'incompatible shape is not supported for {op_info.name}',
        )

def error_inputs_tensor_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for Tensor.gather.
    """
    # input_shape, dim, index_shape, index_low, index_high
    cases = (((3,), -2, (3,), 0, 2),
             ((4, 4, 3), -2, (3, 15, 7), 0, 3),
             ((3, 3), -1, (2, 3, 3), -2, 2),
             ((2, 3), -1, (4, 6), -2, 2),
             # got no exception raised.((3, ), -1, (3,), 3, 5),
             )
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    make_index = functools.partial(make_tensor, device=device, dtype=ms.int32)
    # index is not int32 or int64
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((2, 9)),
            op_args=(0, make_index(shape=(0,), dtype=ms.float32)),
            op_kwargs={},
            sample_name=f"{op_info.name}_index_type_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'index error type is not supported for {op_info.name}',
    )
    # dim is out of range
    # index axis not same input or index shape greater input
    # the shape of index is greater than input
    # x shape must be equal to 4, but got 2
    # index is out of range
    for input_shape, dim, index_shape, index_low, index_high in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_input(input_shape),
                op_args=(dim, make_index(shape=index_shape, low=index_low, high=index_high)),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=ValueError,
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def error_inputs_tensor_index_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for Tensor.index_select.
    """
    # input_shape, dim, index_shape, index_low, index_high
    cases = (((3,), -2, (3,), 0, 2),
             ((3,), -1.0, (3,), 0, 2),
             ((9,), -1, (4, 2), 0, 9),
             )
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    make_index = functools.partial(make_tensor, device=device, dtype=ms.int32)
    # index isnot tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((2, 9)),
            op_args=(0, 1),
            op_kwargs={},
            sample_name=f"{op_info.name}_index_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'index error type is not supported for {op_info.name}',
    )
    # index dtype is bool
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((9,)),
            op_args=(-1, make_tensor(shape=(4,), dtype=ms.bool_)),
            op_kwargs={},
            sample_name=f"{op_info.name}_index_type_error",
        ),
        op_error_type=RuntimeError,
        op_error_info=f'index error type is not supported for {op_info.name}',
    )
    # dim is out of range
    # dim isnot int
    # index rank isnot 1
    for input_shape, dim, index_shape, index_low, index_high in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_input(input_shape),
                op_args=(dim, make_index(shape=index_shape, low=index_low, high=index_high)),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )


def error_inputs_tensor_masked_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for Tensor.masked_scatter.
    """
    # input_shape, source_shape,mask_shape, mask_low, mask_high
    cases = (((8,), (8,), (9,), 0, 2),
             ((8,), (5,), (5,), 0, 2),
             ((3, 4), (5,), (2, 5), 0, 2),
             )
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)
    make_mask = functools.partial(make_tensor, device=device, dtype=ms.bool_)
    # mask/source isnot tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((2, 9)),
            op_args=(0, 1),
            op_kwargs={},
            sample_name=f"{op_info.name}_mask_source_type_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'mask/source error type is not supported for {op_info.name}',
    )
    # mask dtype isnot bool
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_input((9,)),
            op_args=(make_tensor(shape=(9,), dtype=ms.float32), make_input((9,))),
            op_kwargs={},
            sample_name=f"{op_info.name}_mask_type_error",
        ),
        op_error_type=(TypeError, RuntimeError),
        op_error_info=f'mask error type is not supported for {op_info.name}',
    )
    # the shape of input is less than mask
    # source rank is less than input
    # mask cannot be broadcast to the input
    for input_shape, source_shape, mask_shape, mask_low, mask_high in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_input(input_shape),
                op_args=(make_mask(shape=mask_shape, low=mask_low, high=mask_high), make_input(source_shape)),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def error_inputs_tensor_masked_scatter_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for Tensor.masked_scatter_.
    """
    # Reuse the same sample inputs as masked_scatter
    yield from error_inputs_tensor_masked_scatter(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_addmm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (
             ((1, 6), (2, 4), (4, 6),),
             ((5, 1), (5, 8), (8, 3),),
             ((15, ), (1, 12), (12, 15))
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
        op_error_type=ValueError,
        op_error_info='the dim of mat1 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((B, C)),
            op_args=(make_((B, )), make_((B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the dim of mat1 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, B)),
            op_args=(make_((A, B)), make_((B, )),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the dim of mat2 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((A, C)),
            op_args=(make_((A, B)), make_((B, C, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='the dim of mat2 must be 2',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_((1, C-1)),
            op_args=(make_((A, B)), make_((B, C)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='the input.shape should be broadcasted by mat1.shape@mat2.shape',
    )

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

def extra_sample_inputs_addmm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (4, 4), ((4, 4),(4, 4)), {},
                                  dtype=dtype, device=device,exclude_value=(-5062757.7, 5062757.7, 1e20,-1e20,))

    yield from make_extra_tensor(op_info, (0, 0), ((0, 4), (4, 0)), {},
                                 dtype=dtype, device=device, empty_tensor=True)

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

def basic_sample_inputs_dot(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case=(
        (3,),
        (10,)
    )
    for shape in case:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(make_arg(shape),),
            op_kwargs={},
            sample_name=op_info.name,
        )


def error_sample_inputs_dot(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_error_tensor_with_matrix_multiply(op_info, 1, 1, device=device)

def extra_sample_inputs_dot(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (16, ), ((16, ),), {},
                                 dtype=dtype, device=device, exclude_value=(1e20,-1e20))

    # yield from make_extra_tensor(op_info, (0,), ((0,),), {},
    #                              dtype=dtype, device=device, empty_tensor=True)


def make_extra_tensor(op_info, op_input_shape, op_args_shapes, op_kwargs, dtype=None, device=None,
                      exclude_value=(), empty_tensor=False, dtype_promotion=False):


    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def _tuple_to_specified_size_tensor(data, size):
        return make_tensor_with_np_array(
            np.random.choice(np.array(data)[~np.isin(np.array(data), exclude_value)], size=size),
            dtype=dtype, device=device)

    def _make_extra_op(data=None, op_input_shape=None, op_args_shapes=None, op_kwargs=None, discontiguous=False):

        def shape_convert_to_tensor(shape, discontiguous=False):
            if not discontiguous:
                return _tuple_to_specified_size_tensor(data, shape)
            return make_arg(shape, discontiguous=True)

        if not discontiguous:
            yield OpSampleInput(
                op_input=shape_convert_to_tensor(op_input_shape),
                op_args=tuple(shape_convert_to_tensor(i) if isinstance(i,tuple) else i for i in op_args_shapes),
                op_kwargs={k: shape_convert_to_tensor(v) if isinstance(v,tuple) else v for k, v in op_kwargs.items()},
                sample_name=op_info.name,
            )
        else:
            yield OpSampleInput(
                op_input=shape_convert_to_tensor(op_input_shape, discontiguous=True),
                op_args=tuple(shape_convert_to_tensor(i, discontiguous=True) if
                              isinstance(i, tuple) else i for i in op_args_shapes),
                op_kwargs={k: shape_convert_to_tensor(v, discontiguous=True) if
                isinstance(v, tuple) else v for k, v in op_kwargs.items()},
                sample_name=op_info.name,
            )

    if empty_tensor:
        yield OpSampleInput(
            op_input=make_arg(op_input_shape),
            op_args=tuple(make_arg(i) if isinstance(i, tuple) else i for i in op_args_shapes),
            op_kwargs={k: make_arg(v) if isinstance(v, tuple) else v for k, v in op_kwargs.items()},
            sample_name=op_info.name + 'empty_tensor',
        )
    else:
        data_list=[]
        if dtype is not None and dtype.is_floating_point:
            _float_small_values = dtypes_extra_values['_float_small_values']
            if dtype == ms.float16:
                _large_values = dtypes_extra_values['_fp16_large_values']
            else:
                _large_values = dtypes_extra_values['_fp32_large_values']

            data_list.extend([_float_small_values, _large_values])

            if not op_info.disable_extremal_value_tensor_inputs:
                if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
                    data_list.extend([(np.inf,), (-np.inf,), (np.nan,)])

        if dtype in dtypes_integral and dtype != ms.bool_:
            if dtype in dtypes_extra_uint or dtype == ms.uint8:
                data_list.extend([dtypes_extra_values['_uint_small_values']])
            else:
                data_list.extend([dtypes_extra_values['_int_small_values']])

        for data in data_list:
            yield  from _make_extra_op(data, op_input_shape, op_args_shapes, op_kwargs )

        # ---- discontiguous tensors ----
        yield from _make_extra_op(op_input_shape=op_input_shape, op_args_shapes=op_args_shapes, op_kwargs=op_kwargs,
                                  discontiguous=True,)

        if dtype_promotion:
            support_dtype = list(op_info.get_supported_dtypes(device))
            support_dtype.remove(dtype)
            for other_dtype in support_dtype:

                yield OpSampleInput(
                    op_input=make_tensor(op_input_shape, dtype=dtype, device=device),
                    op_args=tuple(make_tensor(i, dtype=other_dtype, device=device)
                                  if isinstance(i, tuple) else i for i in op_args_shapes),
                    op_kwargs={k: make_tensor(v, dtype=other_dtype, device=device)
                    if isinstance(v, tuple) else v for k, v in op_kwargs.items()},
                    sample_name=op_info.name + f'{dtype}_promotion_{other_dtype}',
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
                                 dtype=dtype, device=device, exclude_value=(1e+20,-1e+20))

    yield from make_extra_tensor(op_info, (1, 0), ((0, ),), {},
                                 dtype=dtype, device=device, empty_tensor=True)

def basic_sample_inputs_outer(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ( (2, ), (10, )  ),
        ( (3, ), (5, )  ),
        ( (8, ), (2, )  ),
    )

    for input, vec  in case:

        yield OpSampleInput(
            op_input=make_arg(input),
            op_args=(make_arg(vec),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_sample_inputs_outer(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from  make_error_tensor_with_matrix_multiply(op_info, 2, 2, device=device,)

def extra_sample_inputs_outer(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (16, ), ((16, ),), {},
                                 dtype=dtype, device=device, dtype_promotion=True)

    yield from make_extra_tensor(op_info, (0,), ((0,),), {}, dtype=dtype,
                                 device=device, empty_tensor=True)

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

def basic_unary_op_func_tensor_to_1(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    shapes = (
        (),
        (S,),
        (M, S),
        (S, S, L),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                slen = len(input_shape)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_{dtype}_{slen}d",
                )

def extra_unary_op_func_tensor_to_1(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #discontiguous/empty tensors/small medium large size
    shapes = (
        (1,),
        (3, 1),
        (1, 2, 3),
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_shape_{input_shape}",
                )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for extremal_val, name_suffix in extremal_cases:
                t = make_func((5,))
                t[2] = extremal_val
                yield OpSampleInput(
                    op_input=t,
                    op_args=(),
                    op_kwargs={'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
                )


def basic_unary_op_func_tensor_to_2(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    shapes = (
        (),
        (S,),
        (M, S),
        (S, S, L),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                slen = len(input_shape)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'device':'cpu', 'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_{dtype}_{slen}d",
                )

def extra_unary_op_func_tensor_to_2(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #discontiguous/empty tensors/small medium large size
    shapes = (
        (1,),
        (3, 1),
        (1, 2, 3),
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )

    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'device':'cpu', 'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_shape_{input_shape}",
                )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for extremal_val, name_suffix in extremal_cases:
                t = make_func((5,))
                t[2] = extremal_val
                yield OpSampleInput(
                    op_input=t,
                    op_args=(),
                    op_kwargs={'device':'cpu', 'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
                )

def basic_sample_inputs_allclose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_binary_op_common_func(op_info, dtype=dtype, device=device, **kwargs)
    shapes=[(), (5,) , (3, 4, 5)]
    atols=[1e-2, 1e-16]
    rtols=[1e-1, 0.5]
    equal_nans=[True, False]
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)
    for shape, atol, rtol, equal_nan in itertools.product(shapes, atols, rtols, equal_nans):
        yield OpSampleInput(make_input(shape),
                            op_args=(make_input(shape),),
                            op_kwargs={"atol":atol, "rtol":rtol, "equal_nan":equal_nan},
                            sample_name=f"{op_info.name}_{dtype}_equalnan_{equal_nan}")

def extra_sample_inputs_allclose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for sample_input in extra_reference_inputs_binary_op_common_func(op_info, dtype, device):
        atol = random.choice([0, 1, 1e-5, 1e-16, 31415626, np.inf, 0.5])
        rtol = random.choice([0, 1, 1e-5, 1e-16, 31415626, np.inf, 0.5])
        sample_input.op_kwargs=({'atol':atol, 'rtol':rtol})
        if 'nan' in sample_input.sample_name:
            sample_input.op_kwargs.update({"equal_nan":False})
            yield sample_input
            sample_input.op_kwargs.update({"equal_nan":True})
            yield sample_input
        else:
            yield sample_input

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
        ((1029, 917 ), (812, 1029, 20), (812, 20, 917))
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

def basic_sample_inputs_constantpad1d(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    input_shapes=(
        (3, ), (1, 3), (1, 1, 3), (1, 1, 1, 1, 2, 2, 2, 2))
    paddings=(2, (1, 2))
    values=(False, 1.0)
    for input_shape, padding, values in itertools.product(input_shapes, paddings, values):
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(padding, values),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}"
        )

def extra_sample_inputs_constantpad1d(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    #large shape
    yield OpSampleInput(
        op_input=make_arg((1024, 817)),
        op_args=(1, 1.0),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_large_shape"
    )
    #small/large values
    values = _small_large_values(dtype)
    if values:
        yield OpSampleInput(
            op_input=ms.Tensor(np.random.choice(values, size=(2, 3), replace=False), dtype=dtype),
            op_args=(1, 1.0),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_small_large_vales"
        )
    #int/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((2, 3), t), dtype=dtype),
            op_args=(1, 1.0),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

    #discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((2, 3), discontiguous=discontiguous),
            op_args=(1, 1.0),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def error_sample_inputs_constantpad1d(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #input is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=np.array([[2, 3],[3, 4]]),
            op_args=(1, 0),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_not_tensor"
        ),
        op_error_type=TypeError,
        op_error_info="input is not Tensor"
    )
    #padding is small than zero
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((3, 4)),
            op_args=(-1, 0),
            op_kwargs={},
            sample_name=f"{op_info.name}_neg_padding"
        ),
        op_error_type=ValueError,
        op_error_info="padding is small than zero"
    )
    #input is 0-dimensions
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=()),
            op_args=(2, 0),
            op_kwargs={},
            sample_name=f"{op_info.name}_0d"
        ),
        op_error_type=ValueError,
        op_error_info="input has 0-dimensions"
    )

def basic_sample_inputs_mint_multinomial(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.multinomial.
    Covers: basic 1D/2D input shapes, with/without replacement.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)
    cases = (
        ((6,), 5, True),
        ((10,), 1, True),
        ((10,), 5, False),
        ((3, 10), 1, True),
    )

    for input_shape, num_samples, replacement in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(num_samples,),
            op_kwargs={'replacement': replacement},
            sample_name=f"{op_info.name}_{dtype}_shape{input_shape}_num_samples{num_samples}"
        )
    # generator is None
    yield OpSampleInput(
        op_input=make_func((3, 10)),
        op_args=(5,),
        op_kwargs={'replacement': False, 'generator': None},
        sample_name=f'{op_info.name}_{dtype}_generator_none',
    )

def extra_sample_inputs_mint_multinomial(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate extra sample inputs for mint.multinomial.
    Covers: edge cases.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0)
    cases = (
        ((100,), 10, True),
        ((5, 100), 50, True),
        ((5, 100), 50, False),
        ((1, 1000), 10, True),
        ((10, 10), 3, False),
    )

    for input_shape, num_samples, replacement in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(num_samples,),
            op_kwargs={'replacement': replacement},
            sample_name=f"{op_info.name}_{dtype}_shape{input_shape}_num_samples{num_samples}"
        )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (3, 9)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(31,),
            op_kwargs={'replacement': True, 'generator': None},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
        )

def dynamic_sample_inputs_mint_multinomial(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank inputs for mint.multinomial.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(ms.mutable(5),),
                op_kwargs={'replacement': True},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10,)),
                    op_args=(ms.mutable(5),),
                    op_kwargs={'replacement': True},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(20,)),
                    op_args=(ms.mutable(10),),
                    op_kwargs={'replacement': True},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.mutable(5),),
                op_kwargs={'replacement': True},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10,)),
                    op_args=(ms.mutable(5),),
                    op_kwargs={'replacement': True},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 10)),
                    op_args=(ms.mutable(3),),
                    op_kwargs={'replacement': False},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            ),
        )

def error_inputs_mint_multinomial(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.multinomial.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32, low=0)
    # input_shape, num_samples, replacement
    cases = (
        ((8,), 1.0, True),
        ((8,), 0, True),
        ((8, 3), 8, False),
        ((2, 3, 4), 1, False),
        ((), 5, True),
        ((2, 2**25), 2, False),
    )
    # num_samples isnot int
    # num_samples <= 0
    # num_samples is larger than the input shape of last dim, when replacement is False
    # the input.dim > 2
    # the input is 0D
    # the input shape of last dim is larger than 2^24
    for input_shape, num_samples, replacement in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_input(input_shape),
                op_args=(num_samples,),
                op_kwargs={'replacement': replacement},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # neg value/nan
    # maybe cann error
    # RuntimeError: SyncCopy failed
    # this error will affect the execution of subsequent testcases.
    # input_x = (
    #     ms.Tensor([-2.3, 0, 4], ms.float32),
    #     ms.Tensor([np.nan, 0, 4], ms.float32),
    # )
    # not_support_dtypes
    if op_info.not_support_dtypes is not None:
        for input_dtype in op_info.not_support_dtypes:
            sample_input = ms.tensor([1, 0, 6], dtype=input_dtype)
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=sample_input,
                    op_args=(1,),
                    op_kwargs={'replacement': False},
                    sample_name=f"{op_info.name}_{input_dtype}_dtype_error",
                ),
                op_error_type=RuntimeError,
                op_error_info=f'error input dtype {input_dtype} is not supported for {op_info.name}',
            )

def basic_sample_inputs_mint_rand(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.rand.
    """
    # size, generator, dtype, device
    cases = (
        ((), None, dtype, None),
        (7, None, None, None),
        ((2, 0), None, None, None),
        ((2, 3), None, ms.float16, None),
        ([4, 8 ,2], None, None, None),
    )

    for size, generator, case_dtype, case_device in cases:
        if case_dtype in (ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8):
            # for torch check_uniform_bounds not implemented
            yield OpSampleInput(
                op_input=ms.Tensor(shape=size, dtype=dtype),
                op_args=(),
                op_kwargs={'generator': generator, 'dtype': ms.float16, 'device': case_device},
                sample_name=f"{op_info.name}_{generator}_{case_dtype}_to_float16"
            )
        else:
            yield OpSampleInput(
                op_input=ms.Tensor(shape=size, dtype=dtype),
                op_args=(),
                op_kwargs={'generator': generator, 'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_{generator}_{case_dtype}_{case_device}"
            )

def extra_sample_inputs_mint_rand(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.rand.
    Covers: edge cases.
    """
    # size, generator, dtype, device
    cases = (
        (0, None, None, None),
        (100, None, ms.float64, None),
        ((1, 1000), None, None, None),
        ([3, 6, 2, 4], None, ms.float32, None),
        ((3, 3, 2, 1, 1, 4, 5, 6), None, ms.bfloat16, None),
    )

    for size, generator, case_dtype, case_device in cases:
        yield OpSampleInput(
            op_input=ms.Tensor(shape=size, dtype=dtype),
            op_args=(),
            op_kwargs={'generator': generator, 'dtype': case_dtype, 'device': case_device},
            sample_name=f"{op_info.name}_{generator}_{case_dtype}_{case_device}"
        )

def dynamic_sample_inputs_mint_rand(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.rand.
    """
    if kwargs.get("grad_cmp", False):
        print("element 0 of tensor does not require grad and does not have a grad_fn.")
    else:
        if not kwargs.get("only_dynamic_rank", False):
            # Dynamic shape case
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None,), dtype=dtype),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=ms.Tensor(shape=(3,), dtype=dtype),
                        op_args=(),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                    OpSampleInput(
                        op_input=ms.Tensor(shape=(5,), dtype=dtype),
                        op_args=(),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                ),
            )

        if not kwargs.get("only_dynamic_shape", False):
            # Dynamic rank
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=ms.Tensor(shape=(2, 3), dtype=dtype),
                        op_args=(),
                        op_kwargs={},
                        sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                    ),
                    OpSampleInput(
                        op_input=ms.Tensor(shape=(4, 5, 6), dtype=dtype),
                        op_args=(),
                        op_kwargs={},
                        sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                    ),
                )
            )

def error_inputs_mint_rand(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.rand.
    """
    cases = (
        (1.0, None, None, None),
        (10, None, 1.0, None),
        (-1, None, None, None),
        (np.full((2,), np.nan), None, ms.float32, None),
        (1, None, None, 'GPU'),
        # 20260515 the device is CPU, case can pass.
        # (1, None, None, 'CPU'),
    )

    # size is float
    # dtype is float
    # size is negative
    # size is nan
    # device value is GPU
    # device value is CPU
    for size, generator, case_dtype, case_device in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=None,
                op_args=(size,),
                op_kwargs={'generator': generator, 'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def basic_sample_inputs_mint_rand_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.rand_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    # input_shape, dtype, device
    cases = (
        ((), None, None),
        ((7,), ms.bfloat16, None),
        ((2, 0), ms.float32, None),
        ((5, 3), ms.float64, None),
        ((4, 8 ,2), None, None),
    )

    for input_shape, case_dtype, case_device in cases:
        if case_dtype is None and dtype in (ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8):
            # torch op dtype param should be float
            yield OpSampleInput(
                op_input=make_func(input_shape),
                op_args=(),
                op_kwargs={'dtype': ms.float16, 'device': case_device},
                sample_name=f"{op_info.name}_{case_dtype}_{case_device}"
            )
        else:
            yield OpSampleInput(
                op_input=make_func(input_shape),
                op_args=(),
                op_kwargs={'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_{case_dtype}_{case_device}"
            )

def extra_sample_inputs_mint_rand_like(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.rand_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    # input_shape, dtype, device
    cases = (
        ((3, 6, 2, 4, 5), ms.bfloat16, None),
        ((4, 4, 8, 5, 5, 9, 8, 7), ms.float32, None),
    )

    for input_shape, case_dtype, case_device in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(),
            op_kwargs={'dtype': case_dtype, 'device': case_device},
            sample_name=f"{op_info.name}_{case_dtype}_{case_device}"
        )
    # large mix small values
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
            if dtype in (ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8):
                # torch op dtype param should be float
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                    op_kwargs={'dtype': ms.float32},
                    sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
                )
            else:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
                )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_mint_rand_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.rand_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None, None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={'dtype': ms.float64},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 5, 4, 2, 3, 1, 3, 1)),
                    op_args=(),
                    op_kwargs={'dtype': ms.float64},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 2, 3, 2, 2, 3, 3, 1)),
                    op_args=(),
                    op_kwargs={'dtype': ms.float64},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(),
                op_kwargs={'dtype': ms.float32},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 1, 4, 7)),
                    op_args=(),
                    op_kwargs={'dtype': ms.float32},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 3)),
                    op_args=(),
                    op_kwargs={'dtype': ms.float32},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_inputs_mint_rand_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.rand_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((3,), 1.0, None),
        ((1,), None, 'GPU'),
        # 20260515 the device is CPU, case can pass.
        # ((1,), None, 'CPU'),
    )

    # dtype is float
    # device value is GPU
    # device value is CPU
    for input_shape, case_dtype, case_device in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(input_shape),
                op_args=(),
                op_kwargs={'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # case-'the device of input is CPU, device is None' can also pass.
    # input isnot tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1.0,
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'input not tensor is not supported for {op_info.name}',
    )

def basic_sample_inputs_mint_randint(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.randint.
    """
    # low, high, size, generator, dtype, device
    cases = (
        (0, 1, (3,), None, dtype, None),
        (-3, 100, [25], None, ms.int8, None),
        (0, 2, (2, 0), None, None, None),
        (-38, 1, (), None, ms.float16, None),
        (-28, -3, (5, 12), None, None, None),
    )

    for low, high, size, generator, case_dtype, case_device in cases:
        if case_dtype in (ms.complex64, ms.complex128):
            # for torch check_random_bounds handles only integral, floating-point and bool types
            yield OpSampleInput(
                op_input=ms.Tensor(shape=size, dtype=dtype),
                op_args=(low, high),
                op_kwargs={'generator': generator, 'dtype': ms.int64, 'device': case_device},
                sample_name=f"{op_info.name}_{generator}_{case_dtype}_to_int64"
            )
        else:
            yield OpSampleInput(
                op_input=ms.Tensor(shape=size, dtype=dtype),
                op_args=(low, high),
                op_kwargs={'generator': generator, 'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_{generator}_{case_dtype}_{case_device}"
            )

def extra_sample_inputs_mint_randint(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.randint.
    """
    # low, high, size, generator, dtype, device
    cases = (
        (0, 35, [12, 24, 21], None, None, None),
        (-100, -3, (2, 3, 7, 23, 12, 5), None, ms.float64, None),
        (0, 6, [], None, ms.uint8, None),
        (6, 36, (12, 12, 3, 5, 3), None, ms.float32, None),
        (-38, 1, (3, 3, 2, 1, 1, 4, 5, 6), None, ms.bfloat16, None),
    )

    for low, high, size, generator, case_dtype, case_device in cases:
        yield OpSampleInput(
            op_input=ms.Tensor(shape=size, dtype=dtype),
            op_args=(low, high),
            op_kwargs={'generator': generator, 'dtype': case_dtype, 'device': case_device},
            sample_name=f"{op_info.name}_{generator}_{case_dtype}_{case_device}"
        )

def dynamic_sample_inputs_mint_randint(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.randint.
    """
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.mutable(0), ms.mutable(6)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=ms.Tensor(shape=(5, 4, 7), dtype=dtype),
                    op_args=(ms.mutable(-1), ms.mutable(20)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=ms.Tensor(shape=(58, 49, 72), dtype=dtype),
                    op_args=(ms.mutable(6), ms.mutable(39)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(ms.mutable(-1), ms.mutable(12)),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=ms.Tensor(shape=(12, 17), dtype=dtype),
                    op_args=(ms.mutable(-1), ms.mutable(19)),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=ms.Tensor(shape=(15, 6), dtype=dtype),
                    op_args=(ms.mutable(3), ms.mutable(52)),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_inputs_mint_randint(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.randint.
    """
    # low, high, size, generator, dtype, device
    cases = (
        (1.0, 2, (2, 3), None, None, None),
        (0, 1.0, (2, 3), None, None, None),
        (10, 2, (5, 3), None, None, None),
        (10, 20, (2, 3), None, 1.0, None),
        (10, 20, 1, None, None, None),
        (0, 10, (-2, 1), None, None, None),
        (1, 2, (2, 3), None, None, 'GPU'),
        # 20260515 the device is CPU, case can pass.
        # (1, 2, (2, 3), None, None, 'CPU'),
    )

    # low is float
    # high is float
    # low is larger than high
    # dtype is float
    # size is int
    # size is neg value
    # device value is GPU
    # device value is CPU
    for low, high, size, generator, case_dtype, case_device in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=None,
                op_args=(low, high, size),
                op_kwargs={'generator': generator, 'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def basic_sample_inputs_mint_randint_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.randint_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    # input_shape, low, high, dtype, device
    cases = (
        ((), 0, 45, ms.float64, None),
        ((7,), 3, 25, ms.uint8, None),
        ((2, 0), 0, 1, ms.bfloat16, None),
        ((2, 3, 8 ,2), 0, 1, None, None),
    )

    for input_shape, low, high, case_dtype, case_device in cases:
        if case_dtype is None and dtype.is_complex:
            # for torch op dtype param
            yield OpSampleInput(
                op_input=make_func(input_shape),
                op_args=(low, high),
                op_kwargs={'dtype': ms.int8, 'device': case_device},
                sample_name=f"{op_info.name}_{dtype}_to_int8"
            )
        else:
            yield OpSampleInput(
                op_input=make_func(input_shape),
                op_args=(low, high),
                op_kwargs={'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_{case_dtype}_{case_device}"
            )

def extra_sample_inputs_mint_randint_like(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.randint_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    # input_shape, low, high, dtype, device
    cases = (
        ((3, 6, 2, 4, 5), -4, 12, ms.bfloat16, None),
        ((4, 4, 8, 5, 5, 9, 8, 7), 12, 50, ms.float32, None),
    )

    for input_shape, low, high, case_dtype, case_device in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(low, high),
            op_kwargs={'dtype': case_dtype, 'device': case_device},
            sample_name=f"{op_info.name}_{case_dtype}_{case_device}"
        )
    # large mix small values
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
                op_args=(0, 43),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_args=(0, 45),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )
    # torch op dtype param not support complex
    if dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_args=(0, 45),
                op_kwargs={'dtype': ms.int32},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_mint_randint_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.randint_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(0, 1),
                op_kwargs={'dtype': ms.float16},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 8, 4, 7)),
                    op_args=(0, 1),
                    op_kwargs={'dtype': ms.float16},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 4, 3, 9)),
                    op_args=(-3, 20),
                    op_kwargs={'dtype': ms.float16},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(0, 15),
                op_kwargs={'dtype': ms.float64},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4)),
                    op_args=(0, 15),
                    op_kwargs={'dtype': ms.float64},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9,)),
                    op_args=(6, 21),
                    op_kwargs={'dtype': ms.float64},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_inputs_mint_randint_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.randint_like.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((3,), 0.1, 1, None, None),
        ((3,), 0, 1.0, None, None),
        ((3, 4), 0, 10, 1.0, None),
        ((9,), -3, -12, None, None),
        ((9,), -3, 1, None, 'GPU'),
        # 20260522 the device is CPU, case can pass.
        # ((9,), -3, 1, None, 'CPU'),
    )

    # low isnot int
    # high isnot int
    # dtype is float
    # low is larger than high
    # device value is GPU
    # device value is CPU
    for input_shape, low, high, case_dtype, case_device in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(input_shape),
                op_args=(low, high),
                op_kwargs={'dtype': case_dtype, 'device': case_device},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

    # case-'the device of input is CPU, device is None' can also pass.
    # input isnot tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1.0,
            op_args=(0, 1),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'input not tensor is not supported for {op_info.name}',
    )

def basic_sample_inputs_mint_randn(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.randn.
    """
    # Reuse the same sample inputs as mint.rand
    yield from basic_sample_inputs_mint_rand(op_info, dtype=dtype, device=device, **kwargs)

def extra_sample_inputs_mint_randn(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.randn.
    """
    # Reuse the same sample inputs as mint.rand
    yield from extra_sample_inputs_mint_rand(op_info, dtype=dtype, device=device, **kwargs)

def dynamic_sample_inputs_mint_randn(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.randn.
    """
    # Reuse the same sample inputs as mint.rand
    yield from dynamic_sample_inputs_mint_rand(op_info, dtype=dtype, device=device, **kwargs)

def error_inputs_mint_randn(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.randn.
    """
    # Reuse the same sample inputs as mint.rand
    yield from error_inputs_mint_rand(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_randn_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.randn_like.
    """
    # Reuse the same sample inputs as mint.rand_like
    yield from basic_sample_inputs_mint_rand_like(op_info, dtype=dtype, device=device, **kwargs)

def extra_sample_inputs_mint_randn_like(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.randn_like.
    """
    # Reuse the same sample inputs as mint.rand_like
    yield from extra_sample_inputs_mint_rand_like(op_info, dtype=dtype, device=device, **kwargs)

def dynamic_sample_inputs_mint_randn_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.randn_like.
    """
    # Reuse the same sample inputs as mint.rand_like
    yield from dynamic_sample_inputs_mint_rand_like(op_info, dtype=dtype, device=device, **kwargs)

def error_inputs_mint_randn_like(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.randn_like.
    """
    # Reuse the same sample inputs as mint.rand_like
    yield from error_inputs_mint_rand_like(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_randperm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate basic sample inputs for mint.randperm.
    """
    # n, generator, dtype
    cases = (
        (1, None, dtype),
        (39, None, ms.float32),
        (ms.Tensor(1), None, ms.float64),
        (ms.Tensor(np.random.randint(1, 3, ())), None, ms.float16),
        (ms.Tensor(np.random.randint(1, 5, (1,))), None, ms.float16),
    )

    for n, generator, case_dtype in cases:
        yield OpSampleInput(
            op_input=n,
            op_args=(),
            op_kwargs={'generator': generator, 'dtype': case_dtype},
            sample_name=f"{op_info.name}_{generator}_{case_dtype}"
        )

def extra_sample_inputs_mint_randperm(op_info: OpInfo, dtype=None, device=None, **kwargs ):
    """
    Generate extra sample inputs for mint.randperm.
    """
    cases = (50, 100, 2**24)
    for n in cases:
        yield OpSampleInput(
            op_input=n,
            op_args=(),
            op_kwargs={'dtype': ms.int64},
            sample_name=f"{op_info.name}_{dtype}"
        )

def dynamic_sample_inputs_mint_randperm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.randperm.
    """
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=ms.int64),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=ms.Tensor(5, dtype=ms.int64),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=ms.Tensor(5, dtype=ms.int64),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

def error_inputs_mint_randperm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.randperm.
    """
    cases = (
        (0, ms.int64),
        (1.0, ms.int64),
        (1, 1.0),
        (ms.Tensor(np.random.randint(1, 5, (1, 2))), ms.int64),
        (ms.Tensor(np.random.randint(1, 5, (2, 0))), ms.int64),
    )

    # n is 0
    # n is float
    # dtype is float
    # n is 2d
    # n is empty tensor
    for n, case_dtype in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=None,
                op_args=(n,),
                op_kwargs={'dtype': case_dtype},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def basic_reference_inputs_mint_nn_elu(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        ((), -2.0,  False),
        ((S,), -1.0, False),
        ((M, S), 1.0, True),
        ((S, S, L), 0.4, True),
        ((2, 1, 3, 2), 12.3, False),
        ((2, 3, 4, 1, 2), 2.0, True),
        ((2, 1, 2, 2, 1, 2), 0.0,  False),
        ((2, 1, 2, 2, 1, 2, 1), 1.0, True),
        ((2, 1, 2, 2, 1, 2, 1, 2), 2.0, False),
    )

    for input_shape, i, j in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'alpha':i, 'inplace':j},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_reference_inputs_mint_nn_elu(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #empty tensors/small medium large size
    shapes = (
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )

    for input_shape in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

    # large/small/discontiguous values
    values = _small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for extremal_val, name_suffix in extremal_cases:
        t = make_func((5,))
        t[2] = extremal_val
        yield OpSampleInput(
            op_input=t,
            op_args=(),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
        )

def error_inputs_elu_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    # alpha not float
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func((5,), dtype=ms.float32,),
            op_kwargs={'alpha': '1', 'inplace': False},
            sample_name=f'{op_info.name}_alpha_not_float',
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='elu ops alpha type not float',
    )

def basic_reference_inputs_mint_nn_silu(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        ((), False),
        ((S,), False),
        ((M, S), True),
        ((S, S, L), True),
        ((2, 1, 3, 2), False),
        ((2, 3, 4, 1, 2), True),
        ((2, 1, 2, 2, 1, 2), False),
        ((2, 1, 2, 2, 1, 2, 1), True),
        ((2, 1, 2, 2, 1, 2, 1, 2), False),
    )

    for input_shape, i in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'inplace': i},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def basic_reference_inputs_mint_nn_hardshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        ((), -2.0),
        ((S,), -1.0),
        ((M, S), 1.0),
        ((S, S, L), 0.4),
        ((2, 1, 3, 2), 12.3),
        ((2, 3, 4, 1, 2), 2.0),
        ((2, 1, 2, 2, 1, 2), 0.5),
        ((2, 1, 2, 2, 1, 2, 1), 1.0),
        ((2, 1, 2, 2, 1, 2, 1, 2), 2.0),
    )

    for input_shape, i in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'lambd':i},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_reference_inputs_mint_nn_hardshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #empty tensors/small medium large size
    shapes = (
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )

    for input_shape in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

    # large/small/discontiguous values
    values = _small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for extremal_val, name_suffix in extremal_cases:
        t = make_func((5,))
        t[2] = extremal_val
        yield OpSampleInput(
            op_input=t,
            op_args=(),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
        )

def error_inputs_hardshrink_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    # lambd not number
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func((5,), dtype=ms.float32,),
            op_kwargs={'lambd': '1'},
            sample_name=f'{op_info.name}_lambd_not_number',
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='elu ops lambd type not number',
    )

def basic_reference_inputs_mint_nn_hardsigmoid(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        (),
        (S,),
        (M, S),
        (S, S, L),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for input_shape in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_reference_inputs_mint_nn_hardsigmoid(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #empty tensors/small medium large size
    shapes = (
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )

    for input_shape in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

    # large/small/discontiguous values
    values = _small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for extremal_val, name_suffix in extremal_cases:
        t = make_func((5,))
        t[2] = extremal_val
        yield OpSampleInput(
            op_input=t,
            op_args=(),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
        )

def basic_reference_inputs_mint_nn_logsoftmax(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        ((S,), -1),
        ((M, S), 1),
        ((S, S, L), 0),
        ((2, 1, 3, 2), 1),
        ((2, 3, 4, 1, 2), 1),
        ((2, 1, 2, 2, 1, 2), 1),
        ((2, 1, 2, 2, 1, 2, 1), 2),
        ((2, 1, 2, 2, 1, 2, 1, 2), 3),
    )

    for input_shape, i in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'dim':i},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )
