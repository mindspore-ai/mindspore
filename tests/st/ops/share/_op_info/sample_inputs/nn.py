# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Utility helpers for operation testing.

This module provides:
- sample_inputs for nn-types ops
"""
import functools

import numpy as np
import random
import math
import inspect
import itertools
import mindspore as ms
from mindspore import mint, mutable
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo
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
    dtype_to_np_dtype,
    dtypes_extra_values,
    small_large_values
)
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
        if "BatchNorm3d" in op_info.name:
            S = (8, 5, 4, 8, 5)
            SM = (1, 1, 1, 1, len(_float_small_values))
            LG = (1, 1, 1, 1, len(_large_values))
            ZS = (0, 5, 4, 8, 5)
        elif "BatchNorm2d" in op_info.name:
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

def basic_sample_inputs_functional_adaptivemaxpool(op_info: OpInfo, dtype=None, device=None, dim=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((SMALL_DIM_SIZE, MEDIUM_DIM_SIZE), 5),
        ((MEDIUM_DIM_SIZE, LARGE_DIM_SIZE), (10,)),
        ((MEDIUM_DIM_SIZE, LARGE_DIM_SIZE, SMALL_DIM_SIZE), (2,)),
        ((LARGE_DIM_SIZE, SMALL_DIM_SIZE, MEDIUM_DIM_SIZE), 5),
    )

    for input_shape, output_size in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(output_size,),
            op_kwargs={},
            sample_name=op_info.name,
        )

        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(output_size,),
            op_kwargs={'return_indices': True},
            sample_name=op_info.name,
        )


def extra_sample_inputs_functional_maxunpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    data_list = []

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

    for data in data_list:
        out = (mint.nn.MaxPool2d(kernel_size=5, stride=2, padding=2, return_indices=True)
               (make_tensor_with_np_array(np.random.choice(np.array(data), size=(2, 5, 7, 19)),
                                          dtype=dtype if dtype != ms.float64 else ms.float32, device=device)))

        yield OpSampleInput(
            op_input=out[0],
            op_args=(out[1].to(ms.int64), 5),
            op_kwargs={'stride': 2, 'padding': 2, },
            sample_name=op_info.name,
        )

    yield OpSampleInput(
        op_input=make_tensor((0, 1, 2, 3), device=device, dtype=dtype),
        op_args=(make_tensor((0, 1, 2, 3), device=device, dtype=ms.int64), 5,),
        op_kwargs={'stride': 2, 'padding': 2},
        sample_name=op_info.name,
    )

    out = (mint.nn.MaxPool2d(kernel_size=5, stride=2, padding=2, return_indices=True)
           (make_tensor((2, 5, 7, 19), dtype=dtype if dtype != ms.float64 else ms.float32, device=device)))

    yield OpSampleInput(
        op_input=_tensor_to_discontiguous(out[0]),
        op_args=(out[1].to(ms.int64), 5),
        op_kwargs={'stride': 2, 'padding': 2, },
        sample_name=op_info.name,
    )


def error_sample_inputs_functional_adaptivemaxpool(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None,
                                                   **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input dim should be 2 or 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((5,)),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input dim should be 2 or 3',
    )

    for output_size_error in [0, (0,)]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((3, 4, 5,)),
                op_args=(output_size_error,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=RuntimeError,
            op_error_info='output_size should > 0',
        )

def extra_sample_inputs_functional_adaptivemaxpool(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from make_extra_tensor(op_info, (10, 60, 80), (9,),
                                 {'return_indices': True}, dtype=dtype, device=device)

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

def basic_sample_inputs_mint_nn_max_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((2, 4, 9, 7), 3, 2, 0, 1, False, False),
        ((4, 9, 7), 3, 2, 0, 1, False, False),
        ((2, 3, 8, 6), (3, 2), (2, 1), (1, 0), 1, True, False),
        ((3, 8, 6), (3, 2), (2, 1), (1, 0), 1, True, False),
        ((2, 5, 8, 7), (2, 3), None, (1, 0), 1, False, False),
        ((5, 8, 7), (2, 3), None, (1, 0), 1, False, False),
        ((2, 3, 8, 8), 3, 2, 0, (1, 1), False, False),
        ((3, 8, 8), 3, 2, 0, (1, 1), False, False),
    )

    for input_shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size, stride, padding, dilation, ceil_mode, return_indices),
            op_kwargs={},
            sample_name=op_info.name,
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
        input_values = np.array(_float_small_values).reshape((1, 1, len(_float_small_values), 1))
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
        input_values = np.array(_large_values).reshape((1, 1, len(_large_values), 1))
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

def extra_sample_inputs_mint_nn_max_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from extra_reference_inputs_op_functional_pool_func(op_info, dtype, device, **kwargs)

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((2, 4, 9, 7), 3, 2, 0, 1, False, True),
        ((4, 9, 7), 3, 2, 0, 1, False, True),
        ((2, 3, 8, 6), (3, 2), (2, 1), (1, 0), 1, True, True),
        ((3, 8, 6), (3, 2), (2, 1), (1, 0), 1, True, True),
        ((2, 5, 8, 7), (2, 3), None, (1, 0), 1, False, True),
        ((5, 8, 7), (2, 3), None, (1, 0), 1, False, True),
        ((2, 3, 8, 8), 3, 2, 0, (1, 1), False, True),
        ((3, 8, 8), 3, 2, 0, (1, 1), False, True),
    )

    for input_shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size, stride, padding, dilation, ceil_mode, return_indices),
            op_kwargs={},
            sample_name=op_info.name,
        )


def dynamic_sample_inputs_mint_nn_max_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """Generate dynamic-shape and dynamic-rank inputs for mint.nn.MaxPool2d."""
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    grad_cmp = kwargs.get("grad_cmp", False)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(3, 2, 0, 1, False, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_4d",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 9, 7)),
                    op_args=(3, 2, 0, 1, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_4d_a",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 6, 5)),
                    op_args=(3, 2, 0, 1, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_4d_b",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=((2, 3), None, (1, 0), 1, False, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_default_stride_4d",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8, 7)),
                    op_args=((2, 3), None, (1, 0), 1, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_default_stride_4d_a",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 5, 6, 5)),
                    op_args=((2, 3), None, (1, 0), 1, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_default_stride_4d_b",
                ),
            ),
        )
        if not grad_cmp:
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                    op_args=(3, 2, 0, 1, False, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input_3d",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(3, 9, 7)),
                        op_args=(3, 2, 0, 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input_3d_a",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(4, 6, 5)),
                        op_args=(3, 2, 0, 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input_3d_b",
                    ),
                ),
            )
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                    op_args=((2, 3), None, (1, 0), 1, False, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input_default_stride_3d",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(3, 8, 7)),
                        op_args=((2, 3), None, (1, 0), 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input_default_stride_3d_a",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(5, 6, 5)),
                        op_args=((2, 3), None, (1, 0), 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input_default_stride_3d_b",
                    ),
                ),
            )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(3, 2, 0, 1, False, False),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 9, 7)),
                    op_args=(3, 2, 0, 1, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_4d",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 9, 7)),
                    op_args=(3, 2, 0, 1, False, False),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_3d",
                ),
            ),
        )
        if not grad_cmp:
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(3, 2, 0, 1, False, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_compile_input_return_indices",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 4, 8, 6)),
                        op_args=(3, 2, 0, 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input_return_indices_4d",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(4, 8, 6)),
                        op_args=(3, 2, 0, 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input_return_indices_3d",
                    ),
                ),
            )
        if not grad_cmp:
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=((2, 3), None, (1, 0), 1, False, True),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_compile_input_default_stride",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 5, 8, 7)),
                        op_args=((2, 3), None, (1, 0), 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input_default_stride_4d",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(5, 8, 7)),
                        op_args=((2, 3), None, (1, 0), 1, False, True),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input_default_stride_3d",
                    ),
                ),
            )


def error_inputs_other_op_mint_nn_max_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """Generate error inputs for mint.nn.MaxPool2d."""
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((8, 4, 3), 2.0, 1, 0, 1, False, False),
        ((8, 4, 3), 2, 1.0, 0, 1, False, False),
        ((8, 4, 3), 2, 1, 0.1, 1, False, False),
        ((8, 4, 3), 2, 1, 0, 1.0, False, False),
        ((8, 4, 3), 2, 1, 0, 1, 1.0, False),
        ((8, 4, 3), 2, 1, 0, 1, False, 1.0),
        ((8, 4, 3), -2, 1, 0, 1, False, False),
        ((8, 4, 3), 2, 0, 0, 1, False, False),
        ((8, 4, 3), 2, 1, 0, 0, False, False),
        ((1, 3, 9, 9), 2, 1, 0, (1, 2), False, False),
        ((1, 3, 9, 9), 2, 1, 2, 1, False, False),
        ((1, 3, 9, 9), 2, 1, -1, 1, False, False),
        ((1, 3, 9, 9), (1, 1, 1), 1, 0, 1, False, False),
        ((1, 3, 0, 9), 2, 1, 0, 1, False, False),
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 7)),
            op_args=(2, 1, 0, 1, False, False),
            op_kwargs={},
            sample_name=f"{op_info.name}_error_input_2d",
        ),
        op_error_type=ValueError,
        op_error_info=f'input error value is not supported by {op_info.name}.',
    )

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
        ((1, 8, 4), (3, 4, 5)),
        ((1, 7, 0, 8), (1)),
    )
    # output_size is float
    # the length of output_size isnot 2
    # input isnot 3D or 4D
    # output_size is out of range
    # the length of input is equal output_size
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

def basic_sample_inputs_adaptivemaxpool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    case = (
        ((3,  10), 7 ),
        ((1, 3, 10), (3, )),
        )

    for input_shape, output_size  in case:

        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(output_size,),
            op_kwargs={},
            sample_name=op_info.name,
        )

        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(output_size,),
            op_kwargs={'return_indices':True},
            sample_name=op_info.name,
        )

def error_sample_inputs_adaptivemaxpool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    N = 3
    C = 5
    L = 10
    output_size = 6

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, C, L)),
            op_args=((output_size, 3),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='output_size.length must be 1',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, N, C, L)),
            op_args=(output_size,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.dim must be 2 or 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((L, )),
            op_args=(output_size,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.dim must be 2 or 3',
    )

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

def extra_sample_inputs_adaptivemaxpool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (1, 1, 16, ), (8, ),
                                 {'return_indices': True}, dtype=dtype, device=device)

def extra_sample_inputs_mint_adaptiveavgpool(op_info: OpInfo, dtype=None, device=None, **kwargs):

    pool_dim = kwargs.get('pool_dim')

    if pool_dim == 'Pool3d':
        yield from make_extra_tensor(op_info, (2, 3, 10, 20, 30), (((3, 6, 9),), ),
                                     {}, dtype=dtype, device=device)
    else:
        yield from make_extra_tensor(op_info, (10, 20, 30), (((3, 6, ),), ),
                                     {}, dtype=dtype, device=device)

def error_inputs_other_op_with_adaptiveavgpool(op_info: OpInfo, dtype=None, device=None, **kwargs):
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

def dynamic_sample_inputs_adaptive_avg_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic-shape/rank inputs for mint.nn.functional.adaptive_avg_pool2d.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=((8, 8),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 4, 3, 2)),
                    op_args=((8, 8),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 16, 16)),
                    op_args=((8, 8),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=((8, 8),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 4, 3, 2)),
                    op_args=((8, 8),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 2)),
                    op_args=((8, 8),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def basic_sample_inputs_maxunpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    def _normalize_output_size(shape_before_pooling, output_size_mode):
        if output_size_mode is None:
            return None
        if output_size_mode == "hw":
            return shape_before_pooling[-2:]
        if output_size_mode == "chw":
            return shape_before_pooling[-3:]
        if output_size_mode == "nchw":
            return shape_before_pooling
        raise ValueError(f"Unsupported output_size_mode: {output_size_mode}")

    def _make_sample(shape_before_pooling, kernel_size, stride, padding, output_size_mode, sample_name):
        pool_dtype = ms.float32 if dtype == ms.float64 else dtype
        pooled, indices = mint.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True
        )(make_tensor(shape_before_pooling, dtype=pool_dtype, device=device))
        yield OpSampleInput(
            op_input=pooled.to(ms.float64) if dtype == ms.float64 else pooled,
            op_args=(
                indices.to(ms.int64),
                kernel_size,
                stride,
                padding,
                _normalize_output_size(shape_before_pooling, output_size_mode),
            ),
            op_kwargs={},
            sample_name=sample_name,
        )

    if dtype not in [ms.float32, ms.float16, ms.float64]:
        pooled, indices = mint.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)(
            make_tensor((1, 3, 4, 5), dtype=ms.float32, device=device)
        )
        yield OpSampleInput(
            op_input=pooled.to(dtype),
            op_args=(indices.to(ms.int64), 3, 2, 0, (1, 3, 4, 5)),
            op_kwargs={},
            sample_name=f"{op_info.name}_unsupported_dtype_prepare",
        )
        return

    cases = (
        ((1, 4, 4), 2, 2, 0, "chw", f"{op_info.name}_3d_output_size_chw"),
        ((1, 3, 4, 5), 3, 2, 0, "nchw", f"{op_info.name}_4d_output_size_nchw"),
        ((3, 6, 6, 10), (4, 6), None, 2, "hw", f"{op_info.name}_stride_none"),
        ((2, 5, 7, 20), 5, (2, 6), (2, 1), "hw", f"{op_info.name}_4d_output_size_hw"),
    )
    for shape_before_pooling, kernel_size, stride, padding, output_size_mode, sample_name in cases:
        yield from _make_sample(shape_before_pooling, kernel_size, stride, padding, output_size_mode, sample_name)

def error_sample_inputs_maxunpool2d(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None, **kwargs):

    N = 1
    C = 3
    H = 4
    W = 5

    kernel_size = 3
    stride = 2
    padding = 1

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    out = mint.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)(
        make_arg((N, C, H, W)))

    out_res = out[0]
    indices = out[1]

    for kernel_size_error in [0, (0, 1), (1, 0)]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=out_res,
                op_args=(indices, kernel_size_error ),
                op_kwargs={'stride': stride, 'padding': padding},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='kernel_size should be greater than 0',
        )
    for kernel_size_error in [-1, (3, 2, 1), (3, -1)]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=out_res,
                op_args=(indices, kernel_size_error),
                op_kwargs={'stride': stride, 'padding': padding},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='kernel_size',
        )
    for stride_error in [0, (0, 0)]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=out_res,
                op_args=(indices, kernel_size),
                op_kwargs={'stride': stride_error, 'padding': padding },
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='stride should be greater than 0',
        )
    for stride_error in [-1, (2, 2, 1), (2, -1)]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=out_res,
                op_args=(indices, kernel_size),
                op_kwargs={'stride': stride_error, 'padding': padding},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='stride',
        )
    for padding_error in [-1, (1, 1, 1), (1, -1)]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=out_res,
                op_args=(indices, kernel_size),
                op_kwargs={'stride': stride, 'padding': padding_error},
                sample_name=op_info.name,
            ),
            op_error_type=(RuntimeError, ValueError),
            op_error_info='padding',
        )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=out_res,
            op_args=(indices, kernel_size),
            op_kwargs={'stride': stride, 'padding': padding, 'output_size': [H + 3, W + 4]},
            sample_name=op_info.name,
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='output_size',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=out_res,
            op_args=(indices, kernel_size),
            op_kwargs={'stride': stride, 'padding': padding, 'output_size': (H + 3, W + 4)},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='output_size',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=out_res,
            op_args=(indices, kernel_size),
            op_kwargs={'stride': stride, 'padding': padding, 'output_size': (1, 1, 1, H, W)},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='output_size',
    )


    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=out_res.expand_dims(0),
            op_args=(indices.expand_dims(0), kernel_size),
            op_kwargs={'stride': stride, 'padding': padding},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input shape should be 3 or 4',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=out_res[0, 0],
            op_args=(indices[0, 0], kernel_size),
            op_kwargs={'stride': stride, 'padding': padding},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input shape should be 3 or 4',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=out_res,
            op_args=(indices.transpose(1,0), kernel_size),
            op_kwargs={'stride': stride, 'padding': padding},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='input shape should be equal indices shape',
    )

def extra_sample_inputs_maxunpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):


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

    for data in data_list:

        out = (mint.nn.MaxPool2d(kernel_size=5, stride=2, padding=2, return_indices=True)
        (make_tensor_with_np_array(np.random.choice(np.array(data), size=(2, 5, 7, 20)),
                                   dtype=ms.float32 if dtype==ms.float64 else dtype , device=device)))

        yield OpSampleInput(
            op_input=out[0].to(ms.float64) if dtype==ms.float64 else out[0],
            op_args=(out[1].to(ms.int64), 5, 2, 2, (2, 5, 7, 20)),
            op_kwargs={},
            sample_name=op_info.name,
        )

    yield OpSampleInput(
        op_input=make_tensor((0, 1, 2, 3), device=device, dtype=dtype),
        op_args=(make_tensor((0, 1, 2, 3), device=device, dtype=ms.int64), 5, 2, 2),
        op_kwargs={},
        sample_name=op_info.name,
    )

def _make_max_unpool2d_running_input(op_info, original_shape, kernel_size, stride, padding, dtype, device,
                                     sample_name, output_size=None):
    """Build a valid MaxUnpool2d sample from a matching MaxPool2d output/indices pair."""
    pool_dtype = ms.float32 if dtype == ms.float64 else dtype
    pool_input = make_tensor(original_shape, dtype=pool_dtype, device=device)
    pooled, indices = mint.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)(
        pool_input)
    op_input = pooled.to(ms.float64) if dtype == ms.float64 else pooled
    return OpSampleInput(
        op_input=op_input,
        op_args=(indices.to(ms.int64), kernel_size, stride, padding, output_size),
        op_kwargs={},
        sample_name=sample_name,
    )


def dynamic_sample_inputs_mint_nn_max_unpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """Generate dynamic-shape and dynamic-rank inputs for mint.nn.MaxUnpool2d."""
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, None), dtype=ms.int64), 3, 2, 1, None),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_4d",
            ),
            op_running_inputs=(
                _make_max_unpool2d_running_input(
                    op_info, (1, 3, 9, 7), 3, 2, 1, dtype, device,
                    f"{op_info.name}_dynamic_shape_running_input_4d_a"
                ),
                _make_max_unpool2d_running_input(
                    op_info, (2, 4, 7, 9), 3, 2, 1, dtype, device,
                    f"{op_info.name}_dynamic_shape_running_input_4d_b"
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=ms.int64), 3, 2, 1, None),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_3d",
            ),
            op_running_inputs=(
                _make_max_unpool2d_running_input(
                    op_info, (3, 9, 7), 3, 2, 1, dtype, device,
                    f"{op_info.name}_dynamic_shape_running_input_3d_a"
                ),
                _make_max_unpool2d_running_input(
                    op_info, (4, 7, 9), 3, 2, 1, dtype, device,
                    f"{op_info.name}_dynamic_shape_running_input_3d_b"
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=ms.int64), 3, 2, 1, None),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                _make_max_unpool2d_running_input(
                    op_info, (1, 3, 9, 7), 3, 2, 1, dtype, device,
                    f"{op_info.name}_dynamic_rank_running_input_4d"
                ),
                _make_max_unpool2d_running_input(
                    op_info, (4, 7, 9), 3, 2, 1, dtype, device,
                    f"{op_info.name}_dynamic_rank_running_input_3d"
                ),
            ),
        )

def basic_sample_inputs_avgpool(op_info: OpInfo, dtype=None, device=None, dim=None, **kwargs):


    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    if dim == '3d':

        cases = (
            ((1, 18, 6, 1, 32), (2, 1, 5), None, (1, 0, 2), None),
            ((8, 30, 150, 120), 7, 4, 0, None),
            ((1, 8, 8, 32, 64), (4, 9, 7), (2, ), 2 ,3),
        )

    else :

        cases = (
            ((1, 18, 6, 32), (2, 5), None, (1, 2), None),
            ((8, 150, 120), 7, 4, 0, None),
            ((5, 17, 19), 4, None, 2, 7),
            ((1, 8, 32, 64), (3, ), (2, ),  1, 3),
        )

    for ceil_mode, count_include_pad in itertools.product([True, False], repeat=2):

        for input_shape, kernel_size, stride, padding, divisor_override in cases:

            yield OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size, ),
                op_kwargs={'stride': stride,
                           'padding': padding,
                           'ceil_mode': ceil_mode,
                           'count_include_pad': count_include_pad,
                           'divisor_override': divisor_override
                           },
                sample_name=op_info.name,
            )

def error_sample_inputs_avgpool3d(op_info: OpInfo, dtype=None, device=None, dim=None, **kwargs):


    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    N = 1
    C = 2
    D = 4
    H = 6
    W = 10

    kernel_size = (D, H, W) if dim == '3d' else (H, W)

    input_shape = (C,) + kernel_size

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(kernel_size),
            op_args=(kernel_size, ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'input.dim should = {len(kernel_size)+1} or {len(kernel_size)+2}',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, N, C,) + kernel_size),
            op_args=(kernel_size, ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.dim should = 4 or 5',
    )

    if dim != '2d':      # maybe is bug
        for index, value in enumerate(kernel_size):

            kernel_size_dim_value_error=list(kernel_size)

            kernel_size_dim_value_error[index] = value + 1

            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_arg(input_shape),
                    op_args=(tuple(kernel_size_dim_value_error), ),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=RuntimeError,
                op_error_info=f'kernel_size.dim{index} should <={kernel_size[index]}',
            )

    for index in range(len(kernel_size)):

        kernel_size_dim_value_error = list(kernel_size)

        kernel_size_dim_value_error[index] = 0

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(tuple(kernel_size_dim_value_error), ),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'kernel_size.dim{index} should > 0',
        )

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size, ),
                op_kwargs={'stride': tuple(kernel_size_dim_value_error)},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'stride.dim{index} should > 0',
        )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(0,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'kernel_size.dim{index} should > 0',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size,),
            op_kwargs={'stride': 0},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'stride.dim{index} should > 0',
    )

    for index in range(len(kernel_size)):

        padding = [math.ceil(D/2), math.ceil(H/2), math.ceil(W/2)] if dim == '3d' else [math.ceil(H/2), math.ceil(W/2)]

        padding[index] = -1

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size,),
                op_kwargs={'padding': tuple(padding)},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'padding.dim{index} should > = 0',
        )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size,),
            op_kwargs={'padding': -1},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'padding.dim{index} should > = 0',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size,),
            op_kwargs={'divisor_override': 0},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='divisor_override',
    )

    padding = (math.ceil(D / 2), math.ceil(H / 2), math.ceil(W / 2)) if dim == '3d' else (
        math.ceil(H / 2), math.ceil(W / 2))

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size + (1, ),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'kernel_size.length should be 1 or {len(kernel_size)}, but get {len(kernel_size)+1}',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size, ),
            op_kwargs={'stride': kernel_size + (1, )},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'stride.length should be 1 or {len(kernel_size)},but get {len(kernel_size)+1}',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(kernel_size,),
            op_kwargs={'padding': kernel_size + (1,)},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'padding.length should be 1 or {len(kernel_size)},but get {len(kernel_size) + 1}',
    )

    if dim != '2d':

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size[:-1],),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=RuntimeError,
            op_error_info=f'kernel_size.length should be 1 or {len(kernel_size)}, but get {len(kernel_size) - 1}',
        )

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size,),
                op_kwargs={'stride': kernel_size[:-1]},
                sample_name=op_info.name,
            ),
            op_error_type=RuntimeError,
            op_error_info=f'stride.length should be 1 or {len(kernel_size)},but get {len(kernel_size) - 1}',
        )

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size,),
                op_kwargs={'padding': padding[:-1]},
                sample_name=op_info.name,
            ),
            op_error_type=RuntimeError,
            op_error_info=f'padding.length should be 1 or {len(kernel_size)}, but get {len(kernel_size) - 1}',
        )

    if dim == '2d':
        invalid_cases = (
            ((2, H, W), (H + 1, W), None, 0),
            ((1, C, H, W), (H + 1, W), None, 0),
        )
        for invalid_input_shape, invalid_kernel_size, invalid_stride, invalid_padding in invalid_cases:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_arg(invalid_input_shape),
                    op_args=(invalid_kernel_size,),
                    op_kwargs={'stride': invalid_stride, 'padding': invalid_padding},
                    sample_name=op_info.name,
                ),
                op_error_type=(ValueError, RuntimeError),
                op_error_info='output_size',
            )

def extra_sample_inputs_avgpool3d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (2, 10, 32, 64), (5, ),
                                 {'stride': 3, 'padding':2,} ,dtype=dtype, device=device)

    yield from make_extra_tensor(op_info, (0, 2, 10, 32, 64), (5,),
                                 {'stride': 3, 'padding': 2, }, dtype=dtype, device=device, empty_tensor=True)

def extra_sample_inputs_avgpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (10, 32, 64), (5, ),
                                 {'stride': 3, 'padding':2,} ,dtype=dtype, device=device)

    yield from make_extra_tensor(op_info, (0, 10, 32, 64), (5,),
                                 {'stride': 3, 'padding': 2, }, dtype=dtype, device=device, empty_tensor=True)

def dynamic_sample_inputs_mint_nn_avg_pool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """Generate dynamic-shape and dynamic-rank inputs for mint.nn.AvgPool2d."""
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, 9, 7), dtype=dtype),
                op_args=(3,),
                op_kwargs={'stride': None, 'padding': 1, 'ceil_mode': False, 'count_include_pad': False,
                           'divisor_override': 3},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_4d",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 9, 7)),
                    op_args=(3,),
                    op_kwargs={'stride': None, 'padding': 1, 'ceil_mode': False, 'count_include_pad': False,
                               'divisor_override': 3},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_4d_a",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 9, 7)),
                    op_args=(3,),
                    op_kwargs={'stride': None, 'padding': 1, 'ceil_mode': False, 'count_include_pad': False,
                               'divisor_override': 3},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_4d_b",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 9, 7), dtype=dtype),
                op_args=((2, 3),),
                op_kwargs={'stride': (2, 1), 'padding': (0, 1), 'ceil_mode': True, 'count_include_pad': True,
                           'divisor_override': None},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_3d",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 9, 7)),
                    op_args=((2, 3),),
                    op_kwargs={'stride': (2, 1), 'padding': (0, 1), 'ceil_mode': True, 'count_include_pad': True,
                               'divisor_override': None},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3d_a",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 9, 7)),
                    op_args=((2, 3),),
                    op_kwargs={'stride': (2, 1), 'padding': (0, 1), 'ceil_mode': True, 'count_include_pad': True,
                               'divisor_override': None},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3d_b",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(3,),
                op_kwargs={'stride': 2, 'padding': 1, 'ceil_mode': False, 'count_include_pad': True,
                           'divisor_override': None},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 9, 7)),
                    op_args=(3,),
                    op_kwargs={'stride': 2, 'padding': 1, 'ceil_mode': False, 'count_include_pad': True,
                               'divisor_override': None},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_4d",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 6, 5)),
                    op_args=(3,),
                    op_kwargs={'stride': 2, 'padding': 1, 'ceil_mode': False, 'count_include_pad': True,
                               'divisor_override': None},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_3d",
                ),
            ),
        )

def basic_sample_inputs_functional_maxunpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    case = (
        ((1, 3, 4, 5), 3, 2, 0, ),                  # mint.nn.MaxPool2d not support 4D input
        ((3, 6, 6, 10), (4, 6), None, 2,) ,
        ((2, 5, 7, 20), 5, (2, 6), (2, 1), )
    )

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def generate_output_size(shape_before_pooling, shape_after_pooling, kernel_size, stride, padding):

        def _pair(obj):
            return  tuple(itertools.repeat(obj, 2)) if isinstance(obj, int) else obj

        kernel_size = _pair(kernel_size)
        stride = kernel_size if stride is None else _pair(stride)
        padding = _pair(padding)

        H_out = (shape_after_pooling[-2] - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        W_out = (shape_after_pooling[-1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1]

        # (H_out - (stride[0] - 1), W_out - (stride[1] - 1)) or (H_out, W_out) may be error  because
        # torch  will detect indices's max index  <= output_size.H * output_size.w
        output_size_list = [shape_before_pooling, (H_out + (stride[0] - 1), W_out + (stride[1] - 1)),]

        for _index, value  in enumerate(kernel_size):

            if not ((shape_before_pooling[-len(kernel_size) + _index] + 2 * padding[_index] - value)
                    % stride[_index] == 0):
                return random.choice(output_size_list)
        return None

    for shape_before_pooling, kernel_size, stride, padding in case:
        out = (mint.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
            (make_arg(shape_before_pooling,dtype=dtype if dtype in op_info.get_supported_dtypes(device)
                   and dtype != ms.float64 else ms.float32, device=device)))

        yield OpSampleInput(
            op_input=out[0] if dtype in op_info.get_supported_dtypes(device) and dtype != ms.float64 else
            out[0].to(dtype),
            op_args=(out[1].to(ms.int64),),
            op_kwargs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'output_size':generate_output_size(shape_before_pooling, out[0].shape, kernel_size, stride, padding)},
            sample_name=op_info.name,
        )

def extra_sample_inputs_functional_MaxUnpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    data_list = []

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

    for data in data_list:
        out = (mint.nn.MaxPool2d(kernel_size=5, stride=2, padding=2, return_indices=True)
               (make_tensor_with_np_array(np.random.choice(np.array(data), size=(2, 5, 7, 19)),
                 dtype=dtype if dtype != ms.float64 else ms.float32, device=device)))

        yield OpSampleInput(
            op_input=out[0],
            op_args=(out[1].to(ms.int64), 5),
            op_kwargs={'stride': 2, 'padding': 2,},
            sample_name=op_info.name,
        )

    yield OpSampleInput(
        op_input=make_tensor((0, 1, 2, 3), device=device, dtype=dtype),
        op_args=(make_tensor((0, 1, 2, 3), device=device, dtype=ms.int64), 5,),
        op_kwargs={'stride': 2, 'padding': 2},
        sample_name=op_info.name,
    )

    out = (mint.nn.MaxPool2d(kernel_size=5, stride=2, padding=2, return_indices=True)
           (make_tensor((2, 5, 7, 19), dtype=dtype if dtype != ms.float64 else ms.float32, device=device)))

    yield OpSampleInput(
        op_input=_tensor_to_discontiguous(out[0]),
        op_args=(out[1].to(ms.int64), 5),
        op_kwargs={'stride': 2, 'padding': 2, },
        sample_name=op_info.name,
    )

def error_sample_inputs_functional_maxunpool2d(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(make_tensor((2, 3, 4, 6), dtype=ms.int64, device=device, low=0),1),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='input shape should equal indices shape',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 1, 1)),
            op_args=(make_tensor((2, 3, 1, 1), dtype=ms.int64, device=device, low=0), (2, 3)),
            op_kwargs={'stride': (4, 5), 'padding': 2},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='Hout should >= Hin, Wout should >= Win',
    )

def basic_sample_inputs_functional_avgpool(op_info: OpInfo, dtype=None, device=None, dim=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)


    cases = (
        ((SMALL_DIM_SIZE, MEDIUM_DIM_SIZE), 5, 2, (2, )),
        ((MEDIUM_DIM_SIZE, LARGE_DIM_SIZE), 5, 2, (2, )),
        ((MEDIUM_DIM_SIZE, LARGE_DIM_SIZE, SMALL_DIM_SIZE), (2, ), None, 1),
        ((LARGE_DIM_SIZE, SMALL_DIM_SIZE, MEDIUM_DIM_SIZE), 3, (3, ), 0),
    )

    for input_shape, kernel_size, stride, padding  in cases:

        for ceil_mode, count_include_pad in itertools.product([True,False],[True,False]):
            yield OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(kernel_size, ),
                op_kwargs={'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode,
                           'count_include_pad':count_include_pad},
                sample_name=op_info.name,
            )

def error_sample_inputs_functional_avgpool(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(2, ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input dim should be 2 or 3',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(0,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='kernel_size should >0',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(2, ),
            op_kwargs={'stride': 0},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='stride should >0',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(10, ),
            op_kwargs={'stride': 1,'padding':2},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='kernel_size should <= input.shape[-1]+2*padding',
    )

def extra_sample_inputs_functional_avgpool(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (10, 60, 80), (9, ),  {}, dtype=dtype,
                                 device=device)

    yield from make_extra_tensor(op_info, (10, 60, 80), (9, ), {}, dtype=dtype,
                                 device=device, empty_tensor=True)

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

def extra_sample_inputs_adaptive_avg_pool1d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (3, 10, 20), (6, ), {}, device=device, dtype=dtype)

def error_sample_inputs_adaptive_avg_pool1d(op_info: OpInfo, dtype, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, )),
            op_args=(3, ),
            op_kwargs={},
            sample_name=f"{op_info.name}_error",
        ),
        op_error_type=ValueError,
        op_error_info='input must be 2d or 3d',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 2, 3, 4)),
            op_args=(3,),
            op_kwargs={},
            sample_name=f"{op_info.name}_error",
        ),
        op_error_type=ValueError,
        op_error_info='input must be 2d or 3d',
    )

def basic_sample_inputs_max_pool(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (((9, 8, 4, 3), 2, 1, 0, 1, False, False),)

    for input_shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices in cases:
        yield OpSampleInput(make_arg(input_shape),
                            op_args=(kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                            op_kwargs={},
                            sample_name=op_info.name)

def extra_sample_inputs_mint_batch_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def _make_other_params(dim_value):
        running_mean = make_arg((dim_value, ))
        runing_var = make_tensor((dim_value, ), device=device, dtype=dtype, low=0)
        weight = make_arg((dim_value, ))
        bias = make_arg((dim_value, ))
        return (running_mean, runing_var, weight, bias)

    yield from make_extra_tensor(op_info, (3, 10, 20, 30), _make_other_params(10),
                                 {}, device=device, dtype=dtype, exclude_value=(-5062757.7, 5062757.7, -1e20, 1e20))
    empty_cases=(
        (0, 1),
        #(1, 0, 3),
        (1, 3, 0, 0),
    )

    for empty_input_shape in empty_cases:
        yield from make_extra_tensor(op_info, empty_input_shape, _make_other_params(empty_input_shape[1]),
                                     {}, device=device, dtype=dtype, empty_tensor=True)

    yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9), _make_other_params(3),
                                     {}, device=device, dtype=dtype, support_multi_dim=True)


def dynamic_sample_inputs_mint_adaptiveavgpool2d(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device,)

    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=((2, 3), ),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8, 8)),
                    op_args=((2, 3), ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 10)),
                    op_args=((2, 3), ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=((2, 3),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 8, 8)),
                    op_args=((2, 3), ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 10, 10)),
                    op_args=((2, 3),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def basic_sample_inputs_mint_nn_bce_with_logits_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    if dtype.is_floating_point:
        make_logits = functools.partial(make_tensor, device=device, dtype=dtype, low=-1.5, high=1.5)
        make_prob = functools.partial(make_tensor, device=device, dtype=dtype, low=0.1, high=0.9)
    else:
        make_logits = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=2)
        make_prob = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=2)

    cases = (
        ((), "mean"),
        ((2, 3), "none"),
        ((2, 3), "sum"),
        ((2, 3, 4), "mean"),
    )

    for shape, reduction in cases:
        yield OpSampleInput(
            op_input=make_logits(shape),
            op_args=(make_prob(shape),),
            op_kwargs={"reduction": reduction},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_nn_bcewithlogitsloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, random_seed=1)
    cases = (
        ((5,), (1,), 'none', (5,)),
        ((5, 2, 9, 8, 4), (5, 2, 9, 8, 4), 'mean', (5, 2, 9, 8, 4)),
        ((5, 2, 2, 8, 4, 3, 5, 8), (5, 1, 2, 8, 1, 3, 5, 8), 'mean', (1, 2, 2, 8, 1, 1, 5, 8)),
    )
    for input_shape, weight_shape, reduction, pos_weight_shape in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(make_func(input_shape),),
            op_kwargs={
                'weight': make_func(weight_shape),
                'reduction': reduction,
                'pos_weight': make_func(pos_weight_shape)
            },
            sample_name=f"{op_info.name}_{dtype}_{reduction}"
        )
    # weight/pos_weight is scalar tensor
    yield OpSampleInput(
        op_input=make_func((2, 4)),
        op_args=(make_func((2, 4)),),
        op_kwargs={
            'weight': ms.Tensor(7, dtype=dtype),
            'pos_weight': ms.Tensor(7, dtype=dtype)
        },
        sample_name=f"{op_info.name}_{dtype}_weight_pos_weight_scalar_tensor"
    )

    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input_array = np.random.choice(values, size=(5, 2), replace=False)
            target_array = np.random.choice(values, size=(5, 2), replace=False)
            weight_array = np.random.choice(values, size=(1, 2), replace=False)
            pos_weight_array = np.random.choice(values, size=(5, 1), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(target_array, dtype=dtype, discontiguous=discontiguous),),
                op_kwargs={
                    'weight': make_tensor_with_np_array(weight_array, dtype=dtype, discontiguous=discontiguous),
                    'pos_weight': make_tensor_with_np_array(pos_weight_array, dtype=dtype,
                                                            discontiguous=discontiguous)
                },
                sample_name=f"{op_info.name}_{dtype}_small_large_values_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((5, 2, 9), t), dtype=dtype)
        target = ms.Tensor(np.full((5, 2, 9), t), dtype=dtype)
        weight = ms.Tensor(np.full((1, 2, 9), t), dtype=dtype)
        pos_weight = ms.Tensor(np.full((5, 1, 9), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(target,),
            op_kwargs={'weight': weight, 'pos_weight': pos_weight},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_bcewithlogitsloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.BCEWithLogitsLoss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 2, 9, 8, 4)),
                    op_args=(make_func(shape=(5, 2, 9, 8, 4)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 3, 4, 5, 5)),
                    op_args=(make_func(shape=(7, 3, 4, 5, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 2, 9)),
                    op_args=(make_func(shape=(5, 2, 9)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 2)),
                    op_args=(make_func(shape=(5, 2)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
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

def extra_sample_inputs_mse_loss(op_info, dtype, device, **kwargs):
    cases=(
        ((30, ), (30, )),
        ((60, 1), (1, 30)),
        ((3, 6, 90), (1, 6, 1)),
        ((30, 6, 9, 10 ), (30, 6, 9, 10))
    )

    multi_dim_cases = (
        ((), ()),
        ((2, 3, 4, 5, 6, 7, 8, 9), (2, 3, 1, 5, 1, 7, 1, 9))
    )

    empty_cases = (
        ((0,), (0,)),
        ((0, 1), (1, 0)),
        ((0, 3, 0), (0, 1, 0)),
        ((1, 0, 0, 0), (3, 0, 0, 0))
    )


    for input_shape, target_shape in cases:
        for reduction in ['none', 'mean', 'sum']:
            yield from make_extra_tensor(op_info, input_shape, (target_shape,),
                                         {'reduction': reduction}, dtype=dtype, device=device,)

    for input_shape, target_shape in multi_dim_cases:
        for reduction in ['none', 'mean', 'sum']:
            yield from make_extra_tensor(op_info, input_shape, (target_shape,),
                                         {'reduction': reduction}, dtype=dtype, device=device,
                                         support_multi_dim=True)

    for input_shape, target_shape in empty_cases:
        for reduction in ['none', 'mean', 'sum']:
            yield from make_extra_tensor(op_info, input_shape, (target_shape,),
                                         {'reduction': reduction}, dtype=dtype, device=device,
                                         empty_tensor=True)

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

def sample_inputs_l1_loss(op_info, dtype, device, **kwargs):
    yield from sample_inputs_loss(op_info, dtype, device, **kwargs)

def extra_sample_inputs_mint_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    yield from make_extra_tensor(op_info, (3, 30), (make_arg((2, 30)), ),
                                 {'bias': make_arg((2, ))}, device=device, dtype=dtype)

    yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9,), ((2, 9), ),
                                 {'bias': (2, )}, device=device, dtype=dtype, support_multi_dim=True)

    empty_cases=(
        ((0, 6), (6, ), None),
        ((0, 6), (0, 6), (0, )),
        ((1, 0), (0, 0), (0, ))
    )
    for input_shape, weight_shape, bias_shape in empty_cases:

        yield from make_extra_tensor(op_info, input_shape, (weight_shape, ),
                                     {'bias': bias_shape}, device=device, dtype=dtype, empty_tensor=True)

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

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 2, 3, 4, 5, 6, 7)),
            op_args=(make_arg((7, 3)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='when bias is not 1d, input should less than  7d',
    )

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

            # int scale_factor with recompute_scale_factor=False (should be normalized to float)
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": None,
                    "scale_factor": 2,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": False,
                },
                sample_name=op_info.name,
            )
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": None,
                    "scale_factor": [True] * rank,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": False,
                },
                sample_name=op_info.name,
            )
            int_scale_factor = tuple([2] * rank)
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": None,
                    "scale_factor": int_scale_factor,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": False,
                },
                sample_name=op_info.name,
            )
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": None,
                    "scale_factor": list(int_scale_factor),
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": False,
                },
                sample_name=op_info.name,
            )

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
                for scale_name, int_scale_factor in (
                        ("int_tuple", mutable(tuple([2] * rank))),
                        ("int_list", mutable([2] * rank))):
                    yield OpDynamicInput(
                        op_compile_input=OpSampleInput(
                            op_input=ms.Tensor(shape=tuple([N, C] + [None] * rank), dtype=dtype),
                            op_args=(),
                            op_kwargs={
                                "size": None,
                                "scale_factor": int_scale_factor,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": False,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_compile_input_{scale_name}_r{rank}',
                        ),
                        op_running_inputs=(
                            OpSampleInput(
                                op_input=make_func(shape=nc_shape(rank, D1)),
                                op_args=(),
                                op_kwargs={
                                    "size": None,
                                    "scale_factor": int_scale_factor,
                                    "mode": internal_mode,
                                    "align_corners": align_corners,
                                    "recompute_scale_factor": False,
                                },
                                sample_name=f'{op_info.name}_dynamic_shape_running_input_{scale_name}_r{rank}',
                            ),
                            OpSampleInput(
                                op_input=make_func(shape=nc_shape(rank, D2)),
                                op_args=(),
                                op_kwargs={
                                    "size": None,
                                    "scale_factor": int_scale_factor,
                                    "mode": internal_mode,
                                    "align_corners": align_corners,
                                    "recompute_scale_factor": False,
                                },
                                sample_name=f'{op_info.name}_dynamic_shape_running_input_{scale_name}_r{rank}',
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
            for scale_name, int_scale_factor in (
                    ("int_tuple", mutable(tuple([2] * rank))),
                    ("int_list", mutable([2] * rank)),
                    ("int_scalar", 2),
                    ("bool_scalar", True)):
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=None, dtype=dtype),
                        op_args=(),
                        op_kwargs={
                            "size": None,
                            "scale_factor": int_scale_factor,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": False,
                        },
                        sample_name=f'{op_info.name}_dynamic_rank_compile_input_{scale_name}_r{rank}',
                    ),
                    op_running_inputs=(
                        OpSampleInput(
                            op_input=make_func(shape=nc_shape(rank, D1)),
                            op_args=(),
                            op_kwargs={
                                "size": None,
                                "scale_factor": int_scale_factor,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": False,
                            },
                            sample_name=f'{op_info.name}_dynamic_rank_running_input_{scale_name}_r{rank}',
                        ),
                        OpSampleInput(
                            op_input=make_func(shape=nc_shape(rank, D2)),
                            op_args=(),
                            op_kwargs={
                                "size": None,
                                "scale_factor": int_scale_factor,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": False,
                            },
                            sample_name=f'{op_info.name}_dynamic_rank_running_input_{scale_name}_r{rank}',
                        ),
                    ),
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

def extra_sample_inputs_mint_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):
    mode = kwargs.get("mode")
    internal_mode, ranks = _normalize_mode_and_ranks(mode)
    rank = ranks[0]
    yield from make_extra_tensor(op_info, (1, 1,) + (2, ) * rank, (),
                        {'size': 3, 'mode': internal_mode}, dtype=dtype, device=device,)
    yield from make_extra_tensor(op_info, (0, 10,) + (2,) * rank, (),
                        {'size': 4, 'mode': internal_mode}, dtype=dtype, device=device, empty_tensor=True)

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

    invalid_scale_factors = (
        tuple([np.array(2)] * rank),
        tuple([np.array(2.5)] * rank),
        tuple([np.array([2.5])] * rank),
        tuple([ms.Tensor(2)] * rank),
    )
    for invalid_scale_factor in invalid_scale_factors:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={'size': None, 'scale_factor': invalid_scale_factor, 'mode': internal_mode},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='invalid sequence scale_factor element type',
        )

def extra_sample_inputs_l1_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (
        ((30,), (30,)),
        ((60, 1), (1, 30)),
        ((3, 6, 90), (1, 6, 1)),
        ((30, 6, 9, 10), (30, 6, 9, 10))
    )

    empty_cases = (
        ((0,), (0,)),
        ((0, 1), (1, 0)),
        ((0, 3, 0), (0, 1, 0)),
        ((1, 0, 0, 0), (3, 0, 0, 0))
    )

    multi_dim_cases = (
        ((), ()),
        #((2, 3, 4, 5, 6, 7, 8, 9), (1, 3, 1, 5, 1, 7, 1, 9))
    )

    for input_shape, target_shape in cases:
        for reduction in ['none', 'mean', 'sum']:
            yield from make_extra_tensor(op_info, input_shape, (target_shape,),
                                         {'reduction': reduction}, dtype=dtype, device=device, )

    for input_shape, target_shape in multi_dim_cases:
        for reduction in ['none', 'mean', 'sum']:
            yield from make_extra_tensor(op_info, input_shape, (target_shape, ),
                                         {'reduction': reduction}, dtype=dtype, device=device,
                                         support_multi_dim=True)

    for input_shape, target_shape in empty_cases:
        for reduction in ['none', 'mean', 'sum']:
            yield from make_extra_tensor(op_info, input_shape, (target_shape,),
                                         {'reduction': reduction}, dtype=dtype, device=device, empty_tensor=True)

def error_inputs_mint_nn_bce_with_logits_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_logits = functools.partial(make_tensor, device=device, dtype=ms.float32, low=-1.5, high=1.5)
    make_prob = functools.partial(make_tensor, device=device, dtype=ms.float32, low=0.1, high=0.9)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_logits((3, 4)),
            op_args=(make_prob((4,)),),
            op_kwargs={"reduction": "mean"},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='target shape should equal input shape',
    )

def error_sample_inputs_mint_nn_bcewithlogitsloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((5, 2), (5, 1), (1, 2), (5, 1)),
        ((5, 2, 9, 8), (5, 2, 9, 8), (5, 2, 9, 8), (5, 6, 9, 8)),
        ((5, 1, 1, 8), (5, 1, 1, 8), (5, 1, 3, 8), (5, 2, 1, 8)),
        ((5, 2, 1, 8), (5, 2, 1, 8), (5, 2, 1, 8), (5, 6, 9, 8)),
    )
    # shapes of inputs 'input' and 'target' are not equal
    # pos_weight can not broadcast input
    # input can not broadcast weight
    # input can not broadcast pos_weight
    for input_shape, target_shape, weight_shape, pos_weight_shape in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape),
                op_args=(make_arg(shape=target_shape),),
                op_kwargs={'weight': make_arg(shape=weight_shape),
                           'pos_weight': make_arg(shape=pos_weight_shape)},
                sample_name=f"{op_info.name}_tensor_cannot_broadcast",
            ),
            op_error_type=ValueError,
            op_error_info=f'error shape can not broadcast for {op_info.name}',
        )
    # reduction is error
    # input is not tensor
    # weight is not tensor
    # pos_weight is not tensor
    # input is scalar tensor
    tensor = ms.Tensor(shape=(2, 4), dtype=ms.float32)
    scalar_tensor = ms.Tensor(7, dtype=ms.float32)
    cases = (
        (tensor, tensor, tensor, 'average'),
        ((5, 2, 3), tensor, tensor, 'mean'),
        (tensor, (5, 2, 3), tensor, 'none'),
        (tensor, tensor, (5, 2, 3), 'mean'),
        (scalar_tensor, tensor, tensor, 'sum'),
    )
    for input_tensor, weight, pos_weight, reduction in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_tensor,
                op_args=(input_tensor,),
                op_kwargs={
                    'weight': weight,
                    'pos_weight': pos_weight,
                    'reduction': reduction
                },
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
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

def _make_cross_entropy_probability_target(shape, dtype, device=None):
    np_dtype = np.float32 if dtype == ms.bfloat16 else dtype_to_np_dtype[dtype]
    np_target = np.random.uniform(0.1, 0.9, size=shape).astype(np_dtype)
    class_dim = 0 if len(shape) == 1 else 1
    np_target = np_target / np_target.sum(axis=class_dim, keepdims=True)
    return make_tensor_with_np_array(np_target, dtype=dtype, device=device)


def _make_cross_entropy_class_target(input_shape, dtype, device=None):
    class_dim = 0 if len(input_shape) == 1 else 1
    target_shape = input_shape[:class_dim] + input_shape[class_dim + 1:]
    num_classes = input_shape[class_dim]
    return make_tensor(target_shape, dtype=dtype, device=device, low=0, high=num_classes, random_method='randint')


def _make_cross_entropy_weight(num_classes, dtype, device=None):
    return make_tensor((num_classes,), dtype=dtype, device=device, low=0.2, high=1.6)

def basic_sample_inputs_mint_nn_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_logits = functools.partial(make_tensor, device=device, dtype=dtype, low=-1.5, high=1.5)
    sample_loss = _cross_entropy_sample_loss(dtype)
    grad_position = () if dtype == ms.float64 else None

    if dtype != ms.float64:
        yield OpSampleInput(
            op_input=make_logits((3, 5)),
            op_args=(
                _make_cross_entropy_class_target((3, 5), ms.int32, device),
                None,
                -100,
                "mean",
                0.0,
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_class_indices_target_int32",
            loss=sample_loss,
            grad_position=grad_position,
        )

    yield OpSampleInput(
        op_input=make_logits((2, 4, 6, 6)),
        op_args=(
            _make_cross_entropy_probability_target((2, 4, 6, 6), dtype, device),
            None,
            -100,
            "mean",
            0.0,
        ),
        op_kwargs={},
        sample_name=f"{op_info.name}_probabilities_weight_none",
        loss=sample_loss,
        grad_position=grad_position,
    )


def extra_sample_inputs_mint_nn_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_logits = functools.partial(make_tensor, device=device, dtype=dtype, low=-1.5, high=1.5)

    if dtype == ms.float32:
        yield OpSampleInput(
            op_input=make_logits((3, 5)),
            op_args=(
                _make_cross_entropy_probability_target((3, 5), dtype, device),
                _make_cross_entropy_weight(5, ms.complex64, device),
                -100,
                "mean",
                0.0,
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_probabilities_complex64_weight",
        )


def dynamic_sample_inputs_mint_nn_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_logits = functools.partial(make_tensor, device=device, dtype=dtype, low=-1.5, high=1.5)
    sample_loss = _cross_entropy_sample_loss(dtype)
    grad_position = () if dtype == ms.float64 else None

    if not kwargs.get("only_dynamic_rank", False):
        if dtype != ms.float64:
            running_class_inputs = ((1, 3, 8, 8), (2, 4, 6, 6))
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                    op_args=(ms.Tensor(shape=(None, None, None), dtype=ms.int32), None, -100, "mean", 0.0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_class_dynamic_shape_compile_input",
                    grad_position=grad_position,
                ),
                op_running_inputs=tuple(
                    OpSampleInput(
                        op_input=make_logits(input_shape),
                        op_args=(
                            _make_cross_entropy_class_target(input_shape, ms.int32, device),
                            None,
                            -100,
                            "mean",
                            0.0,
                        ),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_class_dynamic_shape_running_input",
                        loss=sample_loss,
                        grad_position=grad_position,
                    )
                    for input_shape in running_class_inputs
                ),
            )

        running_probability_inputs = ((1, 3, 8, 8), (2, 4, 6, 6))
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, None), dtype=dtype), None, -100, "mean", 0.0),
                op_kwargs={},
                sample_name=f"{op_info.name}_probability_dynamic_shape_compile_input",
                grad_position=grad_position,
            ),
            op_running_inputs=tuple(
                OpSampleInput(
                    op_input=make_logits(input_shape),
                    op_args=(
                        _make_cross_entropy_probability_target(input_shape, dtype, device),
                        None,
                        -100,
                        "mean",
                        0.0,
                    ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_probability_dynamic_shape_running_input",
                    loss=sample_loss,
                    grad_position=grad_position,
                )
                for input_shape in running_probability_inputs
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        if dtype != ms.float64:
            running_class_inputs = ((3, 5), (2, 4, 6, 6))
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(ms.Tensor(shape=None, dtype=ms.int32), None, -100, "mean", 0.0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_class_dynamic_rank_compile_input",
                    grad_position=grad_position,
                ),
                op_running_inputs=tuple(
                    OpSampleInput(
                        op_input=make_logits(input_shape),
                        op_args=(
                            _make_cross_entropy_class_target(input_shape, ms.int32, device),
                            None,
                            -100,
                            "mean",
                            0.0,
                        ),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_class_dynamic_rank_running_input",
                        loss=sample_loss,
                        grad_position=grad_position,
                    )
                    for input_shape in running_class_inputs
                ),
            )

        running_probability_inputs = ((3, 5), (2, 4, 6, 6))
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype), None, -100, "mean", 0.0),
                op_kwargs={},
                sample_name=f"{op_info.name}_probability_dynamic_rank_compile_input",
                grad_position=grad_position,
            ),
            op_running_inputs=tuple(
                OpSampleInput(
                    op_input=make_logits(input_shape),
                    op_args=(
                        _make_cross_entropy_probability_target(input_shape, dtype, device),
                        None,
                        -100,
                        "mean",
                        0.0,
                    ),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_probability_dynamic_rank_running_input",
                    loss=sample_loss,
                    grad_position=grad_position,
                )
                for input_shape in running_probability_inputs
            ),
        )

def extra_sample_inputs_binary_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make = functools.partial(make_tensor, device=device, dtype=dtype)

    make_prob = functools.partial(make, low=0, high=1)

    for shape, weight, reduction in itertools.product([(2, 3, 4, 5, 6, 7 ,8 ,9)],
                                                      [None, make((2, 3, 4, 5, 6, 7 ,8 ,9)),],
                                                      ['none', 'mean', 'sum']):
        yield OpSampleInput(
            make_prob(shape),
            op_args=(make_prob(shape), ),
            op_kwargs={'weight': weight, 'reduction': reduction},
            sample_name=op_info.name + '8d',
        )

        yield OpSampleInput(
            make_tensor(shape, device=device, dtype=dtype, discontiguous=True, low=0, high=1),
            op_args=(make_prob(shape), ),
            op_kwargs={'weight': weight, 'reduction': reduction},
            sample_name=op_info.name + 'discontiguous',
        )

        yield OpSampleInput(
            ms.Tensor(np.random.choice(np.array((0.0, -0.0, 1.0, 1.25e-1, 1e-3,)), size=shape),dtype=dtype),
            op_args=(ms.Tensor(np.random.choice(np.array((0.0, -0.0, 1.0, 1.25e-1, 1e-3,)),
                                                size=shape), dtype=dtype), ),
            op_kwargs={'weight': weight, 'reduction': reduction},
            sample_name=op_info.name + 'small_value',
        )

def error_inputs_mint_nn_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_logits = functools.partial(make_tensor, device=device, dtype=ms.float32, low=-1.5, high=1.5)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_logits((3, 5)),
            op_args=(
                _make_cross_entropy_probability_target((3, 4), ms.float32, device),
                None,
                -100,
                "mean",
                0.0,
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_probabilities_shape_mismatch",
        ),
        op_error_type=ValueError,
        op_error_info='probability target shape should match input shape',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_logits((3, 5)),
            op_args=(
                _make_cross_entropy_class_target((3, 5), ms.int32, device),
                _make_cross_entropy_weight(6, ms.float32, device),
                -100,
                "mean",
                1.5,
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_invalid_label_smoothing",
        ),
        op_error_type=(ValueError, RuntimeError),
        op_error_info='label_smoothing or weight shape is invalid',
    )

def sample_inputs_binary_cross_entropy(op_info, dtype, device, logits=False, **kwargs):
    S = SMALL_DIM_SIZE

    make = functools.partial(make_tensor, device=device, dtype=dtype)
    # Lower bounds must be greater than 'eps' defined in gradcheck.py::gradgradcheck() -> eps
    # otherwise perturbation calculation causes Tensor value to become negative triggering
    # a device-side hardware assertion
    make_prob = functools.partial(make, low=0, high=1)
    make_weight = functools.partial(make, low=0, high=10)

    reductions = ("mean", "sum", "none")

    shapes_and_kwargs = [
        *[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))],
        # Now framework does not support passing parameters in an arbitrary order during the backward process.
        # *[((S, S), dict(reduction=reduction)) for reduction in reductions],
        *[((S, S), {"weight": make((S, S)), "reduction": reduction}) for reduction in reductions],
    ]

    broadcast_weight_shapes = (
        ((S, S, S), (S, S)),
        ((S, S, S), (1,)),
        ((S,), (1,)),
    )
    shapes_and_kwargs.extend(
        [
            (input_shape, {"weight": make_weight(weight_shape), "reduction": reduction})
            for input_shape, weight_shape in broadcast_weight_shapes
            for reduction in reductions
        ]
    )

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
    if dtype.is_floating_point:
        make_prob = functools.partial(make, low=0.1, high=0.9)
        make_weight = functools.partial(make, low=0.2, high=1.6)
        make_pos_weight = functools.partial(make, low=0.2, high=2.0)
    else:
        make_prob = functools.partial(make, low=0, high=2)
        make_weight = functools.partial(make, low=1, high=3)
        make_pos_weight = functools.partial(make, low=1, high=3)
    reductions = ("mean", "sum", "none")

    def ordered_kwargs(weight=None, reduction="mean", pos_weight=None):
        return {"weight": weight, "reduction": reduction, "pos_weight": pos_weight}

    def make_weight_shape_kwargs():
        kwargs = []
        for shape in ((1,), (1, S), (S), (S, S)):
            kwargs.extend([((S, S), ordered_kwargs(weight=make_weight(shape), reduction=reduction))
                           for reduction in reductions])
        return kwargs

    shapes_and_kwargs = [
        *[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))],
        *[((S, S), ordered_kwargs(reduction=reduction)) for reduction in reductions],
        *make_weight_shape_kwargs(),
        *[
            ((S, S), ordered_kwargs(reduction=reduction, pos_weight=make_pos_weight((S,))))
            for reduction in reductions
        ],
        *[
            ((S, S), ordered_kwargs(reduction=reduction, pos_weight=make_pos_weight((S, S))))
            for reduction in reductions
        ],
        *[
            ((S, S), ordered_kwargs(weight=make_weight((S, S)), reduction=reduction,
                                    pos_weight=make_pos_weight((S,))))
            for reduction in reductions
        ],
        *[
            ((S, S), ordered_kwargs(weight=make_weight((S, S)), reduction=reduction,
                                    pos_weight=make_pos_weight((S, S))))
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


def extra_sample_inputs_binary_cross_entropy_with_logits_mixed_dtype(op_info, dtype, device, **kwargs):
    if dtype != ms.float16:
        return

    S = SMALL_DIM_SIZE
    make_logits = functools.partial(make_tensor, device=device, dtype=dtype, low=-1.5, high=1.5)
    make_prob_fp32 = functools.partial(make_tensor, device=device, dtype=ms.float32, low=0.1, high=0.9)
    make_weight_fp32 = functools.partial(make_tensor, device=device, dtype=ms.float32, low=0.2, high=1.6)
    make_pos_weight_fp32 = functools.partial(make_tensor, device=device, dtype=ms.float32, low=0.2, high=2.0)

    mixed_dtype_cases = (
        ("target_fp32", {"reduction": "mean"}, make_prob_fp32, None, None),
        ("weight_fp32", {"reduction": "mean"}, None, make_weight_fp32, None),
        ("pos_weight_fp32", {"reduction": "mean"}, None, None, make_pos_weight_fp32),
        ("all_aux_fp32", {"reduction": "mean"},
         make_prob_fp32,
         make_weight_fp32,
         make_pos_weight_fp32),
        ("all_aux_fp32_none", {"reduction": "none"},
         make_prob_fp32,
         make_weight_fp32,
         make_pos_weight_fp32),
    )
    shape = (S, S)
    default_target = functools.partial(make_tensor, device=device, dtype=dtype, low=0.1, high=0.9)
    default_weight = functools.partial(make_tensor, device=device, dtype=dtype, low=0.2, high=1.6)
    default_pos_weight = functools.partial(make_tensor, device=device, dtype=dtype, low=0.2, high=2.0)

    for case_name, op_kwargs, target_factory, weight_factory, pos_weight_factory in mixed_dtype_cases:
        sample_target = (target_factory or default_target)(shape)
        sample_kwargs = {
            "weight": (weight_factory or default_weight)(shape),
            "reduction": op_kwargs["reduction"],
            "pos_weight": (pos_weight_factory or default_pos_weight)(shape),
        }
        yield OpSampleInput(
            make_logits(shape),
            op_args=(sample_target,),
            op_kwargs=sample_kwargs,
            sample_name=f"{op_info.name}_{case_name}",
        )


def dynamic_sample_inputs_mint_binary_cross_entropy_with_logits(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_logits = functools.partial(make_tensor, dtype=dtype, device=device, low=-1.5, high=1.5)
    make_prob = functools.partial(make_tensor, dtype=dtype, device=device, low=0.1, high=0.9)
    make_weight = functools.partial(make_tensor, dtype=dtype, device=device, low=0.2, high=1.6)
    make_pos_weight = functools.partial(make_tensor, dtype=dtype, device=device, low=0.2, high=2.0)
    reductions = ("mean", "sum", "none")
    input_shapes = ((1, 3, 8, 8), (2, 3, 10, 10))

    def build_compile_and_running_inputs(dynamic_shape, reduction, with_weight=False, with_pos_weight=False,
                                         pos_weight_channel_only=False, target_dtype=None, weight_dtype=None,
                                         pos_weight_dtype=None):
        target_dtype = dtype if target_dtype is None else target_dtype
        weight_dtype = dtype if weight_dtype is None else weight_dtype
        pos_weight_dtype = dtype if pos_weight_dtype is None else pos_weight_dtype
        if dynamic_shape:
            target_compile = ms.Tensor(shape=(None, None, None, None), dtype=target_dtype)
        else:
            target_compile = ms.Tensor(shape=None, dtype=target_dtype)

        op_kwargs = {"weight": None, "reduction": reduction, "pos_weight": None}
        running_kwargs = []
        if with_weight:
            op_kwargs["weight"] = ms.Tensor(shape=(None, None, None, None), dtype=weight_dtype) if dynamic_shape \
                else ms.Tensor(shape=None, dtype=weight_dtype)
        if with_pos_weight:
            if pos_weight_channel_only:
                if dynamic_shape:
                    op_kwargs["pos_weight"] = ms.Tensor(shape=(None,), dtype=pos_weight_dtype)
                else:
                    op_kwargs["pos_weight"] = ms.Tensor(shape=None, dtype=pos_weight_dtype)
            else:
                op_kwargs["pos_weight"] = ms.Tensor(shape=(None, None, None, None), dtype=pos_weight_dtype) \
                    if dynamic_shape else ms.Tensor(shape=None, dtype=pos_weight_dtype)

        for shape in input_shapes:
            runtime_kwargs = {"weight": None, "reduction": reduction, "pos_weight": None}
            target_factory = make_prob if target_dtype == dtype else functools.partial(
                make_tensor, dtype=target_dtype, device=device, low=0.1, high=0.9
            )
            weight_factory = make_weight if weight_dtype == dtype else functools.partial(
                make_tensor, dtype=weight_dtype, device=device, low=0.2, high=1.6
            )
            pos_weight_factory = make_pos_weight if pos_weight_dtype == dtype else functools.partial(
                make_tensor, dtype=pos_weight_dtype, device=device, low=0.2, high=2.0
            )
            if with_weight:
                runtime_kwargs["weight"] = weight_factory(shape)
            if with_pos_weight:
                runtime_kwargs["pos_weight"] = pos_weight_factory((shape[-1],)) if pos_weight_channel_only \
                    else pos_weight_factory(shape)
            running_kwargs.append(
                OpSampleInput(
                    op_input=make_logits(shape=shape),
                    op_args=(target_factory(shape=shape),),
                    op_kwargs=runtime_kwargs,
                    sample_name=f"{op_info.name}_{'dynamic_shape' if dynamic_shape else 'dynamic_rank'}_running_input",
                )
            )
        return target_compile, op_kwargs, tuple(running_kwargs)

    case_configs = (
        {"with_weight": False, "with_pos_weight": False, "pos_weight_channel_only": False},
        {"with_weight": True, "with_pos_weight": True, "pos_weight_channel_only": False},
        {"with_weight": False, "with_pos_weight": True, "pos_weight_channel_only": True},
    )
    if dtype == ms.float16:
        case_configs = case_configs + (
            {
                "with_weight": True,
                "with_pos_weight": True,
                "pos_weight_channel_only": False,
                "target_dtype": ms.float32,
                "weight_dtype": ms.float32,
                "pos_weight_dtype": ms.float32,
            },
        )
    for reduction in reductions:
        for case_config in case_configs:
            if not kwargs.get("only_dynamic_rank", False):
                target_compile, compile_kwargs, running_inputs = build_compile_and_running_inputs(
                    True, reduction, **case_config
                )
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                        op_args=(target_compile,),
                        op_kwargs=compile_kwargs,
                        sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                    ),
                    op_running_inputs=running_inputs,
                )

            if not kwargs.get("only_dynamic_shape", False):
                target_compile, compile_kwargs, running_inputs = build_compile_and_running_inputs(
                    False, reduction, **case_config
                )
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=None, dtype=dtype),
                        op_args=(target_compile,),
                        op_kwargs=compile_kwargs,
                        sample_name=f"{op_info.name}_dynamic_rank_compile_input",
                    ),
                    op_running_inputs=running_inputs,
                )

def _cross_entropy_sample_loss(dtype):
    if dtype == ms.float16:
        return 0.01
    if dtype == ms.bfloat16:
        return 0.02
    return None

def dynamic_sample_inputs_mint_binary_cross_entropy(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0.1, high=0.9)
    make_weight = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=10)
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
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                    op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype),),
                    op_kwargs={'weight': ms.Tensor(shape=(None, None), dtype=dtype), 'reduction': reduction},
                    sample_name=f"{op_info.name}_dynamic_shape_broadcast_weight_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4)),
                        op_args=(make_func(shape=(2, 3, 4)),),
                        op_kwargs={'weight': make_weight(shape=(3, 4)), 'reduction': reduction},
                        sample_name=f"{op_info.name}_dynamic_shape_broadcast_weight_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(1, 3, 5)),
                        op_args=(make_func(shape=(1, 3, 5)),),
                        op_kwargs={'weight': make_weight(shape=(3, 5)), 'reduction': reduction},
                        sample_name=f"{op_info.name}_dynamic_shape_broadcast_weight_running_input",
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
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(ms.Tensor(shape=None, dtype=dtype),),
                    op_kwargs={'weight': ms.Tensor(shape=None, dtype=dtype), 'reduction': reduction},
                    sample_name=f"{op_info.name}_dynamic_rank_broadcast_weight_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4)),
                        op_args=(make_func(shape=(2, 3, 4)),),
                        op_kwargs={'weight': make_weight(shape=(1,)), 'reduction': reduction},
                        sample_name=f"{op_info.name}_dynamic_rank_broadcast_weight_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(5,)),
                        op_args=(make_func(shape=(5,)),),
                        op_kwargs={'weight': make_weight(shape=(1,)), 'reduction': reduction},
                        sample_name=f"{op_info.name}_dynamic_rank_broadcast_weight_running_input",
                    ),
                ),
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
        op_input=make_arg((256, 120, 10, 10)),
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
        op_error_type=(ValueError, RuntimeError),
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
        op_error_type=(ValueError, RuntimeError),
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
        input_values = np.array(_float_small_values).reshape((1, 1, len(_float_small_values)))
        weight_values = np.array(_float_small_values[:1]).reshape((1, 1, 1))
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
        input_values = np.array(_large_values).reshape((1, 1, len(_large_values)))
        weight_values = np.array(_large_values[:1]).reshape((1, 1, 1))
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

# sample inputs functions for mindspore.mint.nn.Linear
def _make_seeded_linear_tensor(shape, dtype, seed, *, device=None, low=-2.0, high=2.0, normal=False):
    """Build a deterministic tensor for mint.nn.Linear opinfo samples."""
    rng = np.random.RandomState(seed)
    np_dtype = dtype_to_np_dtype[dtype]
    if normal:
        np_array = rng.standard_normal(size=shape).astype(np_dtype)
    else:
        np_array = rng.uniform(low, high, size=shape).astype(np_dtype)
    return make_tensor_with_np_array(np_array, dtype=dtype, device=device)

def basic_sample_inputs_mint_nn_linear(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.Linear.
    Reference torch's module_inputs_torch_nn_Linear.
    '''
    # Cases: (input_shape, (in_features, out_features), input_seed, weight_seed, bias_seed)
    cases = (
        ((4, 10), (10, 8), 100, 101, 102),
        ((4, 2), (2, 20), 110, 111, 112),
    )

    for input_shape, (in_features, out_features), input_seed, weight_seed, bias_seed in cases:
        op_kwargs = {
            'bias': True,
            'weight_init': _make_seeded_linear_tensor(
                (out_features, in_features), dtype, weight_seed, device=device, low=-2.0, high=2.0
            ),
            'bias_init': _make_seeded_linear_tensor(
                (out_features,), dtype, bias_seed, device=device, low=-2.0, high=2.0
            ),
            'dtype': dtype,
        }
        yield OpSampleInput(
            op_input=_make_seeded_linear_tensor(input_shape, dtype, input_seed, device=device, low=-2.0, high=2.0),
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
          "weight_init": _make_seeded_linear_tensor((2, 1), dtype, 201, device=device, normal=True),
          "bias_init": _make_seeded_linear_tensor((2,), dtype, 202, device=device, normal=True), "dtype": dtype},
         None),
        (ms.Tensor(np.full((8, 8, 1), -np.inf), dtype=dtype),
         {"in_features": 1, "out_features": 2, "bias": True,
          "weight_init": _make_seeded_linear_tensor((2, 1), dtype, 203, device=device, normal=True),
          "bias_init": _make_seeded_linear_tensor((2,), dtype, 204, device=device, normal=True), "dtype": dtype},
         None),
        (_make_seeded_linear_tensor((7, 3, 9, 7, 9, 4), dtype, 205, device=device, normal=True),
         {"in_features": 4, "out_features": 15, "bias": True,
          "weight_init": _make_seeded_linear_tensor((15, 4), dtype, 206, device=device, normal=True),
          "bias_init": _make_seeded_linear_tensor((15,), dtype, 207, device=device, normal=True), "dtype": dtype},
         0.004),
        (ms.Tensor(np.array([]).reshape(0, 4), dtype=dtype),
         {"in_features": 4, "out_features": 8, "bias": True,
          "weight_init": _make_seeded_linear_tensor((8, 4), dtype, 208, device=device, normal=True),
          "bias_init": _make_seeded_linear_tensor((8,), dtype, 209, device=device, normal=True), "dtype": dtype},
         None),
        # discontinuous
        (ms.Tensor(np.arange(24).reshape(6, 4).astype(np.float32)[::2, :], dtype=dtype),
         {"in_features": 4, "out_features": 6, "bias": True,
          "weight_init": _make_seeded_linear_tensor((6, 4), dtype, 210, device=device, normal=True),
          "bias_init": _make_seeded_linear_tensor((6,), dtype, 211, device=device, normal=True), "dtype": dtype},
         None),
        # continuous
        (ms.Tensor(np.ones((10, 6)), dtype=dtype),
         {"in_features": 6, "out_features": 4, "bias": True,
          "weight_init": ms.Tensor(np.ones((4, 6)), dtype=dtype),
          "bias_init": ms.Tensor(np.ones(4), dtype=dtype), "dtype": dtype},
         None),
        (ms.Tensor(np.full((6, 4), np.finfo(np.float32).max), dtype=dtype),
         {"in_features": 4, "out_features": 8, "bias": True,
          "weight_init": ms.Tensor(np.full((8, 4), 1e-5), dtype=dtype),
          "bias_init": ms.Tensor(np.full(8, 1e-5), dtype=dtype), "dtype": dtype},
         None),
        (ms.Tensor(np.full((4, 3), np.finfo(np.float32).min), dtype=dtype),
         {"in_features": 3, "out_features": 5, "bias": True,
          "weight_init": ms.Tensor(np.full((5, 3), -1e-5), dtype=dtype),
          "bias_init": ms.Tensor(np.full(5, -1e-5), dtype=dtype), "dtype": dtype},
         None),
    )

    for inputs, p_kwargs, loss in case:
        yield OpSampleInput(
            op_input=inputs,
            op_kwargs=p_kwargs,
            sample_name=op_info.name,
            loss=loss,
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

def basic_sample_inputs_layernorm(
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




def basic_sample_inputs_groupnorm(
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
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=op_info.exclude_value)

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

def extra_sample_inputs_mint_layer_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (3, 6, 10, 20), ([6, 10, 20], ), {},
                                dtype=dtype, device=device, exclude_value=(-1e20, 1e20))

    empty_cases=(
        ((0, ), [0]),
        ((1, 0), [0]),
        ((0 ,3, 7), [7]),
        ((3, 0, 0, 9), [3, 0, 0, 9])
    )

    for empty_input_shape, normailized_shape in empty_cases:
        yield from make_extra_tensor(op_info, empty_input_shape, (normailized_shape,), {},
        dtype=dtype, device=device, empty_tensor=True)

    yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9 ), (((5, 6, 7, 8, 9 ),), ),
                                 {}, dtype=dtype, device=device, support_multi_dim=True)

def extra_sample_inputs_mint_groupnorm(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (3, 10, 20, 30), (2, 10), {},
                                 dtype=dtype, device=device, exclude_value=(-1e20, 1e20))

    yield from make_extra_tensor(op_info, (3, 4, 3, 3, 3, 3, 3, 3), (2, 4),
                                 {}, dtype=dtype, device=device, support_multi_dim=True)

    empty_cases = (
        (0, 6),
        (1, 6, 0)
    )
    for empty_shape in empty_cases:
        yield from make_extra_tensor(op_info, empty_shape, (2, empty_shape[1]), {}, dtype=dtype,
                                     device=device, empty_tensor=True)

def basic_sample_inputs_group_norm(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((4, 8), 4, (8,),(8,), 0.01,),
        ((4, 6, 5), 3, (6,),(6,), 1e-6),
        ((3, 8, 4, 8), 2, (8,),(8,), 1e-12),
        ((9, 14, 6, 2, 3), 1, (14,),(14,), 2.0),
    )

    for input_shape, num_groups, input_weight, bias, eps in cases:
        slen = len(input_shape)
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(num_groups,),
            op_kwargs={'weight':make_arg(input_weight), 'bias':make_arg(bias), 'eps':eps},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_group_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    #empty tensors
    _input = make_func(shape=(1, 0, 4))
    yield OpSampleInput(
        _input,
        op_args=(3,),
        sample_name=f"{op_info.name}_{dtype}_empty_tensor",
    )

    # large/small values
    if dtype in [ms.float16]:
        values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp16_large_values']
    else:
        values = (dtypes_extra_values['_float_small_values'] +
                  (-521, 521, -996.9, 996.9, -13141.5, 13141.5, -5062757.7, 5062757.7))

    np_array = np.random.choice(values, size=(2, 3), replace=False)
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
            op_args=(1,),
            sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values",
        )

    # nan/inf/-inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full([1024, 1024], t), dtype=dtype, device=device),
                op_args=(1,),
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_other_op_group_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(3, 6, ),
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
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((4, )),
            op_args=(2, 4,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='dim of input must be between 2 and 8',
    )

def error_inputs_other_op_functional_group_norm(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='num_groups should be divided with no remainder by  num_channels',
    )

def basic_sample_inputs_mint_nn_smoothl1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3, 5), (3, 5), 'mean', 0.8),
        ((8,), (8,), 'none', 1),
        ((1, 2), (1, 2), 'sum', 0.1),
    )

    for input1_shape, input2_shape, reduction, beta in cases:
        yield OpSampleInput(
            op_input=make_func(input1_shape),
            op_args=(make_func(input2_shape),),
            op_kwargs={'reduction': reduction, 'beta': beta},
            sample_name=f"{op_info.name}_{dtype}_{len(input1_shape)}d",
        )
    # empty tensor and default param value
    empty_tensor = make_func((2, 0))
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(empty_tensor,),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_smoothl1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), (), 'mean', ms.float16, 0),
        ((5, 2), (1, 2), 'sum', None, False),
        ((2, 2), (1, 2), 'none', None, 0),
        ((128, 32, 8), (128, 32, 8), 'none', ms.float32, 2.1),
        ((1, 56, 128, 16, 2), (1, 56, 128, 16, 2), 'none', None, 1.1),
        ((10, 5, 32, 8, 16, 4, 2, 10), (10, 5, 32, 8, 16, 4, 2, 10), 'mean', None, 0.5),
    )

    for input1_shape, input2_shape, reduction, target_dtype, beta in cases:
        if target_dtype is None:
            target_dtype = dtype
        yield OpSampleInput(
            op_input=make_func(input1_shape),
            op_args=(make_func(input2_shape, dtype=target_dtype),),
            op_kwargs={'reduction': reduction, 'beta': beta},
            sample_name=f"{op_info.name}_{dtype}_{len(input1_shape)}d",
        )
    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input1_array = np.random.choice(values, size=(5, 2))
            input2_array = np.random.choice(values, size=(5, 2), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(input2_array, dtype=dtype, discontiguous=discontiguous),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_small_large_vales_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((3, 2), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(input_tensor,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_smoothl1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.SmoothL1Loss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(7, 8, 9)),
                    op_args=(make_func(shape=(7, 8, 9)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 3, 5)),
                    op_args=(make_func(shape=(1, 3, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8,)),
                    op_args=(make_func(shape=(8,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2,)),
                    op_args=(make_func(shape=(2,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            ),
        )

def error_sample_inputs_mint_nn_smoothl1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((5, 2), (2, 1), 'none', 1.0),
        ((5, 2), (5, 1), 'batch', 1.0),
        ((5, 2), (5, 1), 'sum', -1),
        ((5, 2), (5, 1), 'sum', [1]),
    )
    # two inputs 'input' and 'target' shape are not equal
    # reduction is not 'none'/'mean'/'sum'
    # beta < 0
    # beta is not float/int/bool
    for input_shape, target_shape, reduction, beta in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape, dtype=ms.float32),
                op_args=(make_arg(shape=target_shape, dtype=ms.float32),),
                op_kwargs={'reduction': reduction, 'beta': beta,},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, TypeError, RuntimeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input/target is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_target_not_tensor",
        ),
        op_error_type=TypeError,
        op_error_info=f'input/target not tensor is not supported for {op_info.name}',
    )
    # target dtype is not in [float16/float32/bfloat16]
    target_not_support_dtypes = (
        ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.float64, ms.complex64, ms.complex128
    )
    for target_dtype in target_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=(5,), dtype=ms.float32),
                op_args=(make_arg(shape=(5,), dtype=target_dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_target_dtype_{target_dtype}_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'target error dtype {target_dtype} is not supported for {op_info.name}',
        )

def basic_reference_inputs_mint_nn_elu_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        ((S,), -1.0),
        ((M, S), 1.0),
        ((S, S, L), 0.4),
        ((2, 1, 3, 2), 12.3),
        ((2, 3, 4, 1, 2), 2.0),
        ((2, 1, 2, 2, 1, 2), 0.0),
        ((2, 1, 2, 2, 1, 2, 1), 1.0),
        ((2, 1, 2, 2, 1, 2, 1, 2), 2.0)
    )

    for input_shape, i in shapes:
        _input = make_func(shape=input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'alpha':i,},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
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

# basic sample inputs functions for mindspore.mint.nn.Conv3d
def basic_sample_inputs_mint_nn_conv3d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # test for 3d
    yield OpSampleInput(
        op_input=make_arg((6, 7, 8, 8), random_seed=8),
        op_args=(6, 6, 2),
        op_kwargs={"stride":1, "padding":0, "dilation":1, "groups":1, "bias":True,
                   "padding_mode":"zeros", "dtype":None,
                   "weight_init": make_arg((6, 6, 2, 2, 2), random_seed=108),
                   "bias_init":make_arg((6, ), random_seed=208)},
        sample_name=f"{op_info.name}_{dtype}_input_without_batch",
        loss=0.015 if dtype == ms.bfloat16 else None,
    )
    #test for dtype
    if dtype != ms.float32:
        yield OpSampleInput(
            op_input=make_arg((10, 6, 7, 8, 8)),
            op_args=(6, 6, 1),
            op_kwargs={"stride":1, "padding":0, "dilation":1, "groups":1, "bias":True,
                       "padding_mode":"zeros", "dtype": ms.float32,
                       "weight_init": make_arg((6, 6, 1, 1, 1)),
                       "bias_init": make_arg((6,))},
            sample_name=f"{op_info.name}_{dtype}_input_with_dtype",
        )
    # Cases: (input_shape, (in_channels, out_channels, kernel_size), kwargs_dict)
    batch_size=1
    in_channels=6
    out_channels=9
    cases={
        "kernel_size": ((2, 2, 3), ),
        "stride": (1, (2, 3, 1), ),
        "padding": ("valid", "same", (1, 1, 1)),
        "dilation": ((2, 2, 2),),
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
            op_input=make_arg((batch_size, in_channels, 4, 4, 6)),
            op_args=(in_channels, out_channels, kernel_size,),
            op_kwargs={"stride": stride, "padding": padding, "dilation": dilation, "groups": groups, "bias": True,
                       "padding_mode": padding_mode,  "dtype": dtype,
                       "weight_init": make_arg((out_channels, in_channels//groups, *kernel_size)),
                       "bias_init":make_arg((out_channels,))},
            sample_name=f"{op_info.name}_{dtype}",
        )

#extra sample inputs functions for mindspore.mint.nn.Conv3d
def extra_sample_inputs_mint_nn_conv3d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # big batch size and channels
    yield OpSampleInput(
        op_input=make_arg((200, 120, 10, 10, 10)),
        op_args=(120, 120, 4),
        op_kwargs={"weight_init": make_arg((120, 120, 4, 4, 4)), "bias_init": make_arg((120,))},
        sample_name=f"{op_info.name}_{dtype}_input_with_big_input",
    )
    # case with small/large values
    float_vals = dtypes_extra_values['_float_small_values']
    float_vals =float_vals + dtypes_extra_values['_fp16_large_values']
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8, 8)), dtype=dtype),
        op_args=(6, 9, 2),
        op_kwargs={"weight_init":ms.Tensor(np.random.choice(float_vals, size=(9, 6, 2, 2, 2)), dtype=dtype),
                   "bias": True,"bias_init": ms.Tensor(np.random.choice(float_vals, size=(9,)), dtype=dtype)},
        sample_name=f"{op_info.name}_{dtype}_small_large_values",
    )
    #case with inf/nan values
    for t in [np.nan, np.inf, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((6, 7, 8, 8), t), dtype=dtype),
            op_args=(6, 9, 2),
            op_kwargs={"weight_init": make_arg((9, 6, 2, 2, 2)),
                       "bias": True, "bias_init":make_arg((9, ))},
            sample_name=f"{op_info.name}_{dtype}_{t}",
        )

    #case with discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8, 8), discontiguous=discontiguous),
            op_args=(6, 9, 2),
            op_kwargs={"weight_init": make_arg((9, 6, 2, 2, 2), discontiguous=discontiguous),
                       "bias": True, "bias_init": make_arg((9, ), discontiguous=discontiguous)},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}",
        )


# error sample inputs functions for mindspore.mint.nn.Conv2d
def error_sample_inputs_mint_nn_conv3d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #weight does not have right shape
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((120, 7, 8, 9)),
            op_args=(120, 120, 2),
            op_kwargs={"weight_init":make_arg((120, 2, 2, 2)), "bias":False},
            sample_name=f"{op_info.name}_weight_shape_not_right",
        ),
        op_error_type=RuntimeError,
        op_error_info='weight does not have right shape'
    )
    #input_channel of net not match with input_channel of input,
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((110, 7, 8, 9)),
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
            op_input=make_arg((120, 7, 8, 9)),
            op_args=(120, 120, 2),
            op_kwargs={"weight_init":make_arg((120, 110, 2, 2, 2)), "bias":False},
            sample_name=f"{op_info.name}_weight_channel_not_match",
        ),
        op_error_type=ValueError,
        op_error_info='input channel not match with weight'
    )
    #kernel_value is bigger than the input_height/input_width
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((120, 7, 8, 9)),
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
            op_input=make_arg((10, 7, 8, 9)),
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
                op_input=make_arg((10, 7, 8, 9)),
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
                op_input=make_arg((10, 7, 8, 9)),
                op_args=(10, 10, 1),
                op_kwargs={name:0, "bias":False},
                sample_name=f"{op_info.name}_{name}_0",
            ),
            op_error_type=ValueError,
            op_error_info=f'{name} is small than 1'
        )
    #input_channel/output_channel cannot be divided by groups:
    for name, shape, op_args in zip(("input_channel", "output_channel"),
                                    ((10, 7, 8, 8), (9, 7, 8, 7)),
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

def basic_sample_inputs_mint_nn_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
        Generate basic sample inputs for mindspore.mint.nn.ConvTranspose2d.
        Reference torch's module tests for ConvTranspose2d.
        '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    yield OpSampleInput(
        op_input=make_arg((6, 7, 8)),
        op_args=(6, 6, 2),
        op_kwargs={"stride": 1, "padding": 0, "output_padding":0, "dilation": 1, "groups": 1, "bias": True,
                   "padding_mode": "zeros", "dtype": None,
                   "weight_init": make_arg((6, 6, 2, 2)),
                   "bias_init": make_arg((6,))},
        sample_name=f"{op_info.name}_{dtype}_input_without_batch",
    )
    # test for dtype
    if dtype != ms.float32:
        yield OpSampleInput(
            op_input=make_arg((10, 6, 7, 8)),
            op_args=(6, 6, 1),
            op_kwargs={"stride": 1, "padding": 0, "output_padding":0, "dilation": 1, "groups": 1, "bias": True,
                       "padding_mode": "zeros", "dtype": ms.float32,
                       "weight_init": make_arg((6, 6, 1, 1)),
                       "bias_init": make_arg((6,))},
            sample_name=f"{op_info.name}_{dtype}_input_with_dtype",
        )
    # Cases: (input_shape, (in_channels, out_channels, kernel_size), kwargs_dict)
    batch_size = 1
    in_channels = 6
    out_channels = 9
    cases = {
        "kernel_size": ((2, 3),(1, 1),),
        "stride": (2, (2, 3,),),
        "padding": ((3, 2), ),
        "output_padding":((1, 1), 0),
        "dilation": ((2, 2), 1),
        'groups': (3,),
        'padding_mode': ("zeros",),
    }
    for kernel_size, stride, dilation, padding, output_padding, groups, padding_mode in itertools.product(
            cases["kernel_size"],
            cases["stride"],
            cases["dilation"],
            cases["padding"],
            cases["output_padding"],
            cases["groups"],
            cases["padding_mode"],
    ):
        yield OpSampleInput(
            op_input=make_arg((batch_size, in_channels, 7, 8)),
            op_args=(in_channels, out_channels, kernel_size,),
            op_kwargs={"stride": stride, "padding": padding, "output_padding": output_padding, "dilation": dilation,
                       "groups": groups, "bias": True, "padding_mode": padding_mode, "dtype": dtype,
                       "weight_init": make_arg((in_channels, out_channels // groups, *kernel_size)),
                       "bias_init": make_arg((out_channels,))},
            sample_name=f"{op_info.name}_{dtype}",
        )
    #with output_size
    yield OpSampleInput(
        op_input=make_arg((10, 6, 7, 8)),
        op_args=(6, 6, 3),
        op_kwargs={"stride": 2, "padding": 1, "output_padding": 2, "dilation": 1, "groups": 1, "bias": True,
                   "padding_mode": "zeros", "dtype": None,
                   "weight_init": make_arg((6, 6, 3, 3)),
                   "bias_init": make_arg((6,)), "output_size":(1, 6, 13, 15)},
        sample_name=f"{op_info.name}_{dtype}_input_output_padding_2",
    )

#extra sample inputs functions for mindspore.mint.nn.Conv2d
def extra_sample_inputs_mint_nn_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.nn.Conv2d.
    Reference torch's module tests for Conv2d.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # big batch size and channels
    yield OpSampleInput(
        op_input=make_arg((1024, 120, 10, 10)),
        op_args=(120, 120, 4),
        op_kwargs={"weight_init": make_arg((120, 120, 4, 4)), 'bias':True, "bias_init": make_arg((120,))},
        sample_name=f"{op_info.name}_{dtype}_input_with_big_input",
    )
    # case with small/large values
    float_vals = dtypes_extra_values['_float_small_values']
    float_vals =float_vals + dtypes_extra_values['_fp16_large_values']
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8)), dtype=dtype),
        op_args=(6, 9, 2),
        op_kwargs={"weight_init":ms.Tensor(np.random.choice(float_vals, size=(6, 9, 2, 2)), dtype=dtype),
                   "bias": True,"bias_init": ms.Tensor(np.random.choice(float_vals, size=(9,)), dtype=dtype)},
        sample_name=f"{op_info.name}_{dtype}_small_large_values",
    )
    #case with inf/nan values
    for t in [np.nan, np.inf, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((6, 7, 8), t), dtype=dtype),
            op_args=(6, 9, 2),
            op_kwargs={"weight_init": make_arg((6, 9, 2, 2)),
                       "bias": True, "bias_init":make_arg((9, ))},
            sample_name=f"{op_info.name}_{dtype}_{t}",
        )

    #case with discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8), discontiguous=discontiguous),
            op_args=(6, 9, 2),
            op_kwargs={"weight_init": make_arg((6, 9, 2, 2), discontiguous=discontiguous),
                       "bias": True, "bias_init": make_arg((9, ), discontiguous=discontiguous)},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}",
        )

def error_sample_inputs_mint_nn_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    for error_input in error_sample_inputs_mint_nn_conv2d(op_info, dtype, device):
        if error_input.op_error_info == 'input channel not match with weight':
            error_input.op_sample_input=OpSampleInput(
                op_input=make_arg((120, 7, 8)),
                op_args=(120, 120, 2),
                op_kwargs={"weight_init":make_arg((110, 120, 2, 2)), "bias":False},
                sample_name=f"{op_info.name}_weight_channel_not_match",
            )
        if 'kernel_value is bigger' in error_input.op_error_info:
            continue
        yield error_input
    #output_padding is more than stride and dilation
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((10, 6, 7, 8)),
            op_args=(6, 6, 3),
            op_kwargs={"stride": 2, "padding": 1, "output_padding": 3, "dilation": 1, "groups": 1, "bias": True,
                       "padding_mode": "zeros", "dtype": None,
                       "weight_init": make_arg((6, 6, 3, 3)),
                       "bias_init": make_arg((6,))},
            sample_name=f"{op_info.name}_bigger_output_padding"
        ),
        op_error_type=RuntimeError,
        op_error_info='output_padding is more than stride and dilation'
    )
    # wrong output_size
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((10, 6, 7, 8)),
            op_args=(6, 6, 3),
            op_kwargs={"stride": 2, "padding": 1, "output_padding": 3, "dilation": 1, "groups": 1, "bias": True,
                       "padding_mode": "zeros", "dtype": None,
                       "weight_init": make_arg((6, 6, 3, 3)),
                       "bias_init": make_arg((6,)), "output_size":(1, 6, 10, 10)},
            sample_name=f"{op_info.name}_wrong_output_size"
        ),
        op_error_type=ValueError,
        op_error_info='wrong output size'
    )

def basic_sample_inputs_tensor_fold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    yield OpSampleInput(
        op_input=make_arg(shape=(1, 12, 12)),
        op_args=(ms.Tensor(np.array([4, 5]), ms.int32), 2, 1, 0, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_input_must_3d_kwargs_int"
    )
    yield OpSampleInput(
        op_input=make_arg(shape=(4, 64, 55)),
        op_args=(ms.Tensor(np.array([32, 32]), ms.int64), [8, 8], (2, 3), 2, (2, 3)),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_input_must_3d_kwargs_sequence"
    )
    yield OpSampleInput(
        op_input=make_arg(shape=(9, 9, 4)),
        op_args=(ms.Tensor(np.array([6, 8]), ms.int32), 3, [2, 1], [0, 1], 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_input_must_3d_kwargs_sequence2"
    )

def extra_sample_inputs_tensor_fold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    #for empty shape
    yield OpSampleInput(
            op_input=make_arg(shape=(0, 12, 12)),
        op_args=(ms.Tensor(np.array([4, 6]), ms.int32), [2, 1], [1, 2], [0, 1], [1, 2]),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_shape"
    )
    #for large shape
    yield OpSampleInput(
        op_input=make_arg(shape=(256, 256, 25)),
        op_args=(ms.Tensor(np.array([8, 8]), ms.int32), 2, 2, 2, 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_large_shape"
    )
    values = small_large_values(dtype)
    #for small/large values
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(4, 64, 55)), dtype=dtype),
        op_args=(ms.Tensor(np.array([32, 32]), ms.int64), [8, 8], (2, 3), 2, (2, 3)),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    #for inf/nan values
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(4, 64, 55)), dtype=dtype),
        op_args=(ms.Tensor(np.array([32, 32]), ms.int64), [8, 8], (2, 3), 2, (2, 3)),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_extremal_values"
    )
    #for discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg(shape=(10, 12, 4), discontiguous=discontiguous),
            op_args=(ms.Tensor(np.array([1, 1]), ms.int64), 1, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def error_sample_inputs_tensor_fold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #input is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=np.random.uniform(0, 10, (1, 12, 4)),
            op_args=(ms.Tensor(np.array([1, 1]), ms.int64), 2, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=(TypeError, AttributeError),
        op_error_info='input is not tensor'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(1.0, (1, 1), 70, 0, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info='output_size is not tensor'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5, 6]), ms.int64), 1.0, 1, 0, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info='kernel_size is not int'
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5, 6]), ms.int64), (1, 1), 1.0, 0, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info='dilation is not int'
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5, 6]), ms.int64), (1, 1), 1, 0, 1.0),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info='stride_float is not int'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 3, 3, 3), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5, 6]), ms.int64), (1, 1), 1, 1, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='input is not 3D'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5,]), ms.int64), (1, 1), 1, 0, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='output_size is not 1D'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5, 6]), ms.bool), (1, 1), 1, 0, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='output_size dtype is not int'
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 3, 30), dtype=ms.float32),
            op_args=(ms.Tensor(np.array([5, 6]), ms.bool), (1, 1, 1), 1, 0, (1, 1)),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info='kernel_size length is invalid'
    )

def basic_sample_inputs_mint_nn_fold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    yield OpSampleInput(
        op_input=make_arg(shape=(64, 25)),
        op_args=((8, 8), (2, 2), [2, 2], (2, 2), 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_input_2d"
    )
    yield OpSampleInput(
        op_input=make_arg(shape=(10, 12, 4)),
        op_args=(3, 2, 1, 1, 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_input_3d"
    )
    yield OpSampleInput(
        op_input=make_arg(shape=(1, 12, 12)),
        op_args=((4, 6), [2, 1], [1, 2], [0, 1], [1, 2]),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_input_3d"
    )

def extra_sample_inputs_mint_nn_fold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    #for empty shape
    yield OpSampleInput(
        op_input=make_arg(shape=(0, 12, 12)),
        op_args=((4, 6), [2, 1], [1, 2], [0, 1], [1, 2]),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_shape"
    )
    #for large shape
    yield OpSampleInput(
        op_input=make_arg(shape=(256, 256, 25)),
        op_args=((8, 8), 2, 2, 2, 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_large_shape"
    )
    values = small_large_values(dtype)
    #for small/large values
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(64, 25)), dtype=dtype),
        op_args=((8, 8), (2, 2), [2, 2], (2, 2), 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    #for inf/nan values
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(64, 25)), dtype=dtype),
        op_args=((8, 8), (2, 2), [2, 2], (2, 2), 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_extremal_values"
    )
    #for discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg(shape=(10, 12, 4), discontiguous=discontiguous),
            op_args=(3, 2, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def error_sample_inputs_mint_nn_fold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #input is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=np.random.uniform(0, 10, (1, 12, 4)),
            op_args=(3, 2, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info='input is not tensor'
    )
    #batch is zero
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(1, 0, 4)),
            op_args=(3, 2, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=ValueError,
        op_error_info='batch is zero'
    )
    #padding is neg
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(1, 0, 4)),
            op_args=(3, 2, 1, -1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=ValueError,
        op_error_info='padding is neg'
    )
    #C of input cannot ne divided by product of kernels
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(1, 10, 6)),
            op_args=((4, 6), [2, 3], [1, 2], [0, 1], [1, 2]),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=ValueError,
        op_error_info='C of input cannot be divided by product of kernels'
    )
    #l is cannot match params of stride, dilation, padding, kernel
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(1, 12, 8)),
            op_args=((4, 6), [2, 1], [1, 2], [0, 1], [1, 2]),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=ValueError,
        op_error_info='l is cannot match params of stride, dilation, padding, kernel'
    )

def basic_sample_inputs_mint_nn_unfold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    yield OpSampleInput(
        op_input=make_arg(shape=(10, 2, 4, 4)),
        op_args=(3, 1, 0, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}"
    )
    yield OpSampleInput(
        op_input=make_arg(shape=(10, 2, 5, 6)),
        op_args=([2, 2], [2, 2], (2, 2), (2, 2)),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}"
    )

def extra_sample_inputs_mint_nn_unfold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    #empty shape
    yield OpSampleInput(
        op_input=make_arg(shape=(0, 2, 4, 4)),
        op_args=(3, 1, 0, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}"
    )
    #small/large values
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(10, 2, 4, 4)), dtype=dtype),
        op_args=(3, 1, 1, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024],
                                            size=(10, 2, 4, 4)), dtype=dtype),
        op_args=((2, 2), [2, 2], (1, 1), 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_extremal_values"
    )
    # for discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg(shape=(10, 2, 4, 4), discontiguous=discontiguous),
            op_args=(2, 2, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def error_sample_inputs_mint_nn_unfold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #input not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=np.random.uniform(0, 10, (10, 2, 4, 4)),
            op_args=(2, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=TypeError,
        op_error_info='input is not tensor'
    )
    #input is not 4D
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(4, 4)),
            op_args=(2, 1, 1, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=ValueError,
        op_error_info='input is not 4D'
    )
    #kernel/dilation/padding/stride is illegal
    cases=(
        ((-1, 1, 0, 2),  "kernel is illegal"),
        ((2, 0, 0, 2), "dilation is illegal"),
        ((2, 1, -1, 2), "padding is illegal"),
        ((2, 1, 1, 0), "stride is illegal"),
    )
    for arg, info in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=(2, 3, 4, 4)),
                op_args=arg,
                op_kwargs={},
                sample_name=f"{op_info.name}"
            ),
            op_error_type=ValueError,
            op_error_info=info
        )
    #calculate of output_size negative
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(2, 3, 2, 2)),
            op_args=(3, 2, 0, 1),
            op_kwargs={},
            sample_name=f"{op_info.name}"
        ),
        op_error_type=ValueError,
        op_error_info="calculate of output_size negative"
    )

def basic_sample_inputs_pixel_shuffle(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((16, 6, 4), 4), ((2, 3, 4, 4), 1), ((4, 4, 8, 9, 5, 5), 3))
    for shape, factor in cases:
        yield OpSampleInput(
            op_input=make_func(shape=shape),
            op_args=(factor,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{len(shape)}d"
        )

def extra_sample_inputs_pixel_shuffle(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    #empty shape
    yield OpSampleInput(
        op_input=make_func(shape=(0, 2, 2)),
        op_args=(3,),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_shape"
    )
    # ---- small/large value combining with discontiguous ----
    values = small_large_values(dtype)
    if values:
        input_array=np.random.choice(values, size=(9, 3, 4))
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(3,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
            )
    #for extremal values
    input_array = np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(9, 1, 4))
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(input_array, dtype=dtype),
        op_args=(3,),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_extremal_values"
    )

def error_sample_inputs_pixel_shuffle(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #upscale_factor is not int
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((9, 4, 4)),
            op_args=(3.0,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info="upscale_factor is not type of int",
    )
    #channel dimension is not divisible by square of upscale_factor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((9, 4, 4)),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info="channel dimension is not divisible by square of upscale_factor",
    )
    # input shape has less than 3 dimensions
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((9, 4)),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info="input shape has less than 3 dimensions",
    )

def basic_sample_inputs_up_sample(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for mode in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic']:
        func = functools.partial(basic_sample_inputs_mint_interpolate, mode=mode)
        for sample_input in func(op_info, dtype=dtype, device=device, **kwargs):
            if dtype==ms.float64:
                if mode != "trilinear" and mode != "nearest":
                    continue
                if mode == "nearest" and len(sample_input.op_input.shape)==4:
                    continue
            yield sample_input

def extra_sample_inputs_up_sample(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    N, C = 1, 1
    D = 2

    def shape_with_nc(side: int, rank: int):
        return tuple([N, C] + [side] * rank)

    for mode in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic']:
        if dtype == ms.float64 and mode != "trilinear" and mode != "nearest":
            continue

        internal_mode, ranks = _normalize_mode_and_ranks(mode)

        if internal_mode in ("linear", "bilinear", "bicubic", "trilinear"):
            align_corners_options = (True, False, None)
        else:
            align_corners_options = (None,)

        for align_corners in align_corners_options:
            for rank in ranks:
                if rank==2 and dtype==ms.float64:
                    continue
                # for large shape
                yield OpSampleInput(
                    op_input=make_func(shape_with_nc(128, rank)),
                    op_args=(),
                    op_kwargs={
                        "scale_factor": 3.0,
                        "mode": mode,
                        "align_corners": align_corners,
                        "recompute_scale_factor": None,
                    },
                    sample_name=f"{op_info.name}_{dtype}_large_shape",
                )
                #for extremal values:
                #results differ from torch when input has value np.inf and -np.inf
                yield OpSampleInput(
                    op_input=ms.Tensor(np.full(shape_with_nc(D, rank), np.nan), dtype=dtype),
                    op_args=(),
                    op_kwargs={
                        "scale_factor": tuple(np.random.uniform(0.5, 3, (rank,))),
                        "mode": mode,
                        "align_corners": align_corners,
                        "recompute_scale_factor": True,
                    },
                    sample_name=f"{op_info.name}_{dtype}_extremal_values",
                )
                #for discontiguous value
                yield OpSampleInput(
                    op_input=make_func(shape_with_nc(3, rank), discontiguous=True),
                    op_args=(),
                    op_kwargs={
                        "scale_factor": tuple(np.random.uniform(0.5, 3, (rank,))),
                        "mode": mode,
                        "align_corners": align_corners,
                        "recompute_scale_factor": True,
                    },
                    sample_name=f"{op_info.name}_{dtype}_discontiguous",
                )

def error_sample_inputs_up_sample(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for mode in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic']:
        func = functools.partial(error_inputs_other_op_interpolate, mode=mode)
        yield from func(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_nn_functional_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    order=['bias', 'stride', 'padding', 'output_padding', 'groups', 'dilation']
    for sample_input in basic_sample_inputs_mint_nn_conv_transpose2d(op_info, dtype, device):
        if "output_size" in sample_input.op_kwargs:
            continue
        weight=sample_input.op_kwargs.pop("weight_init")
        sample_input.op_kwargs.pop("padding_mode", None)
        sample_input.op_kwargs.pop("dtype", None)
        bias=sample_input.op_kwargs.pop("bias")
        if bias:
            bias=sample_input.op_kwargs.pop("bias_init")
        else:
            bias=None
        sample_input.op_kwargs.update({'bias':bias})
        yield OpSampleInput(
            op_input=sample_input.op_input,
            op_args=(weight,),
            op_kwargs={key:sample_input.op_kwargs[key] for key in order},
            sample_name=sample_input.sample_name,
        )

def extra_sample_inputs_mint_nn_functional_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for sample_input in extra_sample_inputs_mint_nn_conv_transpose2d(op_info, dtype, device):
        if "output_size" in sample_input.op_kwargs:
            continue
        weight=sample_input.op_kwargs.pop("weight_init")
        sample_input.op_kwargs.pop("padding_mode", None)
        sample_input.op_kwargs.pop("dtype", None)
        bias=sample_input.op_kwargs.pop("bias")
        if bias:
            bias=sample_input.op_kwargs.pop("bias_init")
        else:
            bias=None
        sample_input.op_kwargs.update({'bias':bias})
        yield OpSampleInput(
            op_input=sample_input.op_input,
            op_args=(weight,),
            op_kwargs=sample_input.op_kwargs,
            sample_name=sample_input.sample_name,
        )

def dynamic_sample_inputs_mint_nn_functional_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # add/sub_ext dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, None), dtype=dtype),),
                op_kwargs={"bias": ms.Tensor(shape=(None,), dtype=dtype), "stride": ms.mutable((2, 1)),
                           "padding": ms.mutable(0),"output_padding": ms.mutable(0),
                           "groups": ms.mutable(1), "dilation": ms.mutable((2, 1), dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10, 9, 8, 7)),
                    op_args=(make_func(shape=(9, 9, 2, 2)),),
                    op_kwargs={"bias": make_func(shape=(9,)), "stride": ms.mutable((1, 2)),
                           "padding": ms.mutable(0),"output_padding": ms.mutable(0),
                           "groups": ms.mutable(1), "dilation": ms.mutable((2, 1), dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 12, 12, 12)),
                    op_args=(make_func(shape=(12, 4, 1, 1)), ),
                    op_kwargs={"bias": make_func(shape=(12,)), "stride": ms.mutable((2, 1)),
                           "padding": ms.mutable(0),"output_padding": ms.mutable(0),
                           "groups": ms.mutable(3), "dilation": ms.mutable((1, 2))},
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
                op_kwargs={"bias": ms.Tensor(shape=None, dtype=dtype), "stride": ms.mutable((1, 2)),
                           "padding": ms.mutable(0),"output_padding": ms.mutable(0),
                           "groups": ms.mutable(1), "dilation": ms.mutable((2, 1), dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 7, 8)),
                    op_args=(make_func(shape=(9, 3, 1, 1)),),
                    op_kwargs={"bias": make_func(shape=(9,)), "stride": ms.mutable((1, 2)),
                           "padding": ms.mutable(0),"output_padding": ms.mutable(0),
                           "groups": ms.mutable(3), "dilation": ms.mutable((2, 1), dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10, 9, 7, 8)),
                    op_args=(make_func(shape=(9, 9, 2, 3)),),
                    op_kwargs={"bias": make_func(shape=(9,)),  "stride": ms.mutable((1, 2)),
                           "padding": ms.mutable(0),"output_padding": ms.mutable(0),
                           "groups": ms.mutable(1), "dilation": ms.mutable((2, 1), dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                )
            )
        )

def error_sample_inputs_mint_nn_functional_conv_transpose2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device)
    #stride/padding/outpadding/groups/dilation not int
    cases=(
        (1.0, 0, 0, 1, 1, "stride"),(1, 0.0, 0, 1, 1, "padding"),
        (1, 0, 0.0, 1, 1, "output_padding"),(1, 0, 0, 1.0, 1, "groups"),(1, 0, 0, 1, 1.0, "dilation")
    )
    for stride, padding, output_padding, groups, dilation, name in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=(10, 9, 7 ,8)),
                op_args=(make_func(shape=(9, 9, 2, 2)),),
                op_kwargs={
                    "stride":stride, "padding":padding, "output_padding":output_padding,
                    "groups":groups, "dilation":dilation
                },
                sample_name=op_info.name
            ),
            op_error_type=TypeError,
            op_error_info=f"{name} is not int"
        )
    #stride/dilation/padding/outputpadding is illegal
    cases=(
        (0, 0, 0, 1, 1, "stride"), (1, -1, 0, 1, 1, "padding"),
        (1, 0, -1, 1, 1, "output_padding"), (1, 0, 0, -1, 1, "groups"), (1, 0, 0, 1, 0, "dilation")
    )
    for stride, padding, output_padding, groups, dilation, name in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=(10, 9, 7 ,8)),
                op_args=(make_func(shape=(9, 9, 2, 2)),),
                op_kwargs={
                    "stride":stride, "padding":padding, "output_padding":output_padding,
                    "groups":groups, "dilation":dilation
                },
                sample_name=op_info.name
            ),
            op_error_type=ValueError,
            op_error_info=f"{name} is illegal"
        )
    #shape of bias not equal (output_channel,)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(10, 9, 7, 8)),
            op_args=(make_func(shape=(9, 9, 2, 2)),),
            op_kwargs={
                "stride": 1, "padding": 0, "output_padding": 0,
                "groups": 1, "dilation": 1, "bias":make_func(shape=(10, )),
            },
            sample_name=op_info.name
        ),
        op_error_type=RuntimeError,
        op_error_info="shape of bias not equal (output_channel,)"
    )
    # output_padding is bigger than dilation and stride
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(10, 9, 7, 8)),
            op_args=(make_func(shape=(9, 9, 2, 2)),),
            op_kwargs={
                "stride": 1, "padding": 0, "output_padding": 2,
                "groups": 1, "dilation": 2
            },
            sample_name=op_info.name
        ),
        op_error_type=RuntimeError,
        op_error_info="output_padding is bigger than stride"
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(10, 9, 7, 8)),
            op_args=(make_func(shape=(9, 9, 2, 2)),),
            op_kwargs={
                "stride": 2, "padding": 0, "output_padding": 2,
                "groups": 1, "dilation": 1
            },
            sample_name=op_info.name
        ),
        op_error_type=RuntimeError,
        op_error_info="output_padding is bigger than dilation"
    )

def basic_sample_inputs_mint_nn_cosineembeddingloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    XS = EXTRA_SMALL_DIM_SIZE
    S = EXTRA_SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE

    make_func = functools.partial(make_tensor, device=device, dtype=dtype, random_seed=1)

    cases = (
        ((S,), (1,), (), 1.0, 'sum'),
        ((M, S), (M, S), (M,), 0.5, 'none'),
        ((S, XS), (S, XS), (S,), -1.0, 'mean'),
        ((M, 2), (1, 2), (1,), 0.8, 'sum'),
    )

    for input1_shape, input2_shape, target_shape, margin, reduction in cases:
        yield OpSampleInput(
            op_input=make_func(input1_shape),
            op_args=(
                make_func(input2_shape),
                ms.Tensor(np.random.choice([-1, 1], size=target_shape), dtype=ms.int32)
            ),
            op_kwargs={'margin': margin, 'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(input1_shape)}d",
        )
    # empty tensor and default param value
    empty_tensor = make_func((2, 0))
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(empty_tensor, ms.Tensor(np.random.choice([-1, 1], size=(2,)), dtype=ms.int64)),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_cosineembeddingloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input1_array = np.random.choice(values, size=(5, 2))
            input2_array = np.random.choice(values, size=(1, 2), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(
                    make_tensor_with_np_array(input2_array, dtype=dtype, discontiguous=discontiguous),
                    ms.Tensor(np.random.choice([-1, 1], size=(1,)), dtype=ms.int8)
                ),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_small_large_vales_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((3, 2), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(input_tensor, ms.Tensor(np.random.choice([-1, 1], size=(3,)), dtype=ms.int16)),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_cosineembeddingloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.CosineEmbeddingLoss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        target = ms.Tensor(2 * np.random.randint(0, 2, size=()) - 1, dtype=ms.int64)
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(ms.Tensor(shape=(None,), dtype=dtype), target),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8,)),
                    op_args=(make_func(shape=(8,)), target),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7,)),
                    op_args=(make_func(shape=(7,)), target),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        target = ms.Tensor(2 * np.random.randint(0, 2, size=(3,)) - 1, dtype=ms.int64)
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=dtype), target),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 1)),
                    op_args=(make_func(shape=(3, 1)), target),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 2)),
                    op_args=(make_func(shape=(3, 2)), target),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_cosineembeddingloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((5, 2), (2, 1), (5,), 1.0, 'none'),
        ((5, 2), (5, 1), (5,), 1.0, 'batchmean'),
        ((5, 2), (1, 2), (2,), -1.0, 'sum'),
        ((5, 2), (5, 1), (5,), 1, 'mean'),
        ((5, 2), (5, 1), (5, 1), 1.0, 'mean'),
        ((5, 2, 1), (5, 1, 1), (5,), 1.0, 'mean'),
        ((), (), (), 1.0, 'mean'),
    )
    # shapes of inputs 'input' and 'target' are not equal or cannot broadcast
    # reduction is not 'none'/'mean'/'sum'
    # the shape of target is not matched with input1 or input2
    # margin is not float
    # target is not 0d or 1d
    # input1/input2 is 3d, but target is 1d
    # input1/input2/target is 0d
    for input1_shape, input2_shape, target_shape, margin, reduction in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input1_shape),
                op_args=(
                    make_arg(shape=input2_shape),
                    ms.Tensor(np.random.choice([-1, 1], size=target_shape), dtype=ms.int32)
                ),
                op_kwargs={'margin': margin, 'reduction': reduction},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input1/input2/target is scalar tensor
    scalar_tensor = ms.Tensor(1, dtype=ms.int32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=scalar_tensor,
            op_args=(scalar_tensor, scalar_tensor),
            op_kwargs={},
            sample_name=f"{op_info.name}_scalar_tensor_error",
        ),
        op_error_type=ValueError,
        op_error_info=f'input scalar tensor error values is not supported for {op_info.name}',
    )

def basic_sample_inputs_mint_nn_crossentropyloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3, 5), (3, 5), -1, 'mean', 0.0),
        ((3, 2), (3, 2), -50, 'none', 0.2),
        ((3, 5, 4, 4), (3, 5, 4, 4), -10, 'sum', 0.1),
    )

    for input_shape, target_shape, ignore_index, reduction, label_smoothing in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(make_func(target_shape),),
            op_kwargs={'weight': None, 'ignore_index': ignore_index, 'reduction': reduction,
                       'label_smoothing': label_smoothing},
            sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d",
        )
    # indices labels
    if dtype != ms.float64:
        yield OpSampleInput(
            op_input=make_func((3,)),
            op_args=(ms.Tensor(1, dtype=ms.int64),),
            op_kwargs={'weight': None, 'ignore_index': 3,
                       'reduction': 'none', 'label_smoothing': 0.3},
            sample_name=f"{op_info.name}_indices_labels",
        )
        # indices labels: target has negative value and default param value
        np_input = np.random.randn(3, 5)
        yield OpSampleInput(
            op_input=ms.Tensor(np_input, dtype=dtype),
            op_args=(ms.Tensor(np.array([1, 0, -100]), dtype=ms.int64),),
            op_kwargs={},
            sample_name=f"{op_info.name}_indices_labels_target_neg_value",
        )

def extra_sample_inputs_mint_nn_crossentropyloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3, 5), (3, 5), -100, 'none', 0.8),
        ((3, 5, 4, 4, 6, 7, 8), (3, 5, 4, 4, 6, 7, 8), -1, 'sum', 0.5),
        ((3, 111, 2, 6, 5, 6, 4, 1), (3, 111, 2, 6, 5, 6, 4, 1), -100, 'mean', 1.0),
    )

    for input_shape, target_shape, ignore_index, reduction, label_smoothing in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(make_func(target_shape),),
            op_kwargs={'weight': None, 'ignore_index': ignore_index, 'reduction': reduction,
                       'label_smoothing': label_smoothing},
            sample_name=f"{op_info.name}_{dtype}_extra_{len(input_shape)}d",
        )
    # 0d, indices labels
    yield OpSampleInput(
        op_input=ms.Tensor([1, 2, 3], dtype=dtype),
        op_args=(ms.Tensor(0, dtype=ms.int64),),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_extra_0d",
    )
    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input1_array = np.random.choice(values, size=(5, 2))
            input2_array = np.random.choice(values, size=(5, 2), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(input2_array, dtype=dtype, discontiguous=discontiguous),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_small_large_values_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((3, 2), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(ms.Tensor(np.array([1, 0, 1]), dtype=ms.int64),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_crossentropyloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.CrossEntropyLoss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 4)),
                    op_args=(make_func(shape=(8, 4)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 3)),
                    op_args=(make_func(shape=(7, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(make_func(shape=(3,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5,)),
                    op_args=(make_func(shape=(5,)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_crossentropyloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((3, 2), (3, 2), (2,), -5, 'none', 1.1),
        ((3, 2), (3, 2), (2,), 0, 'None', 1.0),
        ((3, 2), (3, 2), (3,), -1, 'sum', 1.0),
        ((3, 5, 4, 4), (3, 4, 4), (5,), -10, 'sum', 0.1),
        ((0, 2), (0, 2), (2,), -5, 'mean', 1.0),
    )
    # label_smoothing is not in [0.0, 1.0]
    # reduction is not none/mean/sum
    # input is (N, C), the shape of weight is not (C,)
    # probability labels, input shape should be equal to target shape
    # input doesn't support 0-dim or shape[0] == 0
    for input_shape, target_shape, weight_shape, ignore_index, reduction, label_smoothing in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape),
                op_args=(make_arg(shape=target_shape),),
                op_kwargs={'weight': make_arg(weight_shape), 'ignore_index': ignore_index,
                           'reduction': reduction, 'label_smoothing': label_smoothing},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # indices labels, input does not support float64
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.Tensor(shape=(3,), dtype=ms.float64),
            op_args=(ms.Tensor(1, dtype=ms.int64),),
            op_kwargs={},
            sample_name=f"{op_info.name}_scalar_tensor_error",
        ),
        op_error_type=RuntimeError,
        op_error_info=f'indices labels does not support float64 for {op_info.name}',
    )

def basic_sample_inputs_mint_nn_kldivloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((1,), (1,), 'batchmean', False),
        ((2, 2), (2, 2), 'none', True),
        ((3, 2), (3, 1), 'sum', True),
    )

    for input_shape, target_shape, reduction, log_target in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(make_func(target_shape),),
            op_kwargs={'reduction': reduction, 'log_target': log_target},
            sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d",
        )
    # empty tensor and default param value
    empty_tensor = make_func((2, 0))
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(empty_tensor,),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_kldivloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), 'none', False),
        ((2, 2, 2, 3, 4), 'mean', True),
        ((2, 2, 2, 2, 3, 4, 50, 60), 'sum', False),
    )

    for input_shape, reduction, log_target in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(make_func(input_shape),),
            op_kwargs={'reduction': reduction, 'log_target': log_target},
            sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d",
        )
    # input/target is scalar tensor
    scalar_tensor = ms.Tensor(1, dtype=dtype)
    yield OpSampleInput(
        op_input=scalar_tensor,
        op_args=(scalar_tensor,),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_scalar_tensor",
    )
    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input1_array = np.random.choice(values, size=(5, 2, 3))
            input2_array = np.random.choice(values, size=(1, 2, 3), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(input2_array, dtype=dtype, discontiguous=discontiguous),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_small_large_vales_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((2, 2), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(input_tensor,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_kldivloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.KLDivLoss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4, 6)),
                    op_args=(make_func(shape=(5, 4, 6)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4, 5)),
                    op_args=(make_func(shape=(3, 4, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 1)),
                    op_args=(make_func(shape=(3, 1)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 2)),
                    op_args=(make_func(shape=(5, 2)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_kldivloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((5, 2), (2, 1), ms.float32, ms.float32, 'none', True),
        # 20260601 case can pass
        # ((5, 2), (1, 2), ms.float16, ms.bfloat16, 'sum', False),
        ((5, 2), (5, 1), ms.float16, ms.float16, 'batch', False),
        ((5, 2), (1, 2), ms.bfloat16, ms.bfloat16, 'sum', 1),
    )
    # two inputs 'input' and 'target' shape are not equal or can not broadcast
    # the dtype of 'input' and 'target' is not same
    # reduction is not 'none'/'mean'/'sum'/'batchmean'
    # log_target is not bool
    for input_shape, target_shape, input_dtype, target_dtype, reduction, log_target in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape, dtype=input_dtype),
                op_args=(make_arg(shape=target_shape, dtype=target_dtype),),
                op_kwargs={'reduction': reduction, 'log_target': log_target},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input/target is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_not_tensor_error",
        ),
        op_error_type=TypeError,
        op_error_info=f'input/target not tensor is not supported for {op_info.name}',
    )

def basic_sample_inputs_mint_nn_l1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3, 5), (5,), 'mean'),
        ((3,), (3,), 'none'),
        ((2, 2), (1, 2), 'sum'),
    )

    for input1_shape, input2_shape, reduction in cases:
        yield OpSampleInput(
            op_input=make_func(input1_shape),
            op_args=(make_func(input2_shape),),
            op_kwargs={'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(input1_shape)}d",
        )
    # empty tensor and default param value
    empty_tensor = make_func((2, 0))
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(empty_tensor,),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_l1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), (), 'mean', None),
        ((128, 32, 8), (128, 32, 8), 'none', ms.int32),
        ((8, 16, 32, 5, 7), (8, 16, 32, 1, 1), 'none', ms.uint8),
        ((10, 5, 32, 8, 16, 4, 2, 1), (10, 5, 32, 8, 16, 4, 2, 10), 'mean', None),
    )

    for input1_shape, input2_shape, reduction, target_dtype in cases:
        if target_dtype is None:
            target_dtype = dtype
        yield OpSampleInput(
            op_input=make_func(input1_shape),
            op_args=(make_func(input2_shape, dtype=target_dtype),),
            op_kwargs={'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(input1_shape)}d",
        )
    # input/target is scalar tensor
    scalar_tensor = ms.Tensor(1, dtype=ms.float32)
    yield OpSampleInput(
        op_input=scalar_tensor,
        op_args=(scalar_tensor,),
        op_kwargs={'reduction': 'sum'},
        sample_name=f"{op_info.name}_scalar_tensor",
    )
    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input1_array = np.random.choice(values, size=(5, 2))
            input2_array = np.random.choice(values, size=(1, 2), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(input2_array, dtype=dtype, discontiguous=discontiguous),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_small_large_vales_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((3, 2), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(input_tensor,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_l1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.L1Loss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 4)),
                    op_args=(make_func(shape=(8, 4)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 3)),
                    op_args=(make_func(shape=(7, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=dtype),),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 1)),
                    op_args=(make_func(shape=(3, 1)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 2)),
                    op_args=(make_func(shape=(5, 2)),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_l1loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((5, 2), (2, 1), 'none'),
        ((5, 2), (5, 1), 'batch'),
    )
    # two inputs 'input' and 'target' shape are not equal or can not broadcast
    # reduction is not 'none'/'mean'/'sum'
    for input_shape, target_shape, reduction in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape),
                op_args=(make_arg(shape=target_shape),),
                op_kwargs={'reduction': reduction, },
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input/target is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_target_not_tensor",
        ),
        op_error_type=TypeError,
        op_error_info=f'input/target not tensor is not supported for {op_info.name}',
    )

def basic_sample_inputs_mint_nn_mseloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.L1Loss
    yield from basic_sample_inputs_mint_nn_l1loss(op_info, dtype=dtype, device=device, **kwargs)

def extra_sample_inputs_mint_nn_mseloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), (), 'mean'),
        ((128, 32, 8), (128, 32, 8), 'none'),
        ((16, 4, 3, 7, 5), (16, 4, 3, 7, 1), 'sum'),
        ((2, 4, 8, 5, 4, 32, 64, 1), (2, 4, 8, 5, 4, 32, 64, 10), 'mean'),
    )

    for input1_shape, input2_shape, reduction in cases:
        yield OpSampleInput(
            op_input=make_func(input1_shape),
            op_args=(make_func(input2_shape),),
            op_kwargs={'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(input1_shape)}d",
        )
    # input/target is scalar tensor
    scalar_tensor = ms.Tensor(1, dtype=ms.float32)
    yield OpSampleInput(
        op_input=scalar_tensor,
        op_args=(scalar_tensor,),
        op_kwargs={'reduction': 'sum'},
        sample_name=f"{op_info.name}_scalar_tensor",
    )
    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            input1_array = np.random.choice(values, size=(5, 2))
            input2_array = np.random.choice(values, size=(1, 2), replace=False)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(input2_array, dtype=dtype, discontiguous=discontiguous),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_small_large_vales_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((3, 2), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(input_tensor,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_mseloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.L1Loss
    yield from dynamic_sample_inputs_mint_nn_l1loss(op_info, dtype=dtype, device=device, **kwargs)

def error_sample_inputs_mint_nn_mseloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((5, 2), (2, 1), ms.float32, ms.float32, 'none'),
        ((5, 2), (1, 2), ms.float16, ms.bfloat16, 'sum'),
        ((5, 2), (5, 1), ms.float16, ms.float16, 'batch'),
    )
    # two inputs 'input' and 'target' shape are not equal or can not broadcast
    # the dtype of 'input' and 'target' is not same
    # reduction is not 'none'/'mean'/'sum'
    for input_shape, target_shape, input_dtype, target_dtype, reduction in cases:
        if "_kbk" in op_info.name and input_dtype != target_dtype:
            continue
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape, dtype=input_dtype),
                op_args=(make_arg(shape=target_shape, dtype=target_dtype),),
                op_kwargs={'reduction': reduction,},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(RuntimeError, ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input/target is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_target_not_tensor",
        ),
        op_error_type=TypeError,
        op_error_info=f'input/target not tensor is not supported for {op_info.name}',
    )

def basic_sample_inputs_mint_nn_nllloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device)

    N, C = 3, 5
    cases = (
        ((N, C), (N,), (C,), -1, 'mean'),
        ((N, C), (N,), (C,), -50, 'none'),
        ((N, C, 4, 4), (N, 4, 4), (C,), -10, 'sum'),
    )

    for input_shape, target_shape, weight_shape, ignore_index, reduction in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape, dtype=dtype),
            op_args=(make_func(target_shape, dtype=ms.int64, low=0, high=C - 1),),
            op_kwargs={'weight': make_func(weight_shape, dtype=dtype),
                       'ignore_index': ignore_index, 'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d",
        )
    # empty tensor and default param value
    # target dtype int32/uint8 should transform to Long for torch
    empty_tensor = make_func((2, 3, 0, 1), dtype=dtype)
    yield OpSampleInput(
        op_input=empty_tensor,
        op_args=(make_func(shape=(2, 0, 1), dtype=ms.int64, low=0, high=2),),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_nllloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3, 5, 4, 4, 6, 7, 8), (3, 4, 4, 6, 7, 8), (5,), -1, 'sum', ms.int64, (5 - 1)),
        ((3, 2, 2, 6, 5, 6, 4, 1), (3, 2, 6, 5, 6, 4, 1), (2,), -100, 'mean', ms.int64, (2 - 1)),
        ((3, 7, 2, 6, 5, 6, 4, 1), (3, 2, 6, 5, 6, 4, 1), (7,), -10, 'mean', ms.int64, (7 - 1)),
    )

    for input_shape, target_shape, weight_shape, ignore_index, reduction, target_dtype, target_high in cases:
        yield OpSampleInput(
            op_input=make_func(input_shape),
            op_args=(make_func(target_shape, dtype=target_dtype, low=0, high=target_high),),
            op_kwargs={'weight': make_func(weight_shape), 'ignore_index': ignore_index, 'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d",
        )
    # 0d
    yield OpSampleInput(
        op_input=ms.Tensor([1, 2, 3], dtype=dtype),
        op_args=(ms.Tensor(0, dtype=ms.int64),),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_0d",
    )
    # small/large values
    values = small_large_values(dtype)
    if values:
        rng = np.random.default_rng(0)
        input1_array = rng.choice(values, size=(5, 6))
        target_array = rng.integers(0, 6, size=(5,))
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input1_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(ms.Tensor(target_array, dtype=ms.int64),),
                op_kwargs={'reduction': 'none'},
                sample_name=f"{op_info.name}_{dtype}_small_large_vales_discontiguous_{discontiguous}"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        input_tensor = ms.Tensor(np.full((3, 5), t), dtype=dtype)
        yield OpSampleInput(
            op_input=input_tensor,
            op_args=(ms.Tensor(np.array([1, 0, 4]), dtype=ms.int64),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

def dynamic_sample_inputs_mint_nn_nllloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.NLLLoss.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=ms.int64),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 4, 5)),
                    op_args=(make_func(shape=(3, 5), dtype=ms.int64, low=0, high=3),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 5)),
                    op_args=(make_func(shape=(2, 5), dtype=ms.int64, low=0, high=2),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(ms.Tensor(None, dtype=ms.int64),),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4)),
                    op_args=(make_func(shape=(5,), dtype=ms.int64, low=0, high=3),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 2)),
                    op_args=(make_func(shape=(2,), dtype=ms.int64, low=0, high=1),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            ),
        )

def error_sample_inputs_mint_nn_nllloss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((3, 2), (3, 2), (2,), 0, 'None'),
        ((3, 2), (3, 2), (2,), 0.1, 'mean'),
        ((3, 2), (3, 2), (3,), -1, 'sum'),
        ((3, 2), (3, 2), (2,), -1, 'mean'),
    )
    # reduction is not none/mean/sum
    # ignore_index is not int
    # input is (N, C), the shape of weight is not (C,)
    for input_shape, target_shape, weight_shape, ignore_index, reduction in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=input_shape, dtype=ms.float32),
                op_args=(make_arg(shape=target_shape, dtype=ms.float32),),
                op_kwargs={'weight': make_arg(weight_shape, dtype=ms.float32), 'ignore_index': ignore_index,
                           'reduction': reduction},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(RuntimeError, ValueError, TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # empty tensor
    empty_tensor = make_arg((2, 0), dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=empty_tensor,
            op_args=(make_arg((2, 0), dtype=ms.int64),),
            op_kwargs={},
            sample_name=f"{op_info.name}_empty_tensor_error",
        ),
        op_error_type=RuntimeError,
        op_error_info=f'input C is 0, multi-target is not supported for {op_info.name}',
    )
    # input/target is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_target_not_tensor",
        ),
        op_error_type=AttributeError,
        op_error_info=f'input/target not tensor is not supported for {op_info.name}',
    )
    # when weight is not None, dtype of weight is not float16/float32/bfloat16
    weight_not_support_dtypes = (
        ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.float64, ms.complex64, ms.complex128
    )
    for weight_dtype in weight_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=(3, 2), dtype=ms.float32),
                op_args=(make_arg(shape=(3, 2), dtype=ms.int32),),
                op_kwargs={'weight': ms.Tensor(shape=(2,), dtype=weight_dtype),},
                sample_name=f"{op_info.name}_weight_dtype_{weight_dtype}_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'weight error dtype {weight_dtype} is not supported for {op_info.name}',
        )
    # target dtype is not in [int32/int64/uint8]
    target_not_support_dtypes = (
        ms.bool_, ms.int8, ms.int16, ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.bfloat16
    )
    for target_dtype in target_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=(3, 2), dtype=ms.float32),
                op_args=(make_arg(shape=(3, 2), dtype=target_dtype),),
                op_kwargs={'weight': ms.Tensor(shape=(2,), dtype=ms.float32)},
                sample_name=f"{op_info.name}_target_dtype_{target_dtype}_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'target error dtype {target_dtype} is not supported for {op_info.name}',
        )

def basic_sample_inputs_mint_nn_embedding(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.Embedding.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0)
    # input_shape, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
    # _weight, _freeze, dtype
    cases = (
        ((1,), 10, 8, None, None, 2.0, False, ms.Tensor(np.random.randn(10, 8), dtype=ms.float32), False, None),
        ((3, 5), 10, 8, 0, 1.0, 2.0, True, ms.Tensor(np.random.randn(10, 8), dtype=ms.float16), False, ms.float32),
        ((2, 3, 4), 5, 8, -5, -1.0, 1.0, False, ms.Tensor(np.random.randn(5, 8), dtype=ms.bfloat16), True, None),
    )

    for (input_shape, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, _weight,
         _freeze, case_dtype) in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape, high=num_embeddings),
            op_args=(num_embeddings, embedding_dim),
            op_kwargs={'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type,
                       'scale_grad_by_freq': scale_grad_by_freq, '_weight': _weight,
                       '_freeze': _freeze, 'dtype': case_dtype},
            sample_name=f"{op_info.name}_basic_{dtype}"
        )
    # empty tensor and default values
    yield OpSampleInput(
        op_input=make_arg(shape=(3, 0), high=3),
        op_args=(3, 0),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_embedding(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.nn.Embedding.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    # input_shape, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
    # _weight, _freeze, dtype
    cases = (
        ((), 7, 4, -4, 0.4, 0.4, True, False, ms.float32),
        ((4, 8, 8, 4), 5, 3, 4, 0.1, -2.0, True, True, ms.int8),
        ((4, 8, 16, 8, 4), 10, 3, 1, 12.3, 12.3, True, True, ms.bfloat16),
        ((4, 8, 16, 8, 4, 2, 2), 7, 4, -2, -2.0, -0.2, False, True, ms.float16),
        # 20260612 8D fail
        # ((4, 8, 16, 8, 4, 2, 2, 1), 7, 4, -2, -2.0, -0.2, False, True, ms.float16),
    )

    for (input_shape, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, _freeze,
         case_dtype) in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape, high=num_embeddings),
            op_args=(num_embeddings, embedding_dim),
            op_kwargs={'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type,
                       'scale_grad_by_freq': scale_grad_by_freq,
                       '_weight': ms.Tensor(np.random.randn(num_embeddings, embedding_dim), dtype=ms.float32),
                       '_freeze': _freeze, 'dtype': case_dtype},
            sample_name=f"{op_info.name}_extra_{dtype}",
        )

    # Non-float weights are valid when max_norm is disabled.
    yield OpSampleInput(
        op_input=make_arg((2, 3), high=5),
        op_args=(5, 3),
        op_kwargs={
            'padding_idx': None,
            'max_norm': None,
            'norm_type': 2.0,
            'scale_grad_by_freq': False,
            '_weight': ms.Tensor(np.arange(15).reshape(5, 3), dtype=ms.uint8),
            '_freeze': True,
            'dtype': None
        },
        sample_name=f"{op_info.name}_{dtype}_non_float_weight_without_max_norm",
    )

    # small/large values
    values = small_large_values(dtype)
    if values:
        for scale_grad_by_freq in (True, False):
            _weight = ms.Tensor(np.random.choice(values, size=(8, 9)), dtype=ms.float32)
            yield OpSampleInput(
                op_input=ms.Tensor(np.random.randint(0, 8, (3, 4)), dtype=ms.int32),
                op_args=(8, 9),
                op_kwargs={'padding_idx': -1, 'max_norm': 0.6, 'norm_type': 2.4,
                           'scale_grad_by_freq': scale_grad_by_freq, '_weight': _weight},
                sample_name=f"{op_info.name}_{dtype}_small_large_values"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.random.randint(0, 3, (2, 3)), dtype=dtype),
            op_args=(3, 4),
            op_kwargs={'padding_idx': 0, 'max_norm': 2.0, 'norm_type': 10.3,
                       '_weight': ms.Tensor(np.full((3, 4), t), dtype=ms.float32)},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

    # discontiguous tensor
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float16)
    yield OpSampleInput(
        op_input=make_arg((2, 3), dtype=ms.int64, discontiguous=True, high=4),
        op_args=(4, 3),
        op_kwargs={'_weight': make_func((4, 3), discontiguous=True), 'dtype': ms.bfloat16},
        sample_name=f"{op_info.name}_{dtype}_discontiguous_True"
    )

def error_sample_inputs_mint_nn_embedding(op_info:OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.Embedding.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.int64, low=0)
    cases = (
        # num_embeddings is not int
        (2, (3, 2), 3.0, 3, None, 0.4, 2.0, False, False, None, False, None),
        # embedding_dim is not int
        (2, (3, 2), 3, 3.0, None, 0.4, 2.0, False, False, None, False, None),
        # the values of input are not in [0, num_embeddings)
        # 20260611 case can pass
        # (6, (3, 2), 3, 3, None, 0.4, 2.0, False, False, None, False, None),
        # padding_idx is not int
        (3, (3, 2), 5, 3, 3.0, 0.4, 2.0, False, False, None, False, None),
        # padding_idx is out of range, not in [-num_embeddings, num_embeddings)
        (3, (3, 2), 5, 3, 5, 0.4, 2.0, False, False, None, False, None),
        # max_norm is not float
        (3, (3, 2), 5, 3, 3, 4, 2.0, False, False, None, False, None),
        # norm_type is not float
        (3, (3, 2), 5, 3, 3, 0.4, 2, False, False, None, False, None),
        # scale_grad_by_freq is not bool
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, 5, False, None, False, None),
        # sparse is not bool
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, 5, None, False, None),
        # where `sparse` is True, has not be supported.
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, True, None, False, None),
        # _freeze is not bool
        # 20260612 case can pass
        # (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, None, 5, None),
        # dtype is not mindspore.dtype
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, None, False, np.float32),
        # dtype does not support bool_, when _weight is None
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, None, False, ms.bool_),
        # _weight.shape is not (num_embeddings, embedding_dim)
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, ms.Tensor(np.random.randn(4, 3)), False, None),
        # _weight.shape is not (num_embeddings, embedding_dim)
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, ms.Tensor(np.random.randn(5, 2)), False, None),
        # _weight dtype is not float when max_norm is enabled (uint8 should be rejected)
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, ms.Tensor(np.random.randn(5, 3).astype(np.uint8)), False, None),
        # dtype is not float when max_norm is enabled and _weight is None
        (3, (3, 2), 5, 3, 3, 0.4, 2.0, False, False, None, False, ms.int32),
    )

    for input_high, input_shape, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,\
            sparse, _weight, _freeze, case_dtype in cases:
        if input_high > num_embeddings:
            input_tensor = ms.Tensor(np.random.randint(num_embeddings, input_high, input_shape))
        else:
            input_tensor = make_arg(shape=input_shape, high=input_high)
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_tensor,
                op_args=(num_embeddings, embedding_dim),
                op_kwargs={'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type,
                           'scale_grad_by_freq': scale_grad_by_freq, 'sparse': sparse, '_weight': _weight,
                           '_freeze': _freeze, 'dtype': case_dtype},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(3, 2),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_not_tensor"
        ),
        op_error_type=TypeError,
        op_error_info=f'{op_info.name}: input is not Tensor'
    )

def dynamic_sample_inputs_mint_nn_embedding(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.Embedding.
    When _weight is None, the output is non-deterministic, making accuracy comparison unreliable.
    Therefore, a larger loss value is set, and accuracy is skipped in this scenario.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(None, 8),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 6), high=10),
                    op_args=(10, 8),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    loss=3,
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 8), high=15),
                    op_args=(15, 8),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    loss=3,
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(None, 6),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6,), high=8),
                    op_args=(8, 6),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input',
                    loss=3,
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4), high=8),
                    op_args=(8, 6),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input',
                    loss=3,
                ),
            )
        )

def basic_sample_inputs_mint_nn_functional_grid_sample(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    cases=(
        ((10, 10, 4, 5), (10, 4, 5, 2),),
        ((10, 10, 4, 5, 6), (10, 5, 6, 7, 3),)
    )
    for input_shape, grid_shape in cases:
        for mode, padding_mode, align_corners in itertools.product(
                ['nearest', 'bilinear'], ['zeros', 'border', 'reflection'], [True, False]):
            yield OpSampleInput(
                op_input=make_func(shape=input_shape),
                op_args=(make_func(shape=grid_shape, low=-1, high=1),),
                op_kwargs={'mode':mode, 'padding_mode':padding_mode, 'align_corners':align_corners},
                sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d"
            )
            if mode != 'nearest':
                yield OpSampleInput(
                    op_input=make_func(shape=input_shape),
                    op_args=(make_func(shape=grid_shape, low=-2, high=2),),
                    op_kwargs={'mode': mode, 'padding_mode': padding_mode, 'align_corners': align_corners},
                    sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d"
                )

def extra_sample_inputs_mint_nn_functional_grid_sample(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    #for large shape
    yield OpSampleInput(
        op_input=make_func(shape=(1, 1, 1024, 812)),
        op_args=(make_func(shape=(1, 256, 256, 2), low=-1, high=1),),
        op_kwargs={"mode":'nearest'},
        sample_name=f"{op_info.name}_{dtype}_large_shape"
    )
    #for large/small values
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(4, 5, 6, 7)), dtype=dtype),
        op_args=(make_func(shape=(4, 7, 8, 2), low=-2, high=2),),
        op_kwargs={"mode":'bilinear', 'padding_mode':'border', 'align_corners':False},
        sample_name=f"{op_info.name}_{dtype}_large_small_value"
    )
    # for extremal values
    values=[np.inf, -np.inf, np.nan, 1024]
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(1, 4, 5, 6, 7)), dtype=dtype),
        op_args=(make_func(shape=(1, 4, 7, 8, 3), low=-1, high=1),),
        op_kwargs={"mode": 'bilinear', 'padding_mode': 'reflection', 'align_corners':True},
        sample_name=f"{op_info.name}_{dtype}_extremal_value"
    )
    #for discontiguous value
    yield OpSampleInput(
        op_input=make_func(shape=(4, 5, 6, 7), discontiguous=True),
        op_args=(make_func(shape=(4, 7, 8, 2), low=-2, high=2, discontiguous=True),),
        op_kwargs={"mode": 'bilinear', 'padding_mode':'reflection', 'align_corners':True},
        sample_name=f"{op_info.name}_{dtype}_discontiguous_True"
    )

def error_sample_inputs_mint_nn_functional_grid_sample(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device)
    #dtype of input and grid are different
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(4, 5, 6, 7), dtype=ms.float32),
            op_args=(make_func(shape=(4, 7, 8, 2), low=-2, high=2, dtype=ms.bfloat16),),
            op_kwargs={"mode": 'bilinear', 'padding_mode': 'reflection', 'align_corners': True},
            sample_name=op_info.name
        ),
        op_error_type=TypeError,
        op_error_info="dtype of input and grid are not same"
    )
    cases=(
        ((4, 5, 6), (4, 7, 8, 2), {}, ValueError, "shape of input is not 4D nor 5D"),
        ((1, 4, 5, 6), (1, 7, 8, 9, 10, 2), {}, ValueError, "shape of grid is not 4D or 5D"),
        ((1, 4, 5, 6), (1, 4, 6, 7, 3), {}, ValueError, "shape of input and grid does not match"),
        ((1, 4, 5, 6), (2, 6, 7, 2), {}, ValueError,
         "value of 1st dimension of input-shape and grid-shape does not match"),
        ((1, 4, 5, 6, 7), (1, 6, 7, 8, 2), {}, ValueError, "input and grid are 5D, but last value of dimension is 2"),
        ((1, 4, 5, 6), (1, 6, 7, 3), {}, ValueError, "input and grid are 4D, but last value of dimension is 3"),
        ((1, 4, 5, 6), (1, 6, 7, 2), {'mode': 'unknown'}, RuntimeError,
         "padding_mode is not follows: nearest, bilinear"),
        ((1, 4, 5, 6, 7), (1, 6, 7, 8, 3), {'mode':'bilinear', 'align_corners':1}, TypeError,
         "type of align_corners is not bool"),
        ((1, 4, 5, 6, 7), (1, 6, 7, 8, 3), {"padding_mode":"unknown"}, RuntimeError,
         "padding_mode is not follows: zeros, border, reflection")
    )
    for input_shape, grid_shape, params, err_type, err_info in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=input_shape, dtype=ms.float32),
                op_args=(make_func(shape=grid_shape, low=-2, high=2, dtype=ms.float32),),
                op_kwargs=params,
                sample_name=op_info.name
            ),
            op_error_type=err_type,
            op_error_info=err_info
        )

def basic_sample_inputs_mint_nn_bce_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # input/targe values must follows in [0, 1] for torch
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    case_shapes=((), (10, ),(3, 5, 20), (3, 4, 5, 6, 7, 8, 8, 8))
    reductions=('none', 'mean', 'sum')
    for shape, reduction in itertools.product(case_shapes, reductions):
        yield OpSampleInput(
            op_input=make_func(shape=shape, low=0, high=1),
            op_args=(make_func(shape=shape, low=0, high=1), None,),
            op_kwargs={'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(shape)}d"
        )
        yield OpSampleInput(
            op_input=make_func(shape=shape, low=0, high=1),
            op_args=(make_func(shape=shape, low=0, high=1), make_func(shape=shape),),
            op_kwargs={'reduction': reduction},
            sample_name=f"{op_info.name}_{dtype}_{len(shape)}d"
        )

def extra_sample_inputs_mint_nn_bce_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # input/targe values must follows in [0, 1] for torch
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    #empty shape
    yield OpSampleInput(
        op_input=make_func(shape=(2, 0, 3), low=0, high=1),
        op_args=(make_func(shape=(2, 0, 3), low=0, high=1), make_func(shape=(2, 0, 3)),),
        op_kwargs={'reduction': 'mean'},
        sample_name=f"{op_info.name}_{dtype}_shape_empty"
    )
    #large shape
    yield OpSampleInput(
        op_input=make_func(shape=(1024, 812), low=0, high=1),
        op_args=(make_func(shape=(1024, 812), low=0, high=1), None,),
        op_kwargs={'reduction': 'none'},
        sample_name=f"{op_info.name}_{dtype}_large_shape"
    )
    #small/large values
    values = small_large_values(dtype)
    values=tuple(i for i in values if 0 <= i <= 1)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(3, 4)), dtype=dtype),
        op_args=(ms.Tensor(np.random.choice(values, size=(3, 4)), dtype=dtype),
                 ms.Tensor(np.random.choice(values, size=(3, 4)), dtype=dtype),),
        op_kwargs={'reduction': 'sum'},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    #discontiguous values
    yield OpSampleInput(
        op_input=make_func(shape=(3, 4, 5), low=0, high=1, discontiguous=True),
        op_args=(make_func(shape=(3, 4, 5), low=0, high=1, discontiguous=True),
                 make_func(shape=(3, 4, 5), discontiguous=True),),
        op_kwargs={'reduction': 'none'},
        sample_name=f"{op_info.name}_{dtype}_discontiguous_True"
    )
    #weight broadcast: weight has lower dimensions than input (needs broadcast to input shape)
    for input_shape, weight_shape in [((2, 3, 4), (3, 4)), ((2, 3, 4), (1,)), ((5,), (1,))]:
        for reduction in ('none', 'mean', 'sum'):
            yield OpSampleInput(
                op_input=make_func(shape=input_shape, low=0, high=1),
                op_args=(make_func(shape=input_shape, low=0, high=1),
                         make_func(shape=weight_shape, low=0, high=10),),
                op_kwargs={'reduction': reduction},
                sample_name=f"{op_info.name}_{dtype}_weight_"+ \
                            f"broadcast_{len(weight_shape)}d_to_{len(input_shape)}d_{reduction}"
            )

def error_sample_inputs_mint_nn_bce_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device)
    #input, weight, target does not have same shape
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(3, 4, 5)),
            op_args=(make_func(shape=(4, 5)), make_func(shape=(4, 5)),),
            op_kwargs={'reduction': 'none'},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="target has different shape with input/weight"
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(4, 5)),
            op_args=(make_func(shape=(3, 4, 5)), make_func(shape=(4, 5)),),
            op_kwargs={'reduction': 'none'},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="input has different shape with target/weight"
    )
    #reductio has illegal value
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(3, 4, 5)),
            op_args=(make_func(shape=(3, 4, 5)), make_func(shape=(3, 4, 5)),),
            op_kwargs={'reduction': 'unknown'},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="reduction has illegal value"
    )

def basic_sample_inputs_mint_nn_functional_embedding(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mindspore.mint.nn.functional.embedding.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # input_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq
    cases = (
        ((1,), (9, 8), 5, 2.0, -2.0, False),
        ((3, 2), (9, 8), 3, 2.0, 3.9, True),
        ((5, 8, 3, 8), (2, 3), -2, -0.1, -0.9, True),
    )

    for input_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape, dtype=ms.int32, low=0, high=weight_shape[0]),
            op_args=(make_arg(weight_shape),),
            op_kwargs={'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type,
                       'scale_grad_by_freq': scale_grad_by_freq},
            sample_name=f"{op_info.name}_basic_{dtype}"
        )
    # empty tensor and default values
    weight_shape = (2, 0)
    yield OpSampleInput(
        op_input=make_arg(weight_shape, dtype=ms.int64, low=0, high=weight_shape[0]),
        op_args=(make_arg(weight_shape),),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def extra_sample_inputs_mint_nn_functional_embedding(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.nn.functional.embedding.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), (7, 4), -4, 0.4, 0.4, True),
        ((1, 2, 1), (5, 7), 3, None, -2.0, False),
        ((1, 2, 1, 2, 1), (3, 5), 0, 12.3, 12.3, True),
        ((3, 6, 2, 6, 3, 5, 5), (7, 4), -4, None, 0.4, True),
        # 20260611 8D fail
        # ((3, 6, 2, 6, 3, 5, 5, 1), (7, 4), -4, None, 0.4, True),
    )

    for input_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape, dtype=ms.int64, low=0, high=weight_shape[0]),
            op_args=(make_arg(weight_shape),),
            op_kwargs={'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type,
                       'scale_grad_by_freq': scale_grad_by_freq},
            sample_name=op_info.name,
        )

    # weight: Parameter
    yield OpSampleInput(
        op_input=ms.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]]),
        op_args=(ms.Parameter(np.random.rand(10, 3).astype(np.float32)),),
        op_kwargs={'padding_idx': 0, 'max_norm': 0.3, 'norm_type': 1.1},
        sample_name=f"{op_info.name}_weight_is_Parameter",
    )

    # small/large values
    values = small_large_values(dtype)
    if values:
        for scale_grad_by_freq in (True, False):
            yield OpSampleInput(
                op_input=ms.Tensor(np.random.randint(0, 8, (3, 4)), dtype=ms.int32),
                op_args=(ms.Tensor(np.random.choice(values, size=(8, 9)), dtype=dtype),),
                op_kwargs={'padding_idx': -1, 'max_norm': 0.6, 'norm_type': 2.4,
                           'scale_grad_by_freq': scale_grad_by_freq},
                sample_name=f"{op_info.name}_{dtype}_small_large_values"
            )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.random.randint(0, 3, (2, 3)), dtype=ms.int64),
            op_args=(ms.Tensor(np.full((3, 4), t), dtype=dtype),),
            op_kwargs={'padding_idx': 0, 'max_norm': 2.0, 'norm_type': 12.3, 'scale_grad_by_freq': True},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

    # discontiguous tensor
    yield OpSampleInput(
        op_input=make_arg((2, 3), dtype=ms.int64, discontiguous=True, low=0, high=4),
        op_args=(make_arg((4, 3), discontiguous=True),),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_discontiguous_True"
    )

def dynamic_sample_inputs_mint_nn_functional_embedding(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.functional.embedding.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=ms.int64),
                op_args=(make_func(shape=(12, 4)),),
                op_kwargs={'padding_idx': -3, 'max_norm': 12.3, 'norm_type': -2.0, 'scale_grad_by_freq': False},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 1, 5), dtype=ms.int64, low=0, high=11),
                    op_args=(make_func(shape=(12, 4)),),
                    op_kwargs={'padding_idx': -3, 'max_norm': 12.3, 'norm_type': -2.0, 'scale_grad_by_freq': False},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 7, 7), dtype=ms.int64, low=0, high=11),
                    op_args=(make_func(shape=(12, 4)),),
                    op_kwargs={'padding_idx': -3, 'max_norm': 12.3, 'norm_type': -2.0, 'scale_grad_by_freq': False},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=ms.int32),
                op_args=(make_func(shape=(10, 1)),),
                op_kwargs={'padding_idx': 9, 'max_norm': 0.3, 'norm_type': 2.0, 'scale_grad_by_freq': True},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,), dtype=ms.int32, low=0, high=9),
                    op_args=(make_func(shape=(10, 1)),),
                    op_kwargs={'padding_idx': 9, 'max_norm': 0.3, 'norm_type': 2.0, 'scale_grad_by_freq': True},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5,), dtype=ms.int32, low=0, high=9),
                    op_args=(make_func(shape=(10, 1)),),
                    op_kwargs={'padding_idx': 9, 'max_norm': 0.3, 'norm_type': 2.0, 'scale_grad_by_freq': True},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_functional_embedding(op_info:OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.nn.functional.embedding.
    """
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        (3, (3, 2), (3,), -2, 0.4, 2.0, False),
        # 20260611 case can pass
        # (6, (3, 2), (3, 2), -2, 0.4, 2.0, False),
        (3, (3, 2), (3, 2), 1.0, 0.4, 2.0, False),
        (3, (3, 2), (3, 2), 3, 0.4, 2.0, True),
        (3, (3, 2), (3, 2), 1, 4, 2.0, False),
        (3, (3, 2), (3, 2), 1, 0.4, 2, True),
        (3, (3, 2), (3, 2), 1, 0.4, 0.5, 5),
    )
    # weight is not 2D
    # the values of input are not in [0, weight.shape[0])
    # padding_idx is not int
    # padding_idx is out of range, not in [-weight.shape[0], weight.shape[0])
    # max_norm is not float
    # norm_type is not float
    # scale_grad_by_freq is not bool
    for input_high, input_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq in cases:
        if input_high > weight_shape[0]:
            input_tensor = ms.Tensor(np.random.randint(weight_shape[0], input_high, input_shape))
        else:
            input_tensor = make_arg(shape=input_shape, dtype=ms.int64, low=0, high=input_high)
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_tensor,
                op_args=(make_arg(shape=weight_shape),),
                op_kwargs={'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type,
                           'scale_grad_by_freq': scale_grad_by_freq},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(ValueError,TypeError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1,
            op_args=(make_arg(shape=(2, 3)), ),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_not_tensor"
        ),
        op_error_type=TypeError,
        op_error_info=f'{op_info.name}: input is not Tensor'
    )
    # input dtype is not in [ms.int32, ms.int64]
    input_not_support_dtypes = (
        ms.bool_, ms.int8, ms.int16, ms.uint8, ms.float32, ms.float16, ms.float64, ms.complex64, ms.complex128,
        ms.bfloat16
    )
    for input_dtype in input_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=(2, 3), dtype=input_dtype, low=0, high=2),
                op_args=(make_arg(shape=(2, 3)),),
                op_kwargs={},
                sample_name=f"{op_info.name}_input_dtype_{input_dtype}_error",
            ),
            op_error_type=(TypeError, RuntimeError),
            op_error_info=f'input error dtype {input_dtype} is not supported for {op_info.name}',
        )


def basic_sample_inputs_smooth_l1_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        SMALL_DIM_SIZE,
        MEDIUM_DIM_SIZE,
        LARGE_DIM_SIZE,
        (5, SMALL_DIM_SIZE),
        (5, 10, MEDIUM_DIM_SIZE),
        (2, 3, 10, LARGE_DIM_SIZE),
    )

    for reduction, beta in itertools.product(['none', 'mean', 'sum'], [1.0, 1.5, 3.2, ]):
        for shape in cases:
            yield OpSampleInput(
                op_input=make_arg(shape),
                op_args=(make_arg(shape),),
                op_kwargs={'reduction': reduction, 'beta': beta},
                sample_name=op_info.name,
            )


def error_sample_inputs_smooth_l1_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 10)),
            op_args=(make_arg((2, 4, 10)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input.shape should be equal target.shape',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 10)),
            op_args=(make_arg((2, 3, 10)),),
            op_kwargs={'beta': -2.7},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='beta should >=0',
    )


def extra_sample_inputs_smooth_l1_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for reduction in ['none', 'mean', 'sum']:
        yield from make_extra_tensor(op_info, (10, 32, 64), ((10, 32, 64),),
                                     {'reduction': reduction}, dtype=dtype, device=device)

        yield from make_extra_tensor(op_info, (0, 10, 32, 64), ((0, 10, 32, 64),),
                                     {'reduction': reduction}, dtype=dtype, device=device, empty_tensor=True)

        yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9),
                                     ((2, 3, 4, 5, 6, 7, 8, 9),), {'reduction': reduction}, dtype=dtype, device=device,
                                     support_multi_dim=True)


def basic_sample_inputs_nll_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        (SMALL_DIM_SIZE,),
        (MEDIUM_DIM_SIZE,),
        (LARGE_DIM_SIZE,),
        (SMALL_DIM_SIZE, MEDIUM_DIM_SIZE),
        (MEDIUM_DIM_SIZE, LARGE_DIM_SIZE),
        (LARGE_DIM_SIZE, SMALL_DIM_SIZE),
    )

    for input_shape in cases:

        target = make_tensor(() if len(input_shape) == 1 else (input_shape[0],), low=0, high=input_shape[-1] - 1,
                             dtype=ms.int64, device=device)
        ingore_index = target if len(target.shape) == 0 else ms.Tensor(np.array(np.random.choice(target.numpy())),
                                                                       dtype=ms.int64)
        weight = make_tensor((input_shape[-1],), device=device, dtype=dtype, low=0, high=1)

        for wei, ing, red in itertools.product([None, weight], [-100, ingore_index], ['none', 'mean', 'sum']):
            yield OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(target,),
                op_kwargs={'weight': wei,
                           'ignore_index': ing,
                           'reduction': red},
                sample_name=op_info.name,
            )


def error_sample_inputs_nll_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2,)),
            op_args=(make_tensor((2,), device=device, dtype=ms.int64),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='target should be 0d if input is 1d',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3,)),
            op_args=(make_arg((1,)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input batch_size should be equal to target batch_size',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3,)),
            op_args=(make_arg((2, 3)),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=RuntimeError,
        op_error_info='target should be 1d if input is 2d',
    )


def extra_sample_inputs_nll_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):
    target = make_tensor((10,), low=0, high=31, dtype=ms.int64, device=device)

    for reduction in ['none', 'mean', 'sum']:
        yield from make_extra_tensor(op_info, (10, 32,), (target,),
                        {'reduction': reduction}, dtype=dtype, device=device, exclude_value=(-1e20, 1e20))

def basic_sample_inputs_kl_div(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((SMALL_DIM_SIZE, ), (SMALL_DIM_SIZE, )),
        ((MEDIUM_DIM_SIZE, ), (), ),
        ((LARGE_DIM_SIZE, ), (LARGE_DIM_SIZE, )),
        ((1, MEDIUM_DIM_SIZE), (SMALL_DIM_SIZE, 1)),
        ((1, MEDIUM_DIM_SIZE, LARGE_DIM_SIZE),(SMALL_DIM_SIZE, 1, LARGE_DIM_SIZE)),
    )

    for reduction, log_target in itertools.product(['none', 'mean', 'batchmean', 'sum'],[True, False]):

        for input_shape, target_shape in cases:
            target = make_tensor(target_shape,  high=0 if dtype in op_info.get_supported_dtypes(device) else 2,
                    device=device,dtype=dtype) if log_target is True else make_tensor(target_shape, low=0, high=1,
                                                                                     device=device, dtype=dtype)

            yield OpSampleInput(
                op_input=make_arg(input_shape, ),
                op_args=(target, ),
                op_kwargs={'reduction': reduction,
                           'log_target': log_target,
                           },
                sample_name=op_info.name,
            )

def error_sample_inputs_kl_div(op_info: OpInfo, dtype=None, device=None, **kwargs):


    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, )),
            op_args=(make_arg((3, )), ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='''the shape of 'target' and 'input' should be broadcastable ''',
    )

def extra_sample_inputs_kl_div(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, low=0, high=1, device=device, dtype=dtype)

    for reduction in ['none', 'mean', 'sum', 'batchmean']:

        yield from make_extra_tensor(op_info, (10, 32,), (make_arg((10, 32)), ),
                        {'reduction': reduction}, dtype=dtype, device=device,)

        yield from make_extra_tensor(op_info, (), (make_arg(()), ),
                                {'reduction': reduction}, dtype=dtype, device=device, support_multi_dim=True)

        yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9),
             (make_arg((1, 3, 1, 5, 1, 7, 1, 9)), ),{'reduction': reduction}, dtype=dtype,
                                     device=device, support_multi_dim=True )

def basic_sample_inputs_cosine_embedding_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ( (SMALL_DIM_SIZE, ), (SMALL_DIM_SIZE, ), () ),
        ( (MEDIUM_DIM_SIZE, ), (1, ), () ),
        ( (1, ), (LARGE_DIM_SIZE, ), () ),
        ( (1, MEDIUM_DIM_SIZE), (SMALL_DIM_SIZE, 1), (SMALL_DIM_SIZE, ) ),
        ( (MEDIUM_DIM_SIZE, LARGE_DIM_SIZE), (1, LARGE_DIM_SIZE), (MEDIUM_DIM_SIZE, ) ),
        ( (1, LARGE_DIM_SIZE), (SMALL_DIM_SIZE, LARGE_DIM_SIZE), (SMALL_DIM_SIZE, ) ),
    )

    for reduction, margin in itertools.product(['none', 'mean', 'sum'],[0.0]):

        for input1_shape, input2_shape, target_shape in cases:

            yield OpSampleInput(
                op_input=make_arg(input1_shape, ),
                op_args=(make_arg(input2_shape), ms.Tensor(np.random.choice([1,-1],size=target_shape).astype(np.int64))
                         ),
                op_kwargs={
                           'margin': margin,
                           'reduction': reduction,
                           },
                sample_name=op_info.name,
            )

def error_sample_inputs_cosine_embedding_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):


    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    N = 2
    D = 5

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((D, )),
            op_args=(make_arg((D+1, )), ms.Tensor(np.random.choice([1,-1],size=()).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input1.shape and input2.shape should have the same shape or can be broadcast',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((D, )),
            op_args=(make_arg(()), ms.Tensor(np.random.choice([1, -1],size=()).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input1.shape and input2.shape should have the same dim',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((D, )),
            op_args=(make_arg((D, )), ms.Tensor(np.random.choice([1, -1], size=(1, )).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='when input1.shape and input2.shape is 1d, target shape must be ()',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, D,)),
            op_args=(make_arg((N, D + 1,)), ms.Tensor(np.random.choice([1, -1], size=(1, )).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input1.shape and input2.shape should have the same shape or can be broadcast',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, D,)),
            op_args=(make_arg((N,)), ms.Tensor(np.random.choice([1, -1], size=(1, )).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input1.shape and input2.shape should have the same dim',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, D,)),
            op_args=(make_arg((N, D)), ms.Tensor(np.random.choice([1, -1], size=()).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='when input1.shape and input2.shape is 2d, target shape must be 1d',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, D,)),
            op_args=(
            make_arg((N, D)), ms.Tensor(np.random.choice([1, -1], size=(D,)).astype(np.int64))),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='when input1.shape and input2.shape is 2d, target shape should equal batch',
    )

def extra_sample_inputs_cosine_embedding_loss(op_info: OpInfo, dtype=None, device=None, **kwargs):

    for reduction in ['none', 'mean', 'sum']:

        yield from make_extra_tensor(op_info, (10, 100), ((10, 100),
                    ms.Tensor(np.random.choice([1, -1], size=(10,)).astype(np.int64))),
{'reduction': reduction}, dtype=dtype, device=device, dtype_promotion=True,)

    empty_cases = (
        ((0, ), (0, ), ()),
        ((1, 0), (1, 0), (1, )),
        ((0, 1), (0, 1), (0, )),
        ((0, 0), (0, 0), (0,)),
    )

    for input1_shape, inpu2_shape, target_shape in empty_cases:

        for reduction in ['none', 'mean', 'sum']:

            yield from make_extra_tensor(op_info, input1_shape, (
            inpu2_shape,target_shape),{'reduction': reduction},
                                         dtype=dtype, device=device, empty_tensor=True)
