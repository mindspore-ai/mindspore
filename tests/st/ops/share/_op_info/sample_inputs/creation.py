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
- sample_inputs for creation-types ops
"""
import functools

import numpy as np
import mindspore as ms
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_info import  basic_reference_inputs_unary_op_common_func

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array
)

from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    small_large_values,
)

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

def basic_sample_inputs_mint_linspace(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((3,), 0, 6),
        ((2,), 6, -3),
        ((1,), 0.4, 0.4),
    )

    for steps, start, end in cases:
        yield OpSampleInput(
            op_input=make_func(shape=steps),
            op_args=(start, end),
            op_kwargs={'dtype': dtype},
            sample_name=f"{op_info.name}_{dtype}",
        )

def extra_sample_inputs_mint_linspace(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((), -2.0, -2.0),
        ((16,), 6.9, 10),
        ((19,), 6, -3),
        ((22,), -21, -2.0),
        ((38,), -12.3, 12.3),
        ((50,), 0, 6.5),
    )

    for steps, start, end in cases:
        if dtype is ms.uint8 and start < 0 and end == start:
            # for torch cannot be converted to type uint8_t without overflow
            dtype = ms.int8
        yield OpSampleInput(
            op_input=make_func(shape=steps),
            op_args=(start, end),
            op_kwargs={'dtype': dtype},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    yield OpSampleInput(
        op_input=make_func(shape=()),
        op_args=(-1, 3),
        op_kwargs={},
        sample_name=f"{op_info.name}_0d",
    )
    # large mix small values
    values = small_large_values(dtype)
    if values:
        start_end = ms.Tensor(np.random.choice(values))
        yield OpSampleInput(
            op_input=make_func(shape=(6,)),
            op_args=(start_end, start_end),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_large_mix_small_values",
        )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            start_end = ms.Tensor(t)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5,), t), dtype=dtype),
                op_args=(start_end, start_end),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_mint_linspace(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(0.4, -2.0),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2,)),
                    op_args=(12.3, 0.4),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(28,)),
                    op_args=(12.3, -2.0),
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
                op_args=(12.3, 0.4),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5,)),
                    op_args=(12.3, 5.4),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10,)),
                    op_args=(0.4, 13.6),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_linspace(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    test_cases = [
        # start is not float/int
        ((5,), (1, 2), 2, ms.int32),
        # end is not float/int
        ((5,), 1, 2j, None),
        # dtype is not ms.dtype
        ((5,), 1, 2, np.float32),
    ]
    for steps, start, end, case_dtype in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=steps),
                op_args=(start, end),
                op_kwargs={'dtype': case_dtype},
                sample_name=op_info.name,
            ),
            op_error_type=TypeError,
            op_error_info=f"{op_info.name}_{case_dtype}_error_params"
        )
    # steps is float/0/negative
    steps_cases = (1.0, 0, None)
    for case_input in steps_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=case_input,
                op_args=(1, 3),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f"{op_info.name}_steps_{case_input}_error"
        )
    # Tensor out not implemented for bool_/complex64/complex128
    output_not_support_dtypes = (
        ms.bool_, ms.complex64, ms.complex128
    )
    for case_dtype in output_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=(3,)),
                op_args=(1, 3),
                op_kwargs={'dtype': case_dtype},
                sample_name=f"{op_info.name}_dtype_{case_dtype}_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'error dtype {case_dtype} is not supported for {op_info.name}',
        )

def extra_sample_inputs_mint_full(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    # currently, fill_value does not supported complex and bfloat16
    if dtype.is_complex or dtype is ms.bfloat16:
        fill_dtype = ms.float32
    else:
        fill_dtype = dtype
    cases = (
        ((), ms.Tensor([1.0])),
        ((3, 5, 1), ms.Tensor(shape=(), dtype=fill_dtype)),
        ((6, 1, 3, 6), ms.Tensor(np.random.rand(1,))),
        ((2, 2, 3, 6, 5, 6, 7, 8), 12.2),
    )

    for size, fill_value in cases:
        yield OpSampleInput(
            op_input=make_func(shape=size),
            op_args=(fill_value,),
            op_kwargs={'dtype': dtype},
            sample_name=f"{op_info.name}_{dtype}_extra",
        )
    # large mix small values
    values = small_large_values(dtype)
    if values:
        input_np_array = np.random.choice(values, size=(1, 3), replace=False)
        np_array = np.random.choice(values, size=(1,), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(make_tensor_with_np_array(np_array, dtype=fill_dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype),
                op_args=(make_tensor_with_np_array(np.full((1,), t), dtype=fill_dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_mint_full(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(-3,),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2)),
                    op_args=(-3,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 2)),
                    op_args=(-3,),
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
                op_args=(ms.Tensor(shape=None, dtype=ms.float16),),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 7, 2)),
                    op_args=(ms.Tensor(shape=(1,), dtype=ms.float16),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 7, 8)),
                    op_args=(ms.Tensor(shape=(), dtype=ms.float16),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_full(op_info:OpInfo, dtype=None, device=None, **kwargs):
    yield from error_sample_inputs_mint_full_like(op_info, dtype, device, **kwargs)
    # size has negative value and not list or tuple
    # size is not tuple[int]/list[int]
    n_cases = (None, 2.2)
    for op_input in n_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_size_error"
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f'{op_info.name}: size is not supported negative and not list or tuple.'
        )

def basic_sample_inputs_mint_full_like(op_info:OpInfo, dtype=None, device=None, **kwargs):
    yield from basic_sample_inputs_mint_full(op_info, dtype, device, **kwargs)

def extra_sample_inputs_mint_full_like(op_info:OpInfo, dtype=None, device=None, **kwargs):
    yield from extra_sample_inputs_mint_full(op_info, dtype, device, **kwargs)

def dynamic_sample_inputs_mint_full_like(op_info:OpInfo, dtype=None, device=None, **kwargs):
    yield from dynamic_sample_inputs_mint_full(op_info, dtype, device, **kwargs)

def error_sample_inputs_mint_full_like(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    test_cases = [
        # fill_value is complex
        ((4, 5), 1.0j, ms.float16),
        # fill_value is 3D
        ((4, 5), ms.Tensor(np.random.randn(2, 3, 4)), ms.float16),
        # fill_value tensor with more than one element
        ((4, 5), ms.Tensor(np.random.randn(2,)), ms.float16),
        # dtype is not ms.dtype
        ((4, 5), 1, np.float32),
    ]
    for size, fill_value, case_dtype in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=size),
                op_args=(fill_value,),
                op_kwargs={'dtype': case_dtype},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f"{op_info.name}_{case_dtype}_error_params"
        )
    # op_input is not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1.0,
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_error"
        ),
        op_error_type=(TypeError, AttributeError),
        op_error_info=f'{op_info.name}: input is not tensor.'
    )

def basic_sample_inputs_mint_full(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((3, 5), 1.0),
        ((3, 5), 0),
        ((2, 1), 50),
        ((1, 3, 6), -2.3),
    )

    for size, fill_value in cases:
        if dtype is ms.uint8 and fill_value < 0:
            # for torch cannot be converted to type uint8_t without overflow
            fill_value = 2.3
        yield OpSampleInput(
            op_input=make_func(shape=size),
            op_args=(fill_value,),
            op_kwargs={'dtype': dtype},
            sample_name=f"{op_info.name}_{dtype}",
        )
    # empty tensor
    x_shape = (2, 0)
    yield OpSampleInput(
        op_input=make_func(x_shape),
        op_args=(0.4,),
        op_kwargs={},
        sample_name=f"{op_info.name}_empty_tensor",
    )

def basic_sample_inputs_mint_eye(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((), None),
        ((2, 23), 1),
        ((2, 23), 23),
        ((1, 5), 35),
    )

    for input_shape, m in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={'m': m, 'dtype': dtype},
            sample_name=f"{op_info.name}_{dtype}",
        )

def extra_sample_inputs_mint_eye(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((33, 23), 1),
        ((50, 2), 50),
        ((23, 23), 17),
        ((0, 2), 0),
    )

    for input_shape, m in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={'m': m, 'dtype': dtype},
            sample_name=f"{op_info.name}_{dtype}",
        )
    # int large mix small values
    values = (1, 31, 63, 127, 128, 195, 254, 1024)
    for input_n in values:
        yield OpSampleInput(
            op_input=make_func(shape=(input_n,)),
            op_args=(),
            op_kwargs={'m': np.random.choice(values)},
            sample_name=f"{op_info.name}_large_mix_small_values",
        )

def dynamic_sample_inputs_mint_eye(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic inputs for mint.eye.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6,)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9,)),
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
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 2)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 5)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_eye(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        # m is not int
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), 1.0, ms.int32),
        # m is negative
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), -1, ms.int32),
        # dtype is not ms.dtype
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), 1, np.float32),
    ]
    for op_inputs, m, case_dtype in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(),
                op_kwargs={'m': m, 'dtype': case_dtype},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f"{op_info.name}_{case_dtype}_error_params"
        )
    # n is negative and not int
    n_cases = (None, 1.0)
    for op_input in n_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_n_error"
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f'{op_info.name}: n is not supported negative and not int.'
        )
    # Tensor out not implemented for float64/complex64/complex128
    output_not_support_dtypes = (
        ms.float64, ms.complex64, ms.complex128
    )
    for case_dtype in output_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=ms.Tensor([[1, 2, 3], [4, 5, 6]]),
                op_args=(),
                op_kwargs={'dtype': case_dtype},
                sample_name=f"{op_info.name}_dtype_{case_dtype}_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'error dtype {case_dtype} is not supported for {op_info.name}',
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
    sample_name = f"{op_info.name}_{dtype}"
    samples = int_samples
    # Only add float_samples when output dtype is floating-point;
    # for integer dtypes, mixing float ranges can cause length mismatches vs MindSpore behavior.
    if dtype is not None and getattr(dtype, "is_floating_point", False):
        samples += float_samples
    if dtype not in (ms.int8, ms.uint8):
        samples += large_samples
        sample_name += "_large_sample"

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
            sample_name=sample_name,
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

def basic_sample_inputs_mint_bincount(op_info: OpInfo, dtype=None, device=None, **kwargs):
    base_input = np.array([0, 2, 1, 2, 1], dtype=np.int64)
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(base_input, dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_no_weights",
    )

    weights = np.array([0.5, -1.0, 2.0, 3.5, 0.25], dtype=np.float32)
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(base_input, dtype=dtype, device=device),
        op_args=(make_tensor_with_np_array(weights, dtype=ms.float32, device=device),),
        op_kwargs={"minlength": 8},
        sample_name=f"{op_info.name}_{dtype}_float_weights",
    )

    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.array([], dtype=np.int64), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={"minlength": 6},
        sample_name=f"{op_info.name}_{dtype}_empty_input",
    )

def extra_sample_inputs_mint_bincount(op_info: OpInfo, dtype=None, device=None, **kwargs):
    if dtype in (ms.uint8, ms.int16, ms.int32, ms.int64):
        large_bin_input = np.array([163, 196, 148, 253, 185, 167, 135, 211, 144, 61], dtype=np.int64)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(large_bin_input, dtype=dtype, device=device),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_large_bin_range",
        )

        weights = np.linspace(0.1, 1.0, num=10, dtype=np.float32)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(large_bin_input, dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(weights, dtype=ms.float32, device=device),),
            op_kwargs={"minlength": 260},
            sample_name=f"{op_info.name}_{dtype}_large_bin_range_weighted",
        )

    discontiguous_input = make_tensor_with_np_array(
        np.array([0, 4, 1, 4, 2, 4, 3, 4], dtype=np.int64), dtype=dtype, device=device
    )[::2]
    discontiguous_weights = make_tensor_with_np_array(
        np.array([1, 9, 2, 9, 3, 9, 4, 9], dtype=np.int32), dtype=ms.int32, device=device
    )[::2]
    yield OpSampleInput(
        op_input=discontiguous_input,
        op_args=(discontiguous_weights,),
        op_kwargs={"minlength": 6},
        sample_name=f"{op_info.name}_{dtype}_discontiguous_int_weights",
    )

def dynamic_sample_inputs_mint_bincount(op_info: OpInfo, dtype=None, device=None, **kwargs):
    compile_weights_dtype = ms.float32
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(ms.Tensor(shape=(None,), dtype=compile_weights_dtype),),
                op_kwargs={"minlength": 12},
                sample_name=f"{op_info.name}_{dtype}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_tensor_with_np_array(np.array([0, 2, 1, 2], dtype=np.int64), dtype=dtype,
                                                       device=device),
                    op_args=(make_tensor_with_np_array(np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32),
                                                       dtype=compile_weights_dtype, device=device),),
                    op_kwargs={"minlength": 12},
                    sample_name=f"{op_info.name}_{dtype}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_tensor_with_np_array(np.array([3, 1, 4, 1, 5, 9], dtype=np.int64), dtype=dtype,
                                                       device=device),
                    op_args=(make_tensor_with_np_array(np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32),
                                                       dtype=compile_weights_dtype, device=device),),
                    op_kwargs={"minlength": 12},
                    sample_name=f"{op_info.name}_{dtype}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=compile_weights_dtype),),
                op_kwargs={"minlength": 12},
                sample_name=f"{op_info.name}_{dtype}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_tensor_with_np_array(np.array([0, 1, 1], dtype=np.int64), dtype=dtype,
                                                       device=device),
                    op_args=(make_tensor_with_np_array(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                                       dtype=compile_weights_dtype, device=device),),
                    op_kwargs={"minlength": 12},
                    sample_name=f"{op_info.name}_{dtype}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_tensor_with_np_array(np.array([2, 7, 1, 8, 2, 8, 1], dtype=np.int64), dtype=dtype,
                                                       device=device),
                    op_args=(make_tensor_with_np_array(np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5], dtype=np.float32),
                                                       dtype=compile_weights_dtype, device=device),),
                    op_kwargs={"minlength": 12},
                    sample_name=f"{op_info.name}_{dtype}_dynamic_rank_running_input",
                ),
            ),
        )

def error_inputs_mint_bincount(op_info: OpInfo, dtype=None, device=None, **kwargs):
    valid_input = make_tensor_with_np_array(np.array([0, 1, 1], dtype=np.int64), dtype=ms.int32, device=device)
    valid_weights = make_tensor_with_np_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=ms.float32,
                                              device=device)
    error_cases = (
        ([0, 1, 1], (), {}, TypeError, "input_not_tensor"),
        (valid_input, ([1.0, 2.0, 3.0],), {}, TypeError, "weights_not_tensor"),
        (valid_input, (), {"minlength": True}, TypeError, "minlength_bool"),
        (make_tensor_with_np_array(np.array([[0, 1], [1, 2]], dtype=np.int64), dtype=ms.int32, device=device),
         (), {}, (ValueError, RuntimeError), "input_not_1d"),
        (valid_input,
         (make_tensor_with_np_array(np.array([1.0, 2.0], dtype=np.float32), dtype=ms.float32, device=device),),
         {}, (ValueError, RuntimeError), "weights_shape_mismatch"),
        (make_tensor_with_np_array(np.array([0, -1, 2], dtype=np.int64), dtype=ms.int32, device=device),
         (), {}, (ValueError, RuntimeError), "negative_input"),
        (make_tensor_with_np_array(np.array([], dtype=np.int64), dtype=ms.int32, device=device),
         (valid_weights,), {}, (ValueError, RuntimeError), "empty_input_weights_mismatch"),
    )
    for op_input, op_args, op_kwargs, op_error_type, error_info in error_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=op_args,
                op_kwargs=op_kwargs,
                sample_name=f"{op_info.name}_{error_info}",
            ),
            op_error_type=op_error_type,
            op_error_info=f"{op_info.name}_{error_info}",
        )

def basic_sample_inputs_mint_ones(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    sizes = (
        (),
        2,
        (M,),
        (M, S),
        (S, S, L),
        (3, 1, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        [2, 1, 2, 2, 1, 2],
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )
    for size in sizes:
        yield OpSampleInput(
             op_input=size,
             op_kwargs={'dtype': dtype},
             sample_name=op_info.name,
        )

def error_sample_inputs_mint_ones(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = ('2', ['3'], ('1'),)
    for op_input in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_kwargs={'dtype': dtype},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.ones error values'
        )
