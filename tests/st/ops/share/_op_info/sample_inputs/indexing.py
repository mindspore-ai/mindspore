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
- sample_inputs for indexing-types ops
"""
import functools

import numpy as np
import mindspore as ms
from mindspore import mint, mutable
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_info import (
    basic_reference_inputs_unary_op_common_func,
    extra_reference_inputs_unary_op_common_func,
)
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array,
)
from tests.st.ops.share._op_info.sample_inputs.common import make_extra_tensor
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_extra_values,
    small_large_values
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
        input_values = np.array(_float_small_values).reshape((1, len(_float_small_values), 1))
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


def basic_reference_inputs_tensor_choose_func(op_info: OpInfo, dtype, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    base_shapes = (
        ((S,),make_arg((S, 1)), "clip"),
        ((M, S),make_arg((1, S)), "wrap"),
        ((S, S, L), make_arg((S, L)), "wrap"),
        ((2, 0, 3, 2), [0, 1, 2], "clip"),#list choices
        ((2, 3, 4, 1, 2), make_arg((2, 1, 1, 1, 1)), "wrap"),
        ((2, 1, 2, 2, 1, 2), (2, 2), "clip"),
        ((2, 1, 2, 2, 1, 2, 1), make_arg((2, 1, 1, 1, 1)), "wrap"),
    )
    #0d < input <8d
    for input_shape, choices, modes in base_shapes:
        _input = make_arg(input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(choices, modes,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_tensor_choose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(make_arg(input_shape),),
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
            op_args=(make_arg(input_shape),),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 1, 3), t), dtype=dtype, device=device),
                    op_args=(make_tensor_with_np_array(np.ones((1, 1, 3)), dtype=dtype, device=device),),
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

def dynamic_sample_inputs_tensor_choose(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic sample inputs for Tensor.choose.
    Consider both dynamic_shape and dynamic_rank.
    """
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
                    op_args=(make_func(shape=(2, 3, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4 ,5, 7)),
                    op_args=(make_func(shape=(4 ,5, 5)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3D",
                ),
            ),
        )

def error_inputs_other_op_tensor_choose_func(op_info, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (2, make_arg((S,)), "clip"),#input is not tensor
        (make_arg((M, S)), 1.0, "clip"),#choices not tuple, list/tensor
        (make_arg((1024, 1)), make_arg((1024,)), "raise"),#raise error
        (make_arg((2, 0)), make_arg((2,)), True),#mode not str
        (make_arg((2, 0)), make_arg((2,)), "jaksn"),#mode values error
    )
    for input_shape, choices, modes in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(choices, modes),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.choose error values')

def basic_reference_inputs_tensor_index_add_func(op_info: OpInfo, dtype, device=None,**kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    make_arg_index = functools.partial(make_tensor, dtype=ms.int64, device=device, low=None, high=None)
    shapes = (
        ((S,), 0, make_arg_index((M,), low=0, high=3), make_arg((M,)), 1),
        ((M, S), 0, make_arg_index((S,), low=0, high=3), make_arg((S, S)), 1),
        ((S, S, S), -1, make_arg_index((1,), low=0, high=3), make_arg((S, S, 1)), 0.8),
        ((2, 2, 2, 2), -2, make_arg_index((1,), low=0, high=2), make_arg((2, 2, 1, 2)), 0.5),
        ((3, 3, 3, 3), -1, make_arg_index((1,), low=0, high=3), make_arg((3, 3, 3, 1)), 1),
        ((L, L, L, L, L), 2, make_arg_index((1,), low=0, high=3), make_arg((L, L, 1, L, L)), 1),
        ((1, 1, 1, 1, 1, 1), -2, make_arg_index((2,), low=0, high=1), make_arg((1, 1, 1, 1, 2, 1)), 1.2),
        ((2, 2, 2, 2, 2, 2, 2), -2, make_arg_index((1,), low=0, high=2), make_arg((2, 2, 2, 2, 2, 1, 2)), 1),
        ((4, 4, 4, 4, 4, 4, 4, 4), 3, make_arg_index((2,), low=0, high=3), make_arg((4, 4, 4, 2, 4, 4, 4, 4)), 1),
    )

    for input_shape, dims, indexes, sources, alphas in shapes:
        _input = make_arg(input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input=_input,
            op_args=(dims, indexes, sources,),
            op_kwargs={'alpha': alphas},
            sample_name=f"{op_info.name}_{dtype}_{slen}d", )

def extra_sample_inputs_tensor_index_add(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    cases = (
        ((0,), 0, make_arg((0,), dtype=ms.int64,), make_arg((0,)), 1),
        ((1029, 917), 1,  make_arg((1,), dtype=ms.int64, low=0, high=2), make_arg((1029, 1)), 1),
    )
    for input_shape, dims, indexes, sources, alphas in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input_shape),
            op_args=(dims, indexes, sources,),
            op_kwargs={'alpha': alphas},
            sample_name=f"{op_info.name}_{dtype}",
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(2, make_arg((2,), dtype=ms.int64, low=0, high=2),
                     make_arg((1, 1, 2)),),
            op_kwargs={'alpha': 1},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_shape = (1, 1, len(_large_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(2, make_arg((2,), dtype=ms.int64, low=0, high=1), make_arg((1, 1, 2)),),
            op_kwargs={'alpha': 1},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 1, 3), t), dtype=dtype, device=device),
                    op_args=(1, make_arg((2,), dtype=ms.int64, low=0, high=1), make_arg((1, 2, 3)),),
                    op_kwargs={'alpha': 1},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3, 4), discontiguous=discontiguous),
            op_args=(0, make_arg((1,), dtype=ms.int64, low=0, high=1), make_arg((1, 3, 4)),),
            op_kwargs={'alpha': 1},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_index_add(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (make_arg((2, 2, 2, 2, 2, 2, 2)), -2, make_arg((1,), dtype=ms.int8, low=0, high=2),
         make_arg((2, 2, 2, 2, 2, 1, 2),), 1),#index dtype not/int32/64
        (make_arg((3, 2)), 1.0, make_arg((1,), low=0, high=3), make_arg((3, 1)), 1),#dim not int dtype
        (make_arg((3, 2)), 1, make_arg((1,), low=0, high=3), False, 1),#source not Tensor
        (make_arg((3, 2)), 1, False, make_arg((3, 1)), 1),  # index not Tensor dtype
        (make_arg((3, 2)), 1, make_arg((1,), low=0, high=3), make_arg((3, 1)), "1"),  # alpha not int dtype
    )
    for input_shape, dims, indexes, sources, alphas in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(dims, indexes, sources,),
                op_kwargs={'alpha': alphas},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.fill_ error values')

def dynamic_sample_inputs_tensor_index_add(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(0, make_func((), dtype=ms.int64, low=0, high=2), make_func((),)),
                op_kwargs={'alpha': 1},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2, 2)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 2, 2),),),
                    op_kwargs={'alpha': 1},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 4, 4)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 4, 4),),),
                    op_kwargs={'alpha': 1},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0, make_func((), dtype=ms.int64, low=0, high=2), make_func((),)),
                op_kwargs={'alpha': 1},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 5),)),
                    op_kwargs={'alpha': 1},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3, 3)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 3, 3),)),
                    op_kwargs={'alpha': 1},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def basic_reference_inputs_tensor_index_copy_func(op_info: OpInfo, dtype, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    make_arg_index = functools.partial(make_tensor, dtype=ms.int64, device=device, low=None, high=None)
    shapes = (
        ((S,), 0, make_arg_index((M,), low=0, high=3), make_arg((M,))),
        ((M, S), 0, make_arg_index((S,), low=0, high=3), make_arg((S, S))),
        ((S, S, S), -1, make_arg_index((1,), low=0, high=3), make_arg((S, S, 1))),
        ((2, 2, 2, 2), -2, make_arg_index((1,), low=0, high=2), make_arg((2, 2, 1, 2))),
        ((3, 3, 3, 3), -1, make_arg_index((1,), low=0, high=3), make_arg((3, 3, 3, 1))),
        ((L, L, L, L, L), 2, make_arg_index((1,), low=0, high=3), make_arg((L, L, 1, L, L))),
        ((1, 1, 1, 1, 1, 1), -2, make_arg_index((2,), low=0, high=1), make_arg((1, 1, 1, 1, 2, 1))),
        ((2, 2, 2, 2, 2, 2, 2), -2, make_arg_index((1,), low=0, high=2), make_arg((2, 2, 2, 2, 2, 1, 2))),
        ((4, 4, 4, 4, 4, 4, 4, 4), 3, make_arg_index((2,), low=0, high=3), make_arg((4, 4, 4, 2, 4, 4, 4, 4))),
    )

    for input_shape, dims, indexes, tensors in shapes:
        _input = make_arg(input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input=_input,
            op_args=(dims, indexes, tensors,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d", )

def extra_sample_inputs_tensor_index_copy(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    cases = (
        ((0,), 0, make_arg((0,), dtype=ms.int64,), make_arg((0,))),
        ((1029, 917), 1,  make_arg((1,), dtype=ms.int64, low=0, high=2), make_arg((1029, 1))),
    )
    for input_shape, dims, indexes, tensors in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input_shape),
            op_args=(dims, indexes, tensors,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}",
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(2, make_arg((2,), dtype=ms.int64, low=0, high=2),
                     make_arg((1, 1, 2)),),
            op_kwargs={},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_shape = (1, 1, len(_large_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(2, make_arg((2,), dtype=ms.int64, low=0, high=1), make_arg((1, 1, 2)),),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 1, 3), t), dtype=dtype, device=device),
                    op_args=(1, make_arg((2,), dtype=ms.int64, low=0, high=1), make_arg((1, 2, 3)),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3, 4), discontiguous=discontiguous),
            op_args=(0, make_arg((1,), dtype=ms.int64, low=0, high=1), make_arg((1, 3, 4)),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_index_copy(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (make_arg((2, 2, 2, 2, 2, 2, 2)), -2, make_arg((1,), dtype=ms.int8, low=0, high=2),
         make_arg((2, 2, 2, 2, 2, 1, 2),)),#index dtype not/int32/64
        (make_arg((3, 2)), 1.0, make_arg((1,), low=0, high=3), make_arg((3, 1))),#dim not int dtype
        (make_arg((3, 2)), 1, make_arg((1,), low=0, high=3), False),#tensor not Tensor
        (make_arg((3, 2)), 1, False, make_arg((3, 1))),  # index not Tensor dtype
    )
    for input_shape, dims, indexes, tensors in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(dims, indexes, tensors,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.fill_ error values')

def dynamic_sample_inputs_tensor_index_copy(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(0, make_func((), dtype=ms.int64, low=0, high=2), make_func((),)),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2, 2)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 2, 2),),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 4, 4)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 4, 4),),),
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
                op_args=(0, make_func((), dtype=ms.int64, low=0, high=2), make_func((),)),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 5),)),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3, 3)),
                    op_args=(0, make_func((1,), dtype=ms.int64, low=0, high=2), make_func((1, 3, 3),)),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def basic_reference_inputs_tensor_index_fill(op_info: OpInfo, dtype, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    make_arg_index = functools.partial(make_tensor, dtype=ms.int32, device=device, low=None, high=None)
    shapes = (
        ((S,), 0, make_arg_index((1,), low=0, high=3), False),
        ((M, S), 0, make_arg_index((S,)), 0),
        ((S, S, S), -1, make_arg_index((1,), low=0, high=3), 1),
        ((3, 3, 3, 3), -1, make_arg_index((1,), low=0, high=3), True),
        ((1, 1, 1, 1, 1, 1), -2, make_arg_index((), low=0, high=1), 1.0),
        ((4, 4, 4, 4, 4, 4, 4, 4), 3, make_arg_index((2,), low=0, high=3), False),
    )
    shapes = list(shapes)
    if op_info.name == "Tensor.index_fill_":
        shapes.append(((2, 2, 2, 2), -2, make_arg_index((1,), low=0, high=2), make_arg((),)))
        shapes.append(((L, L, L, L, L), 2, make_arg_index((L,), low=0, high=3), make_arg((),)))
        shapes.append(((2, 2, 2, 2, 2, 2, 2), -2, make_arg_index((1,), low=0, high=2), make_arg((), )),)
    shapes = tuple(shapes)

    for input_shape, dims, indexes, values in shapes:
        _input = make_arg(input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input=_input,
            op_args=(dims, indexes, values,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d", )

def extra_sample_inputs_tensor_index_fill(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    cases = (
        ((0,), 0, make_arg((0,), dtype=ms.int32, ), False),
        ((20,), 0,  make_arg((), dtype=ms.int32), True),
        ((812,), 0,  make_arg((), dtype=ms.int32), False),
        ((1029, 917), 1,  make_arg((), dtype=ms.int32), False),
    )
    for input_shape, dims, indexes, values in cases:
        yield OpSampleInput(
            op_input=make_arg(shape=input_shape),
            op_args=(dims, indexes, values,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}",
        )

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=( 0, make_arg((), dtype=ms.int32, low=0, high=1), False,),
            op_kwargs={},
            sample_name=f"{op_info.name}_small",
        )

    # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_shape = (1, 1, len(_large_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=( 0, make_arg((), dtype=ms.int32, low=0, high=1), False,),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 1, 3), t), dtype=dtype, device=device),
                    op_args=( 0, make_arg((), dtype=ms.int32, low=0, high=1), False,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3, 4), discontiguous=discontiguous),
            op_args=( 0, make_arg((), dtype=ms.int32, low=0, high=1), False,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_index_fill(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (make_arg((2, 2, 2, 2, 2, 2, 2)), -2, make_arg((1,), dtype=ms.int8, low=0, high=2),
         make_arg((),)),#index dtype not/int32/64
        (make_arg((3,)), 0, make_arg((1, 1), dtype=ms.int32, low=0, high=3), 1.0),#index > 1D
        (make_arg((3,)), 0, make_arg((1,), dtype=ms.int32, low=0, high=3), make_arg((1, 1),)),  # values > 0D
        (1, 0, make_arg((1,), dtype=ms.int32, low=0, high=3), False),  # input not tensor
    )
    for input_shape, dims, indexes, values in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(dims, indexes, values,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.fill_ error values')

def dynamic_sample_inputs_tensor_index_fill(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(0, make_func((), dtype=ms.int32, low=0, high=2), False),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2, 2)),
                    op_args=(0, make_func((), dtype=ms.int32, low=0, high=2), False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 4, 4)),
                    op_args=(0, make_func((), dtype=ms.int32, low=0, high=2), False),
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
                op_args=(0, make_func((), dtype=ms.int32, low=0, high=2), False),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5)),
                    op_args=(0, make_func((), dtype=ms.int32, low=0, high=2), False),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3, 3)),
                    op_args=(0, make_func((), dtype=ms.int32,  low=0, high=2), False),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def basic_reference_inputs_tensor_index_put(op_info: OpInfo, dtype, device=None,**kwargs):

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    make_arg_indices = functools.partial(make_tensor, dtype=ms.int32, device=device, low=None, high=None)
    shapes = (
        (ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=dtype),
         [ms.Tensor(np.array([0, 1, 1]),dtype=ms.int64), ms.Tensor(np.array([1, 2, 1]), dtype=ms.int64),],
         ms.Tensor(np.array([3]), dtype=dtype), False),#2d
        (ms.Tensor(np.random.randn(8), dtype=dtype),
         [ms.Tensor(np.random.randint(0, 8, 10), dtype=ms.int32)],
         ms.Tensor(np.random.randn(10), dtype=dtype), False),#1d
        (ms.Tensor(np.random.randn(10, 5, 6), dtype=dtype),
         [ms.Tensor(np.random.randint(0, dim, 10), dtype=ms.int32) for dim in (10, 5, 6)],
         ms.Tensor(np.random.randn(1), dtype=dtype), False),  # 3d
        (ms.Tensor(np.random.randn(2, 12, 6, 3), dtype=dtype),
         [ms.Tensor(np.random.randint(0, dim, 10), dtype=ms.int32) for dim in (2, 12, 6, 3)],
         ms.Tensor(np.random.randn(10), dtype=dtype), True),  # 4d
        (ms.Tensor(np.random.randn(6, 7, 8, 9, 10), dtype=dtype),
         [ms.Tensor(np.random.randint(0, dim, 8), dtype=ms.int32) for dim in (6, 7, 8, 9, 10)],
         ms.Tensor(np.random.randn(8), dtype=dtype), False),  # 5d
        (ms.Tensor(np.random.randn(11, 6, 5, 4, 10, 2), dtype=dtype),
         [ms.Tensor(np.random.randint(0, dim, 12), dtype=ms.int32) for dim in (11, 6, 5, 4, 10, 2)],
         ms.Tensor(np.random.randn(12), dtype=dtype), False),  # 6d
        (ms.Tensor(np.random.randn(1, 3, 6, 9, 7, 12, 3), dtype=dtype),
         [ms.Tensor(np.random.randint(0, dim, 13), dtype=ms.int64) for dim in (1, 3, 6, 9, 7, 12, 3)],
         ms.Tensor(np.random.randn(13), dtype=dtype), True),  # 7d
        (ms.Tensor(np.random.randn(1, 3, 6, 9, 7, 12, 3), dtype=dtype),
         tuple(ms.Tensor(np.random.randint(0, dim, 13), dtype=ms.int64) for dim in (1, 3, 6, 9, 7, 12, 3)),
         ms.Tensor(np.random.randn(13), dtype=dtype), True),  # shape 0
        (make_arg((2, 1, 0, 2)), [make_arg_indices((1,)), make_arg_indices((1,)),], make_arg((1,)), False),
    )

    for input_shape, indices, values, accumulate in shapes:
        _input = input_shape
        yield OpSampleInput(
            op_input=_input,
            op_args=(indices, values, accumulate,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{input_shape.shape}", )

def extra_sample_inputs_tensor_index_put(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)

    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=([ms.Tensor(np.random.randint(0, dim, len(_float_small_values)), dtype=ms.int64) for
                      dim in (1, len(_float_small_values))],
         ms.Tensor(np.random.randn(len(_float_small_values)), dtype=dtype), True,),
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
            op_args=([ms.Tensor(np.random.randint(0, dim, len(_large_values)), dtype=ms.int64) for
                      dim in (1, len(_large_values))],
         ms.Tensor(np.random.randn(len(_large_values)), dtype=dtype), True,),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(np.full((1, 3), t), dtype=dtype, device=device),
                    op_args=([ms.Tensor(np.random.randint(0, dim, 3), dtype=ms.int64) for dim in (1, 3)],
         ms.Tensor(np.random.randn(3), dtype=dtype), True,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3, 4), discontiguous=discontiguous),
            op_args=([ms.Tensor(np.random.randint(0, dim, 4), dtype=ms.int64) for dim in (2, 3, 4)],
         ms.Tensor(np.random.randn(4), dtype=dtype), True,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_index_put(op_info, dtype=None, device=None, **kwargs):
    shapes = (
        (ms.Tensor(np.random.randn(8), dtype=ms.int64),
         [ms.Tensor(np.random.randint(0, 8, 10), dtype=ms.int32)],
         ms.Tensor(np.random.randn(10), dtype=ms.uint8), False), #input and values dtype not same
        (ms.Tensor(np.random.randn(8), dtype=ms.int64),
         ms.Tensor(np.random.randint(0, 8, 10), dtype=ms.int32),
         ms.Tensor(np.random.randn(10), dtype=ms.int64), False),  # indices not tuple/list
        (ms.Tensor(np.random.randn(8), dtype=ms.int64),
         [ms.Tensor(np.random.randint(0, 8, 10), dtype=ms.uint32)],
         ms.Tensor(np.random.randn(10), dtype=ms.int64), False),  # indices dtype not int32/64
        # (ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=dtype),
        #  [ms.Tensor(np.array([0, 1, 1]), dtype=ms.int64), ms.Tensor(np.array([1, 2, 1]), dtype=ms.int32),],
        #  ms.Tensor(np.array([3]), dtype=dtype), False),  # indices dtype not same, index_put_not raise error
        (ms.Tensor(np.random.randn(8), dtype=ms.int64),
         [ms.Tensor(np.random.randint(0, 8, 10), dtype=ms.int32)],
         ms.Tensor(np.random.randn(10), dtype=ms.int64), "False"),  # accumulate not bool
        (ms.Tensor(np.random.randn(8), dtype=ms.int64),
         [ms.Tensor(np.random.randint(0, 8, 10), dtype=ms.int32)],
         ms.Tensor(np.random.randn(1, 10, 10), dtype=ms.int64), False),  # values not 1D
        (ms.Tensor(np.random.randn(8, 1), dtype=ms.int64),
         [ms.Tensor(np.random.randn(1, 10), dtype=ms.int32)],
         ms.Tensor(np.random.randn(10), dtype=ms.int64), False),  # indices not 1D
        (ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=dtype),
         [ms.Tensor(np.array([0, 1, 1]), dtype=ms.int32), ms.Tensor(np.array([1, 2, 1]), dtype=ms.int32),
          ms.Tensor(np.array([1, 1, 1]), dtype=ms.int32),],
         ms.Tensor(np.array([3]), dtype=dtype), False),  # indices size > input dim
        (ms.Tensor(np.random.randn(10, 5), dtype=ms.int64),
         [ms.Tensor(np.random.randn(0, dim, 10), dtype=ms.int32) for dim in (10, 5)],
         ms.Tensor(np.random.randn(2), dtype=ms.int64), False),#values.shape equal to 1 or indices.shape
    )
    for input_shape, indices, values, accumulate in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(indices, values, accumulate,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.fill_diagonal_ error values')


def dynamic_sample_inputs_tensor_index_put(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(10, 5, 6)),
                    op_args=([ms.Tensor(np.random.randint(0, dim, 10), dtype=ms.int32) for dim in (10, 5, 6)],
                             ms.Tensor(np.random.randn(1), dtype=dtype), True,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 6)),
                    op_args=([ms.Tensor(np.random.randint(0, dim, 10), dtype=ms.int32) for dim in (2, 3, 6)],
                             ms.Tensor(np.random.randn(1), dtype=dtype), True,),
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
                    op_input=make_func(shape=(10, 5, 6)),
                    op_args=([ms.Tensor(np.random.randint(0, dim, 10), dtype=ms.int32) for dim in (10, 5, 6)],
                             ms.Tensor(np.random.randn(1), dtype=dtype), True,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 12, 6, 3)),
                    op_args=([ms.Tensor(np.random.randint(0, dim, 10), dtype=ms.int32) for dim in (2, 12, 6, 3)],
                             ms.Tensor(np.random.randn(10), dtype=dtype), True,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def basic_sample_inputs_mint_searchsorted(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor([1, 2, 2, 3, 4, 5, 5], dtype=dtype), ms.Tensor([2], dtype=dtype), False, False, None, None),#default
        (ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype), ms.Tensor([[2], [5]], dtype=dtype), True, False, None, None),
        (ms.Tensor(ms.Tensor(np.random.randint(-50, 50, (5, 4, 5)), dtype=dtype).asnumpy()),
         ms.Tensor(ms.Tensor(np.random.randint(-50, 50, (5, 4, 3)), dtype=dtype).asnumpy()), True, False, "left",
         ms.Tensor(np.argsort(ms.Tensor(np.random.randint(-50, 50, (5, 4, 5)), dtype=ms.int64), axis=-1),
                   dtype=ms.int64)),
        (ms.Tensor(ms.Tensor(np.random.randint(-50, 50, (5, 4, 5)), dtype=dtype).asnumpy()),
         ms.Tensor(ms.Tensor(np.random.randint(-50, 50, (5, 4, 3)), dtype=dtype).asnumpy()), False, True, "right",
         ms.Tensor(np.argsort(ms.Tensor(np.random.randint(-50, 50, (5, 4, 5)), dtype=ms.int64), axis=-1),
                   dtype=ms.int64)),
        (ms.Tensor(np.random.randint(0, 255, (2, 1, 2, 1, 2, 1, 2)), dtype=dtype),
         ms.Tensor(np.random.randint(0, 255, (2, 1, 2, 1, 2, 1, 2)), dtype=dtype), True, True, "right", None)
    ]

    for sorted_sequence, values, out_int32, right, side, sorter in cases:
        yield OpSampleInput(
            op_input=sorted_sequence,
            op_args=(values,),
            op_kwargs={"out_int32": out_int32, "right": right, "side": side, "sorter": sorter},
            sample_name=op_info.name,
        )

def error_inputs_mint_searchsorted(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (0, ms.Tensor([2], dtype=dtype), False, False, None, None),#sorted_sequence int
        (None, ms.Tensor([2], dtype=dtype), False, False, None, None),#sorted_sequence None
        (ms.Tensor([1, 2, 2, 3, 4, 5, 5], dtype=dtype), 1, False, False, None, None),#values int
        (ms.Tensor([1, 2, 2, 3, 4, 5, 5], dtype=dtype), None, False, False, None, None),#values none
        (ms.Tensor([1, 2, 2, 3, 4, 5, 5], dtype=dtype), ms.Tensor([2], dtype=dtype),
         "False", False, None, None),#out_int32 not bool
        (ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype), ms.Tensor([[2], [5]], dtype=dtype),
         True, "False", None, None),#right not bool
        # (ms.Tensor(np.random.randn(9,), dtype=dtype), ms.Tensor(np.random.randn(9,), dtype=dtype), True, False, "11313",
        #  ms.Tensor(np.random.randint(-50, 50, (9,)), dtype=dtype)),#side not right/left
        (ms.Tensor(np.random.randn(9, ), dtype=dtype), ms.Tensor(np.random.randn(9, ), dtype=dtype), True, True,
         "right", 12),#sorter not tensor
    ]

    for sorted_sequence, values, out_int32, right, side, sorter in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=sorted_sequence,
                op_args=(values,),
                op_kwargs={"out_int32": out_int32, "right": right, "side": side, "sorter": sorter},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.searchsorted error values',
        )

def extra_sample_inputs_mint_searchsorted(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        #inf/nan
        (ms.Tensor(ms.Tensor(np.full((5,), np.nan), dtype=dtype).asnumpy()),
         ms.Tensor(ms.Tensor(np.full((5,), np.nan), dtype=dtype).asnumpy()), False, True,
         "right", ms.Tensor(np.argsort(ms.Tensor(np.full((5,), np.nan), dtype=dtype), axis=-1), dtype=ms.int64)),
        (ms.Tensor(ms.Tensor(np.full((7, 9), np.inf), dtype=dtype).asnumpy()),
         ms.Tensor(ms.Tensor(np.full((7, 9), np.inf), dtype=dtype).asnumpy()), True, False,
         "left",ms.Tensor(np.argsort(ms.Tensor(np.full((7, 9), np.inf), dtype=dtype), axis=-1), dtype=ms.int64)),
        #discontinuous
        (ms.Tensor(np.random.randint(0, 255, (2, 1, 2, 1, 2, 1, 2)), dtype=dtype)[::2, ::2],
         ms.Tensor(np.random.randint(0, 255, (2, 1, 2, 1, 2, 1, 2)), dtype=dtype)[::2, ::2], True, True, "right", None),
        #shape 0
        (ms.Tensor(np.random.randn(3, 0, 2), dtype=dtype), ms.Tensor(np.random.randn(3, 0, 2), dtype=dtype),
        True, False, "left", None),
        #broadcast
        (ms.Tensor(np.random.randn(2, 1, 3), dtype=dtype), ms.Tensor(np.random.randn(2, 1, 3), dtype=dtype),
         True, False, "left", None),
    ]

    for sorted_sequence, values, out_int32, right, side, sorter in cases:
        yield OpSampleInput(
            op_input=sorted_sequence,
            op_args=(values,),
            op_kwargs={"out_int32": out_int32, "right": right, "side": side, "sorter": sorter},
            sample_name=op_info.name,
        )

    # small/large values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        out_int32, right, sorter = False, False, None
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_args=(make_tensor_with_np_array(np_array, dtype=dtype),),
                op_kwargs={"out_int32": out_int32, "right": right, "side": "left", "sorter": sorter},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
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

def dynamic_sample_inputs_tensor_flip_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=((0, 1),),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(1, 2, 4)),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 4)),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=((0, 1),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 4, 1)),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
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

    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((S, S), 1, M).to(ms.int64), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        # (_tensor(()), (0, zero.copy(), _tensor(()))),
        # (_tensor(()), (0, zero.copy(), ms.tensor(2.5).to(dtype))), input must dim > 1
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
        # mint.randint requires low < high (aligned with torch.randint). When
        # max_indices is 0 the requested shape always contains a zero dim, so
        # the result is an empty index tensor and the value range is irrelevant;
        # clamp high to 1 to satisfy randint instead of raising from<to error.
        high = max(max_indices, 1)
        return mint.randint(0, high, shape)
    sample_input = make_arg((3, 4, 5))
    src=ms.Tensor(np.random.uniform(0,10,size=(1,1,1)), dtype=ms.float32)
    #index is either not int32 nor int64
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=sample_input,
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
            op_args=(1, _gather_variable((1, 0, 1), 1), ms.Tensor((1, 0, 1), dtype=ms.float32)),
            op_kwargs={},
            sample_name=f"{op_info.name}_dim1_shape_empty_3d",
        ),
        op_error_type=(RuntimeError, ValueError),
        op_error_info='input shape can not be empty'
    )
    # input and src does not have same shape-dim
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=sample_input,
            op_args=(1, _gather_variable((3, 4), 4), make_arg((3, 4))),
            op_kwargs={},
            sample_name=f"{op_info.name}_shape_src_not_equal_input",
        ),
        op_error_type=(RuntimeError, ValueError),
        op_error_info='input is empty shape'
    )
    # src do not have the same dtype with input
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=sample_input,
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
            op_input=sample_input,
            op_args=(0, make_tensor((1, 2, 1), dtype=ms.int32, low=-2, high=0), src),
            op_kwargs={},
            sample_name=f"{op_info.name}_index_bigger_shape",
        ),
        op_error_type=RuntimeError,
        op_error_info='src has bigger shape than index'
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

    make_arg = functools.partial(make_tensor, device=device, low=0, high=1)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape((1, len(_float_small_values), 1))
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
        input_values = np.array(_large_values).reshape((1, len(_large_values), 1))
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
        input_x = make_arg(shape=(2, 2), dtype=dtype, discontiguous=discontiguous)
        index = make_arg(shape=(2, 2), dtype=ms.int64, discontiguous=discontiguous)
        yield OpSampleInput(
            op_input=input_x,
            op_args=(0, index),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_noncontiguous',
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
        (make_tensor(shape=(3, 2, 2, 2, 2, 2), dtype=dtype), {'dim': -1, "descending": False, "stable": True}),
        (make_tensor(shape=(2, 3), dtype=dtype), {'dim': 0, "descending": False, "stable": False}),
        #broadcast
        (make_tensor(shape=(1, 4, 3), dtype=dtype), {'dim': 1, "descending": True, "stable": True}),
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
    # Keep static reference samples shape-stable. The value-dependent
    # `num_classes=-1` branch is covered by dynamic one_hot samples below.
    num_classes = 10
    for shape in shapes:
        yield OpSampleInput(
            op_input=make_input(
                shape,
                low=0,
                high=num_classes // 2,
            ),
            op_args=(),
            op_kwargs={"num_classes": num_classes},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    def make_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high)

    shapes = (
        ((2, 2, 2, 3, 2), 10),
        ((2, 2, 2, 2, 2, 2, 3), 10),
    )
    for shape, num_classes in shapes:
        yield OpSampleInput(
            op_input=make_input(
                shape,
                low=0,
                high=num_classes // 2,
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
    def make_input(shape, *, num_classes):
        high = 10 if num_classes == -1 else num_classes
        return make_tensor(shape, dtype=dtype, device=device, low=0, high=high)

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
                    op_input=make_input(shape=(50, 47), num_classes=51),
                    op_args=(),
                    op_kwargs={'num_classes': 51},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_input(shape=(43, 46), num_classes=52),
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
                    op_input=make_input(shape=(50, 5), num_classes=100),
                    op_args=(),
                    op_kwargs={'num_classes': 100},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_input(shape=(40,), num_classes=51),
                    op_args=(),
                    op_kwargs={'num_classes': 51},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        # `num_classes=-1` is value-dependent and does not fit this compile/run
        # reference-comparison path for dynamic-rank opinfo cases.
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(),
                op_kwargs={'num_classes': 100},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_input(shape=(50, 21, 6, 25, 50), num_classes=100),
                    op_args=(),
                    op_kwargs={'num_classes': 100},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_input(shape=(1, 1), num_classes=51),
                    op_args=(),
                    op_kwargs={'num_classes': 51},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
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
        input_values = np.array(_float_small_values).reshape((1, 1, len(_float_small_values), 1))
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
        input_values = np.array(_large_values).reshape((1, len(_large_values), 1))
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

    sample_name = f'{op_info.name}_{dtype}'
    yield OpSampleInput(get_tensor_input((S, )), op_args=(1, 0, True, True), sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, ), sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1), sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2), sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1, True), sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2, True), sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1, True, True),
                        sample_name=sample_name+"_largest_True")
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2, True, True),
                        sample_name=sample_name+"_largest_True")


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

    for inputs, k, dims, largest, is_sort in all_cases:
        if '_kbk' in op_info.name and k==0:
            continue
        yield OpSampleInput(
            op_input=inputs,
            op_args=(k, dims, largest, is_sort),
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_largest_{largest}',
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

def basic_sample_inputs_mint_narrow(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.narrow op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S,), 0, 0, 1),
        ((S, S), 1, 1, 1),
        ((S, S, S), 0, 2, 2),
        ((S, 1, S), 1, 1, 0),
        ((S, 0, S), 0, ms.tensor(1), 0),
    )
    for shape, dims, starts, lengths in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(dims, starts, lengths),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_narrow(op_info: OpInfo, dtype=None, device=None):
    """
    Generate extra sample inputs for mint.narrow op.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3,), 0, 0, 0),
        ((3, 6), 1, 1, 0),
        ((2, 3, 4), 1, 2, 1),
        ((2, 2, 2, 2), 2, 1, 1),
        ((2, 2, 2, 3, 2), 1, 1, 1),
        ((2, 2, 3, 2, 2, 2, 2, 3), 3, 1, 1),
    )
    for shape, dims, starts, lengths in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(dims, starts, lengths),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_narrow_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        #input, dim, start, length
        ("1",  0, 0, 0), #input not tensor
        (make_tensor(shape=(2,), dtype=ms.float32), "1", 0, 0), # dim not int
        (make_tensor(shape=(2,), dtype=ms.float32), 1, "0", 0),  # start not int
        (make_tensor(shape=(2,), dtype=ms.float32), 1, 3, 0),  # start out shape
        (make_tensor(shape=(2,), dtype=ms.float32), 1, 0, "0"),  # length not int
        (make_tensor(shape=(2,), dtype=ms.float32), 1, 0, 4),  # length out shape
    ]
    for op_input, dims, starts, lengths in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(dims, starts, lengths),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.narrow error values'
        )

def dynamic_sample_inputs_mint_narrow(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.narrow.

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
                op_args=(1, 2, 2),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 2, 4, 7)),
                    op_args=(1, 2, 2),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 5, 6, 9, 3)),
                    op_args=(1, 2, 2),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, 2, 1),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 4, 1, 1)),
                    op_args=(1, 2, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5, 2, 1, 4, 1)),
                    op_args=(1, 2, 1),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, 1, 0),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4)),
                    op_args=(1, 1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 1)),
                    op_args=(1, 1, 0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
            ),
        )

def basic_sample_inputs_mint_diag(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.diag op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S,), (0,)),
        ((S, S), (1,)),
        ((S, S), (0,)),
        ((S, 1), (0,)),
        #((S, 0), (-2,)),
    )
    for shape, diagonal in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=diagonal,
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_diag(op_info, dtype, device=None, **kwargs):
    cases = [
        # discontinuous
        (ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32)[:, ::2], dtype=dtype), 1),
        (ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), dtype=dtype), 0),
        # broadcast
        (ms.Tensor(np.array([[1, 3, 2], [4, 1, 5]]), dtype=dtype), 1),
    ]
    for op_inputs, diagonal in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(diagonal,),
            op_kwargs={},
            sample_name=op_info.name,
        )
    #small/large values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )
        # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_mint_diag(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        (0, -3),#dim valueerror
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), 1.7),
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), True),
        (ms.Tensor(np.random.randn(9, 8), dtype=ms.float32), (1, 2)),
        (ms.Tensor(np.random.randn(9, 8, 2, 1, 1, 6), dtype=ms.float32), 1)#only support 1-2d
    ]
    for op_input, diagonal in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(diagonal,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops mint.diag error values'
        )

def dynamic_sample_inputs_mint_diag(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate dynamic inputs for mint.diag ops.
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
                op_input=ms.Tensor(shape=(None, 3), dtype=dtype),
                op_args=(1,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(1,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 3)),
                    op_args=(1,),
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
                op_args=(0,),
                op_kwargs={},
                grad_position=(),
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5,)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 1)),
                    op_args=(0,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def basic_reference_inputs_tensor_argsort(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = EXTRA_SMALL_DIM_SIZE
    M = MEDIUM_DIM_SIZE
    L = LARGE_DIM_SIZE

    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )

    cases = [
        ((), 0, True, False),
        ((S,), -1, True, True),
        ((M, S), 1, False, True),
        ((S, S, L), -2, False, False),
        ((5, 3, 4, 2), 3, True, False),
    ]
    for shape, dim, descending, stable in cases:
        yield OpSampleInput(
            make_func(shape),
            op_args=(),
            op_kwargs={'dim': dim, 'descending': descending, 'stable': stable},
            sample_name=op_info.name,
        )

def extra_reference_inputs_tensor_argsort(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (2, 3, 4),(),
                        {'dim': -2, 'descending': True, 'stable': False}, dtype=dtype, device=device)
    yield from make_extra_tensor(op_info, (0, 4),(),
        {'dim': 0, 'descending': True, 'stable': False}, dtype=dtype, device=device, empty_tensor=True)


def error_reference_inputs_tensor_argsort(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_func = functools.partial(make_tensor,device=device,dtype=ms.float32,)

    for error_dim in [-3, 2]:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func((2, 3)),
                op_args=(),
                op_kwargs={'dim': error_dim},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='dim error values'
        )

def basic_sample_inputs_tensor_diag(op_info: OpInfo, dtype=None, device=None):
    yield from basic_sample_inputs_mint_diag(op_info, dtype, device)
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((5, 0), (-2,)),
    )
    for shape, diagonal in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=diagonal,
            op_kwargs={},
            sample_name=f'{op_info.name}_{dtype}_empty_tensor',
        )

def extra_sample_inputs_tensor_diagflat(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from extra_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((5, 4, 7), 19),
        ((7,), -4),
        ((9, 5, 7, 4, 4), 10),
        ((1024, 8), 0),
    )

    for input_shape, offset in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={'offset': offset},
            sample_name=f"{op_info.name}_{dtype}_offset{offset}_{len(input_shape)}d_extra",
        )

def error_sample_inputs_tensor_diagflat(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        # input not tensor
        (0, -3),
        # offset is not int
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), 1.7),
        # case can pass
        # (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), True),
        (ms.Tensor(np.random.randn(9, 8), dtype=ms.float32), (1, 2)),
    ]
    for op_input, offset in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(),
                op_kwargs={'offset': offset},
                sample_name=f'{op_info.name}_error_param_values',
            ),
            op_error_type=(TypeError, AttributeError),
            op_error_info=f'error param value for {op_info.name}'
        )

def basic_sample_inputs_tensor_diagonal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    for sample_input in basic_reference_inputs_unary_op_common_func(op_info, dtype=dtype, device=device, **kwargs):
        dim_len = len(sample_input.op_input.shape)
        sample_input.op_kwargs.update({'offset': 0, 'axis1': (dim_len-2), 'axis2': (dim_len-1)})
        yield sample_input

def extra_sample_inputs_tensor_diagonal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((9, 7, 5), 0, 2, 0),
        ((10, 3, 7, 9, 9), -4, -5, -4),
        ((2, 3, 4, 2, 3, 4, 6, 8), 19, 1, 0),
        ((12, 687, 1, 13), -3, 3, 2),
        ((512, 17, 17), 2, 0, 1),
    )

    for input_shape, offset, axis1, axis2 in cases:
        _input = make_arg(input_shape)
        yield OpSampleInput(
            op_input=_input,
            op_args=(),
            op_kwargs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
            sample_name=f"{op_info.name}_{dtype}_offset{offset}_extra",
        )
    # large mix small values
    values = small_large_values(dtype)
    if values:
        input_np_array = np.random.choice(values, size=(2, 3, 4, 5))
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(),
                op_kwargs={'offset': -3, 'axis1': -4},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((4, 3, 5), t), dtype=dtype),
                op_args=(),
                op_kwargs={'axis2': 2},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def dynamic_sample_inputs_tensor_diagonal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 2)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5, 2)),
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
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
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
                    op_input=make_func(shape=(2, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_sample_inputs_tensor_diagonal(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device)
    cases = (
        # input is 1D, dim is less than 2
        ((1,), 0, 0, 1),
        # offset is not int
        ((1, 2), 1.0, 0, 1),
        # axis1 is not int
        ((1, 2), 0, 0.1, 1),
        # axis2 is not int
        ((1, 2), 0, 0, 1.0),
        # axis1 = axis2
        ((1, 2), 0, 1, 1),
        # axis1 is not in [-len(input.shape), len(input.shape))
        ((1, 2), 0, 2, 1),
        # axis2 is not in [-len(input.shape), len(input.shape))
        ((1, 2), 0, 0, -3),
    )
    for input_shape, offset, axis1, axis2 in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=input_shape),
                op_args=(),
                op_kwargs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
                sample_name=f"{op_info.name}_error_params",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'error values of input/offset/axis1/axis2 is not supported for {op_info.name}.',
        )

def basic_reference_inputs_tensor_diagonal_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((5, 1), (1,), 0, 0, 1),
        ((10, 5), (4,), 1, 0, 1),
        ((5, 10, 5), (5, 5), 0, 1, 2),
        ((10, 1, 5), (1, 5), -3, 0, 2),
        ((10, 1, 5), (5, 0), 2, 0, 1),
    )

    for input_shape, other_shape, offset, dim1, dim2 in cases:
        _input = make_func(input_shape)
        _other = make_func(other_shape)
        _splice = "empty_" if 0 in _input.shape+_other.shape else ""
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'offset': offset, 'dim1': dim1, 'dim2': dim2},
            sample_name=f"{op_info.name}_{dtype}_{_splice}{len(_input.shape)}d_x_{len(_other.shape)}d_basic",
        )

def extra_reference_inputs_tensor_diagonal_scatter(op_info:OpInfo, dtype=None, device=None, **kwargs):
    def _cross_entropy_sample_loss(other_dtype):
        # type promotion precision issue: large loss due to inconsistent result dtypes
        if other_dtype == ms.float16:
            return 3.5
        if other_dtype == ms.float32:
            return 0.32
        if other_dtype == ms.float64:
            return 0.24
        if other_dtype in (ms.int8, ms.int16, ms.int32, ms.int64):
            return 9
        return None

    # tensors with many kinds of shapes
    cases = (
        ((1, 0, 3), (1, 0), 1, 2, 1),  # empty tensors
        ((3, 2), (2,), 0, 0, 1),
        ((2, 1, 3), (3, 1), -1, 0, 1),
        ((1029, 917), (917,), 0, 0, 1),  # 2D tensors with large size
        ((7, 3, 5, 4), (7, 3, 4), 1, -1, 2),  # 4D tensors
        ((4, 4, 7, 9, 9, 7), (4, 4, 7, 9, 2), 5, 4, -1),  # 6D tensors
        ((5, 9, 9, 5, 4, 5, 5), (9, 9, 5, 5, 5, 1), 4, -3, 0),  # 7D tensors
    )

    for input_shape, other_shape, offset, dim1, dim2 in cases:
        _input = make_tensor(
            input_shape,
            dtype,
            device=device,
        )
        _other = make_tensor(
            other_shape,
            dtype,
            device=device,
        )
        _splice = "empty_" if 0 in _input.shape else ""
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'offset': offset, 'dim1': dim1, 'dim2': dim2},
            sample_name=(f"{op_info.name}_{dtype}_{_splice}{len(_input.shape)}d_x_{len(_other.shape)}d_extra")
        )
    # large mix small values
    values = small_large_values(dtype)
    if values:
        input_np_array = np.random.choice(values, size=(1, 2, 3))
        np_array1 = np.random.choice(values, size=(1, 1))
        for discontiguous in [True, False]:
            _src = make_tensor_with_np_array(np_array1, dtype=dtype, discontiguous=discontiguous)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(input_np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(_src,),
                op_kwargs={'offset': -1, 'dim1': 1, 'dim2': 2},
                sample_name=f"{op_info.name}_{dtype}_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            _src = make_tensor_with_np_array(np.full((1, 3), t), dtype=dtype)
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((4, 3, 1), t), dtype=dtype),
                op_args=(_src,),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )
    # type promotion precision issue
    src_dtypes = (ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.float16, ms.float32, ms.float64)
    for src_dtype in src_dtypes:
        sample_loss = _cross_entropy_sample_loss(src_dtype)
        if dtype == src_dtype:
            continue
        yield OpSampleInput(
            op_input=make_tensor(shape=(5, 5), dtype=dtype),
            op_args=(make_tensor(shape=(5,), dtype=src_dtype),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_2d_x_{src_dtype}_1d_promotion",
            loss=sample_loss,
        )

def dynamic_sample_inputs_tensor_diagonal_scatter(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if op_info.support_dynamic_shape and not kwargs.get("only_dynamic_rank", False):
        # dyncamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=dtype),),
                op_kwargs={'offset': -3, 'dim1': -3, 'dim2': -2},
                grad_position=(),
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 9, 3)),
                    op_args=(make_func(shape=(4, 1)),),
                    op_kwargs={'offset': -2, 'dim1': -1, 'dim2': -2},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 8)),
                    op_args=(make_func(shape=(8, 7)),),
                    op_kwargs={'offset': 2, 'dim1': 1, 'dim2': 0},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

def error_sample_inputs_tensor_diagonal_scatter(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device)
    cases = (
        # input is 1D, dim is less than 2
        ((1,), make_func(shape=(1,)), 0, 0, 1),
        # src is not tensor
        ((1, 2), 1, 0, 0, 1),
        # the dimension of src is larger than 6D
        ((5, 9, 9, 5, 4, 5, 5, 1), make_func(shape=(9, 9, 5, 4, 5, 1, 1)), 4, -3, 0),
        # offset is not int
        ((1, 2), make_func(shape=(1,)), 1.0, 0, 1),
        # dim1 is not int
        ((1, 2), make_func(shape=(1,)), 0, 0.1, 1),
        # dim2 is not int
        ((1, 2), make_func(shape=(1,)), 0, 0, 1.0),
        # dim1 = dim2
        ((1, 2), make_func(shape=(1,)), 0, 1, 1),
        # dim1 is not in [-len(input.shape), len(input.shape))
        ((1, 2), make_func(shape=(1,)), 0, 2, 1),
        # dim2 is not in [-len(input.shape), len(input.shape))
        ((1, 2), make_func(shape=(1,)), 0, 0, -3),
        # the shape of src is not equal to input or cannot be broadcast
        ((1, 2), make_func(shape=(3,)), 0, 0, 1),
    )
    for input_shape, src, offset, dim1, dim2 in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=input_shape),
                op_args=(src,),
                op_kwargs={'offset': offset, 'dim1': dim1, 'dim2': dim2},
                sample_name=f"{op_info.name}_error_params",
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info=f'error values of input/offset/dim1/dim2 is not supported for {op_info.name}.',
        )
