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
- sample_inputs for shape-types ops
"""
import functools

import numpy as np
import random
import warnings
import itertools
import mindspore as ms
from mindspore import mutable
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.sample_inputs.common import make_extra_tensor

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_extra_values,
    small_large_values
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
            op_args=(mutable(input_data=(2, 3, 1, 1, 1), dynamic_len=True),),
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
                op_args=(mutable(input_data=(8, 1, 1, -1, 1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )
    # dynamic rank
    yield OpDynamicInput(
        op_compile_input=OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=dtype),
            op_args=(mutable(input_data=(2, 6, 1, 1), dynamic_len=True),),
            op_kwargs={},
            sample_name=f"{op_info.name}_dynamic_rank_compile_input",
        ),
        op_running_inputs=(
            OpSampleInput(
                op_input=make_func(shape=(7, 3)),
                op_args=(mutable(input_data=(21, 1, -1, 1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(9, 9, 3, 5, 8)),
                op_args=(mutable(input_data=(9, 3, 4, -1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(4,)),
                op_args=(mutable(input_data=(2, 2, 1, -1), dynamic_len=True),),
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

def basic_sample_inputs_pad(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    N = 1
    C = 3
    D = 5
    H = 7
    W = 10
    shape = {1: (W, ), 2: (H, W), 3: (D, H, W)}[dim]

    yield OpSampleInput(
        op_input=make_arg((C, ) + shape),
        op_args=(3, ) if mode != 'ConstantPad' else (3, 2),
        op_kwargs={},
        sample_name=op_info.name,
    )

    yield OpSampleInput(
        op_input=make_arg((N, C,) + shape),
        op_args=((0, 3) * dim, ) if mode != 'ConstantPad' else ((0, 3) * dim, -8 if dtype != ms.uint8 else 0),
        op_kwargs={},
        sample_name=op_info.name,
    )

    yield OpSampleInput(
        op_input=make_arg((N, C,) + shape),
        op_args=((1, 3) * dim,) if mode != 'ConstantPad' else ((1, 3) * dim, 3.9),
        op_kwargs={},
        sample_name=op_info.name,
    )

    yield OpSampleInput(
        op_input=make_arg((0, C,) + shape),
        op_args=((3, 0) * dim,) if mode != 'ConstantPad' else ((3, 0) * dim, -12.7 if dtype != ms.uint8 else 0),
        op_kwargs={},
        sample_name=op_info.name,
    )

def extra_sample_inputs_pad(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None, **kwargs):

    op_args=(8, ) if mode != 'ConstantPad' else  (8, 9)

    yield from make_extra_tensor(op_info, (1, 1,) + (16, ) * dim, op_args, {},
                                 dtype=dtype, device=device)

    if mode == 'ZeroPad':
        yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9), op_args, {},
                                     dtype=dtype, device=device, support_multi_dim=True)

def error_sample_inputs_pad(op_info: OpInfo, dtype=None, device=None, dim=None, mode=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    N = 1
    C = 3
    D = 5
    H = 7
    W = 10
    shape = {1: (W, ), 2: (H, W), 3: (D, H, W)}[dim]

    if mode not in  ['ZeroPad', 'ConstantPad']:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape),
                op_args=(3,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'input.dim must be {dim+1} or {dim+2}',
        )

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((1, N, C,) + shape),
                op_args=(3,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'input.dim must be {dim+1} or {dim+2}',
        )

    else:

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape[1:]),
                op_args=(3,) if mode != 'ConstantPad' else (3, 3),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info=f'input.dim must be >={dim}',
        )

    if mode == 'ReflectionPad':
        for _index in range(len(shape)):
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_arg((C, ) + shape),
                    op_args=(tuple(itertools.chain.from_iterable([(value, value) if index == _index else
                                                (value-1, value-1)  for index,value in enumerate(shape[::-1])])) ,),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=RuntimeError,
                op_error_info=f'input.dim must be {dim+1} or {dim+2}',
            )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, C,) + shape),
            op_args=((1, 3) * (dim-1),) if mode != 'ConstantPad' else ((1, 3) * (dim-1), 3),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'pad.length must be {dim*2}',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, C,) + shape),
            op_args=((1, 3) * (dim + 1 ),) if mode != 'ConstantPad' else ((1, 3) * (dim + 1), 3),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=f'pad.length must be {dim*2}',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, C,) + shape),
            op_args=(-(min(shape) // 2 + 1),) if mode != 'ConstantPad' else (-(min(shape) // 2 + 1), 3),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(RuntimeError, ValueError),
        op_error_info='negative pad makes output size invalid',
    )

    invalid_tuple_pad = tuple(
        itertools.chain.from_iterable((-(value + 1), 0) for value in shape[::-1])
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((N, C,) + shape),
            op_args=(invalid_tuple_pad,) if mode != 'ConstantPad' else (invalid_tuple_pad, 3),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=(RuntimeError, ValueError),
        op_error_info='negative pad makes output size invalid',
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

def dynamic_sample_inputs_mint_pad(op_info: OpInfo, dtype=None, device=None, mode=None, **kwargs):
    """
    Generate dynamic-shape/rank inputs for mint.nn.functional.pad.
    """
    assert mode in ('constant', 'reflect', 'replicate', 'circular')
    if mode == 'circular' and device != 'cpu':
        return
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        compile_shape = (None, None, None, None)
        if mode == 'constant':
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=compile_shape, dtype=dtype),
                    op_args=((1, 1, 1, 1), mode, 1.0),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4, 5)),
                        op_args=((1, 1, 1, 1), mode, 1.0),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(3, 4, 5, 6)),
                        op_args=((1, 1, 1, 1), mode, 1.0),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                ),
            )
        elif mode in ('reflect', 'replicate'):
            pad = (1, 1, 1, 1)
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=compile_shape, dtype=dtype),
                    op_args=(pad, mode),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4, 5)),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(3, 4, 5, 6)),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                ),
            )
        else:
            pad = (1, 1, 1, 1)
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=compile_shape, dtype=dtype),
                    op_args=(pad, mode),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4, 5)),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(3, 4, 5, 6)),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_shape_running_input",
                    ),
                ),
            )

    if not kwargs.get("only_dynamic_shape", False):
        if mode == 'constant':
            for pad, run_shapes in (
                    ((1, 1), ((2, 3), (2, 3, 4))),
                    ((1, 1, 1, 1), ((2, 3, 4), (3, 4, 5, 6))),
                    ((1, 1, 1, 1, 1, 1), ((2, 3, 4, 5), (3, 4, 5, 6, 7))),
            ):
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=None, dtype=dtype),
                        op_args=(pad, mode, 1.0),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_compile_input",
                    ),
                    op_running_inputs=(
                        OpSampleInput(
                            op_input=make_func(shape=run_shapes[0]),
                            op_args=(pad, mode, 1.0),
                            op_kwargs={},
                            sample_name=f"{op_info.name}_dynamic_rank_running_input",
                        ),
                        OpSampleInput(
                            op_input=make_func(shape=run_shapes[1]),
                            op_args=(pad, mode, 1.0),
                            op_kwargs={},
                            sample_name=f"{op_info.name}_dynamic_rank_running_input",
                        ),
                    ),
                )
        elif mode in ('reflect', 'replicate'):
            for pad, run_shapes in (
                    ((1, 1), ((2, 3), (2, 3, 4))),
                    ((1, 1, 1, 1), ((2, 3, 4), (3, 4, 5, 6))),
            ):
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=None, dtype=dtype),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_compile_input",
                    ),
                    op_running_inputs=(
                        OpSampleInput(
                            op_input=make_func(shape=run_shapes[0]),
                            op_args=(pad, mode),
                            op_kwargs={},
                            sample_name=f"{op_info.name}_dynamic_rank_running_input",
                        ),
                        OpSampleInput(
                            op_input=make_func(shape=run_shapes[1]),
                            op_args=(pad, mode),
                            op_kwargs={},
                            sample_name=f"{op_info.name}_dynamic_rank_running_input",
                        ),
                    ),
                )
        else:
            pad = (1, 1, 1, 1)
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(pad, mode),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_compile_input",
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4)),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input",
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(3, 4, 5, 6)),
                        op_args=(pad, mode),
                        op_kwargs={},
                        sample_name=f"{op_info.name}_dynamic_rank_running_input",
                    ),
                ),
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
    if mode == 'circular' and device == 'ascend':
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((1, 3, 3)),
                op_args=((1, 2),),
                op_kwargs={'mode': mode},
                sample_name=op_info.name,
            ),
            op_error_type=RuntimeError,
            op_error_info='circular mode is not supported on Ascend',
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
            sample_name=f"{op_info.name}_{dtype}",
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
        input_values = np.array(_float_small_values).reshape((1, 1, len(_float_small_values), 1))
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
        input_values = np.array(_large_values).reshape((1, 1, len(_large_values), 1))
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
        ((3, 4, 5), 2),
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
        (ms.Tensor(np.array([]).reshape(3, 0), dtype=dtype), 1),
        # discontinuous
        (ms.Tensor(np.arange(32).reshape(8, 4).astype(np.float32)[:, ::2], dtype=dtype), 1,),
        (ms.Tensor(np.zeros((8, 4)), dtype=dtype), 0),
        (ms.Tensor(np.full((6, 3), np.finfo(np.float32).max), dtype=dtype), 1),
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

def basic_sample_inputs_mint_squeeze(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from basic_sample_inputs_mint_unsqueeze(op_info, dtype, device, **kwargs)
    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.array([2, 3, 27]), dtype=dtype, device=device),
        op_args=((-1),), #dim tuple
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_mint_squeeze_dim_tuple",
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

def basic_sample_inputs_mint_unbind(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.unbind op.
    """
    S = SMALL_DIM_SIZE
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S,), 0),
        ((S, S), 1),
        ((S, S, S), -1),
        ((S, 1, S), 1),
        ((S, 0, S), 0),
    )
    for shape, dims in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(dims,),
            op_kwargs={},
            sample_name=op_info.name,
        )
def extra_sample_inputs_mint_unbind(op_info: OpInfo, dtype=None, device=None):
    """
    Generate extra sample inputs for mint.unbind op.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((3,), 0),
        ((3, 6), 1),
        ((2, 3, 4), 1),
        ((2, 2, 2, 2), 2),
        ((2, 2, 2, 3, 2), -1),
        ((2, 2, 2, 2, 2, 2, 2, 3), 3),
    )
    for shape, dims in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(dims,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_unbind_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        # input, dim
        ("1", 0,),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), "None",),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), (0, 1),),
    ]
    for op_inputs, dims in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(dims,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f"{op_info.name}_{dtype}_{device}_error_mint_special_log_softmax"
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
    values = small_large_values(dtype)
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

def dynamic_inputs_mint_tile_op_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate dynamic-shape/rank inputs for a mint.tile.

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
                op_args=((0, 1),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=((0, 1),),
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
                op_args=((0, 1),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=((0, 1),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
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

def basic_sample_inputs_constantpad1d(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    input_shapes=(
        (3, ), (1, 3), (1, 1, 3), (1, 1, 1, 1, 2, 2, 2, 2))
    paddings=(0, 2, (-1, 2), (1, 2))
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
    values = small_large_values(dtype)
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

def basic_sample_inputs_mint_flip(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor(np.random.randn(25,), dtype=dtype), ([-1],),),#dim list
        (ms.Tensor(np.random.randn(4, 4), dtype=dtype), ((1, -2),),),#dim tuple
        (ms.Tensor(np.random.randn(9, 9, 9, 9, 9, 9), dtype=dtype), ([5, 4, 2, 3],),),
    ]
    for op_inputs, dims in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=dims,
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_flip(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        # discontinuous
        (ms.Tensor(np.random.randn(12,), dtype=dtype)[::2], ([0,],),),
        (ms.Tensor(np.random.randn(4, 6), dtype=dtype)[:, ::2], ([1,],),),
        #shape 0
        (ms.Tensor(np.array([]), dtype=dtype).reshape(3, 0), ([1,],),),
        #broadcast, size 1 dim
        (ms.Tensor(np.random.randn(1, 1, 1), dtype=dtype), ([0, 1, 2],),),
    ]
    for op_inputs, dims in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=dims,
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
                op_args=((0, 1),),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_args=([-2, -1],),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_mint_flip(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (0, ([-1],),),#input int
        (None, ([-1],),),#input none
        (ms.Tensor(np.random.randn(4, 4), dtype=dtype), (True,),),#dim bool
        (ms.Tensor(np.random.randn(3, 3), dtype=dtype), ([1.5, 2.7],),),# dim not int
        (ms.Tensor(np.random.randn(3, 3), dtype=dtype), (1,),),#dim not list/tuple
        (ms.Tensor(np.random.randn(3, 4), dtype=dtype), ((5,),),),# dim out of range
    ]

    for op_inputs, dims in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=dims,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.flip error values',
        )

def basic_sample_inputs_mint_histc(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor(np.random.randn(3,), dtype=dtype), 48, -812.614, 50),
        (ms.Tensor(np.random.randn(4, 4), dtype=dtype), 1, -639.63, 108.023),
        (ms.Tensor(np.random.randn(3,), dtype=dtype), 100, 0, 0),#default values
        (ms.Tensor(np.random.randn(3, 3, 3, 8, 4, 7), dtype=dtype),  50, 278, 515)
    ]
    for op_inputs, bins, mins, maxs in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(bins, mins, maxs),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_mint_histc(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (0, 48, -812.614, 50),#input int
        (None, 48, -812.614, 50),# input int
        (ms.Tensor(np.random.randn(4, 4), dtype=dtype), 100.42112, -639.63, 108.023),#bins is float
        (ms.Tensor(np.random.randn(4, 4), dtype=dtype), "100", -639.63, 108.023),# bins is str
        (ms.Tensor(np.random.randn(3,), dtype=dtype), 100, "0", 0),#min is str
        (ms.Tensor(np.random.randn(3, ), dtype=dtype), 100, 0, "0"),# max is str
        (ms.Tensor(np.random.randn(4, 4), dtype=dtype), 100, 700, 108),# min>max
    ]

    for op_inputs, bins, mins, maxs in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(bins, mins, maxs),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.histc error values',
        )

def extra_sample_inputs_mint_histc(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        #inf/nan
        (ms.Tensor(np.full((5, 8, 9, 8), np.nan), dtype=dtype), 15, -233.457, -50),
        (ms.Tensor(np.full((5,), np.inf), dtype=dtype), 50, -302.337, 610),
        (ms.Tensor(np.full((3, 4), -np.inf), dtype=dtype), 50, -302.337, 610),
        #shape 0
        (ms.Tensor(np.random.randn(0, 5), dtype=dtype), 20, -50.1, 50),
        #broadcast1x1
        (ms.Tensor(np.random.randn(1, 1), dtype=dtype), 10, 0, 1.0),
        #discontinuous
        (ms.Tensor(np.random.randn(12, ), dtype=dtype)[::2], 100, -50, 50),
        (ms.Tensor([5, 5, 5, 5], dtype=dtype), 5, 0.0, 10),
    ]
    for op_inputs, bins, mins, maxs in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(bins, mins, maxs),
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
                op_args=(100, -50, 50),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

def basic_sample_inputs_mint_meshgrid(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        ([ms.Tensor(np.random.randint(-20, 20, 5), dtype=dtype),
          ms.Tensor(np.random.randint(-20, 20, 1), dtype=dtype)], "ij"),
        ((ms.Tensor(np.random.randint(-20, 20, 5), dtype=dtype),
          ms.Tensor(np.random.randint(-20, 20, 1), dtype=dtype),
          ms.Tensor(np.random.randint(-255, 255, 6), dtype=dtype)), "xy"),
        ([ms.Tensor(np.random.randn(7, ), dtype=dtype) for _ in range(6)], "xy"),
        ([ms.Tensor([1, 2, 3], dtype=dtype)], None),
        ([ms.Tensor(5, dtype=dtype)], "ij"),
        ([ms.Tensor([], dtype=dtype) for _ in range(3)], "xy"),
    ]
    for op_inputs, indexing in cases:
        if "_kbk" in op_info.name:
            if len(op_inputs) <=1 or dtype in [ms.complex64, ms.complex128, ms.bfloat16]:
                continue
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(),
            op_kwargs={"indexing": indexing},
            sample_name=op_info.name,
        )

def error_inputs_mint_meshgrid(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (0, "ij"),#input int
        (None, "xy"),#input none
        ([], "ij"),  # input []
        ([ms.Tensor(np.random.randn(7,), dtype=dtype) for _ in range(6)], 123),#indexing int
        ([ms.Tensor(np.random.randn(7, ), dtype=dtype) for _ in range(6)], ""),# indexing ""
        ([ms.Tensor(np.random.randn(7, 2, 3), dtype=dtype) for _ in range(3)], "xy"),#input not 1d
    ]

    for op_inputs, indexing in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(),
                op_kwargs={"indexing": indexing},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.meshgrid error values',
        )

def extra_sample_inputs_mint_meshgrid(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        #inf/nan
        ([ms.Tensor(np.full((7,), np.nan), dtype=dtype) for _ in range(3)], "xy"),
        ([ms.Tensor(np.full((7,), np.inf), dtype=dtype) for _ in range(3)], "xy"),
        # discontinuous
        ([ms.Tensor(np.random.randn(12,), dtype=dtype)[::2],
          ms.Tensor(np.random.randn(12,), dtype=dtype)[::2]], "ij"),
       #shape 0
        ([ms.Tensor(np.array([]).reshape(0), dtype=dtype), ms.Tensor([1, 2], dtype=dtype)], "ij"),
        #broadcast
        ([ms.Tensor(np.random.randn(5), dtype=dtype),
          ms.Tensor(np.random.randn(1), dtype=dtype), ms.Tensor(np.random.randn(6), dtype=dtype)], "ij"),
    ]
    for op_inputs, indexing in cases:
        if "_kbk" in op_info.name:
            if len(op_inputs) <= 1 or dtype in [ms.complex64, ms.complex128, ms.bfloat16]:
                continue
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(),
            op_kwargs={"indexing": indexing},
            sample_name=op_info.name,
        )

    # small/large values
    values = small_large_values(dtype)
    if values:
        for discontiguous in [True, False]:
            if "_kbk" in op_info.name:
                if len(op_inputs) <= 1 or dtype in [ms.complex64, ms.complex128, ms.bfloat16]:
                    continue
            yield OpSampleInput(
                op_input=(ms.Tensor(np.random.randint(-20, 20, 5), dtype=dtype),
                          ms.Tensor(np.random.randint(-20, 20, 1), dtype=dtype),
                          ms.Tensor(np.random.randint(-255, 255, 6), dtype=dtype)),
                op_args=(),
                op_kwargs={"indexing": "xy"},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

def basic_sample_inputs_mint_roll(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor(np.random.randn(5,), dtype=dtype), ((-1, -1, -1, 0), (-1, -1, -1, -1))),
        (ms.Tensor(np.random.randn(), dtype=dtype), (1, None,),),
        (ms.Tensor(np.random.randn(3, 8, 5, 8, 4, 4, 8, 8), dtype=dtype), ([-1, 1, -2, -7], [3, -8, -3, 4],))
    ]
    #input, shifts, dims
    for op_inputs, args in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_mint_roll(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor(np.random.randn(5,), dtype=dtype), ("1", (-1, -1, -1, -1),)),#shifts str
        (ms.Tensor(np.random.randn(), dtype=dtype), (1, "None",)),#dims str
        (ms.Tensor(np.random.randn(3, 4), dtype=dtype), ((2,), [5],)),# dims out of range
        (ms.Tensor(np.random.randn(5, ), dtype=dtype), (-12.978, (-1, -1, -1, -1))),# shifts not int
        (ms.Tensor(np.random.randn(5, ), dtype=dtype), ([], (-1, -1, -1, -1))),# shifts[]
        (ms.Tensor(np.random.randn(3, 8, 5, 8, 4, 4, 8, 8), dtype=dtype), ([-1, 1, -2, -7], -1213.11,)),#dim not int
        (0, ([-1, 1, -2, -7], 0,)),# input scalar
        (None, ((2,), None,))# input none
    ]
    #input, shifts, dims
    for op_inputs, args in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=args,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.roll error values',
        )

def extra_sample_inputs_mint_roll(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        #inf/nan
        (ms.Tensor(np.full((9, ), np.inf),dtype=dtype), (0, (-1,),)),
        (ms.Tensor(np.full((9, 3, 4, 4, 8, 8), np.nan), dtype=dtype), (5, 5,),),
        #shape 0
        (ms.Tensor(np.random.randn(3, 0, 5), dtype=dtype), ([1, 2], [0, 2],)),
        #broadcast
        (ms.Tensor(np.random.randn(1, 4, 1), dtype=dtype), ([1, 2], [0, 2],),),
        # discontinuous
        (ms.Tensor(np.random.randn(6, 8)[::2, ::2], dtype=dtype), ([1, 2], [0, 1],)),
        (ms.Tensor(np.random.randn(6, 8), dtype=dtype), ([1, 2], [0, 1],))
    ]
    # input, shifts, dims
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
                op_input=make_tensor_with_np_array(np_array, dtype=dtype),
                op_args=([-2, 1], [-2, 1],),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
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

def extra_sample_inputs_tensor_repeat(op_info, dtype, device=None, **kwargs):
    # large mix small values
    values = small_large_values(dtype)
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
    if not kwargs.get("only_dynamic_shape", False):
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
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(mutable(input_data=(3, 1), dynamic_len=True),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 3)),
                    op_args=(mutable(input_data=(1, 1), dynamic_len=True),),
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
    values = small_large_values(dtype)
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

def basic_reference_inputs_tensor_chunk_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield typical shape cases for tensor.chunk ops.

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

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    #input, chunks, dim
    base_shapes = (
        ((S,), 1, 0),
        ((M, S), 1, 1),
        ((S, S, L), 2, 1),
        ((2, 0, 3, 2), 2, 1),
        ((2, 3, 4, 1, 2), 3, 2),
        ((2, 1, 2, 2, 1, 2), 3, 1),
        ((2, 1, 2, 2, 1, 2, 1), 4, 2),
        ((2, 1, 2, 2, 1, 2, 1, 2), 3, 5),
    )
    for input_shape, chunks, dims in base_shapes:
        _input = make_arg(input_shape)
        slen = len(input_shape)
        yield OpSampleInput(
            op_input= _input,
            op_args=(chunks, dims,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_tensor_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for tensor.chunk
    Covers: various padding modes, dilation, groups, small/large/extremal value
            tensors, discontiguous tensors, type promotion.
    Skips: scalar inputs (conv2d requires tensor inputs).
    '''
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=None, high=None)
    # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_shape = (1, 1, len(_float_small_values))
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(2,),
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
            op_args=(2,),
            op_kwargs={},
            sample_name=f"{op_info.name}_large",
        )

    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((1, 1, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(2,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_inf",
                )

    # ---- discontiguous tensors ----
    for discontiguous in (True,):
        input_shape = (1, 3, 3)
        yield OpSampleInput(
            op_input=make_arg(input_shape, discontiguous=discontiguous),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        )

def error_inputs_other_op_tensor_chunk_func(op_info, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device, low=None, high=None)
    shapes = (
        (1.0, 1, 0),#input is not tensor
        (make_arg((M, S)), True, 1),#CHUNKS is not int
        (make_arg((S, S, L)), 2, make_arg((S, S, L))),#dims is tensor
        (make_arg((2, 3, 4, 1, 2)), 3, 10),#dims out range
        (make_arg((2, 1, 2, 2, 1, 2)), -3, 1),
    )
    for input_shape, chunks, dims in shapes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=input_shape,
                op_args=(chunks, dims,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError, AttributeError),
            op_error_info='other ops tensor.chunk error values')

def dynamic_sample_inputs_tensor_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic sample inputs for Tensor.chunk.
    Consider both dynamic_shape and dynamic_rank.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape
        chunk, dim = 1, 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(chunk, dim,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_2D",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4)),
                    op_args=(chunk,dim,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3D",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4 ,5, 7)),
                    op_args=(chunk,dim,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_3D",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        chunk, dim = 1, 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(chunk, dim,),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(chunk, dim,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input_1'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4)),
                    op_args=(chunk, dim,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input_2'
                ),
            )
        )
