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

# ====================================linalg basic sample inputs functions===================================
def basic_reference_inputs_tensor_addr_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield typical shape cases for tensor.addr ops.

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

def basic_reference_inputs_tensor_addcdiv_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield typical shape cases for tensor.addcdiv ops.

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

def basic_reference_inputs_tensor_addmv_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield typical shape cases for tensor.addmv ops.

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

# ====================================linalg extra sample inputs functions===================================
def extra_sample_inputs_tensor_addr(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for tensor.addr
    Covers: various padding modes, dilation, groups, small/large/extremal value
            tensors, discontiguous tensors, type promotion.
    Skips: scalar inputs (conv2d requires tensor inputs).
    '''
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

def extra_sample_inputs_tensor_addcdiv(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for tensor.addcdiv
    Covers: various padding modes, dilation, groups, small/large/extremal value
            tensors, discontiguous tensors, type promotion.
    Skips: scalar inputs (conv2d requires tensor inputs).
    '''
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
    '''
    Generate extra sample inputs for tensor.addmv
    Covers: various padding modes, dilation, groups, small/large/extremal value
            tensors, discontiguous tensors, type promotion.
    Skips: scalar inputs (conv2d requires tensor inputs).
    '''
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