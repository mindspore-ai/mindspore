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
- sample_inputs for reduction-types ops
"""
import functools

import numpy as np
import random
import itertools
import mindspore as ms
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo, ReductionOpInfo, \
    extra_reference_inputs_unary_op_common_func
from tests.st.ops.share._op_info.op_info import (
    basic_reference_inputs_binary_op_common_func,
    basic_reference_inputs_reduction_op_common_func,
    basic_reference_inputs_unary_op_common_func,
    extra_reference_inputs_binary_op_common_func,
    extra_reference_inputs_reduction_op_common_func,
    dynamic_inputs_reduction_op_common_func,
    convert_params_name
)
from tests.st.ops.share._op_info.sample_inputs.common import make_extra_tensor

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array,
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtype_to_np_dtype,
    op_dtype_need_convert,
    dtypes_extra_values,
    small_large_values
)

#convert tensor for op
def convert_tensor_for_op(op_sample_input, op_info):
    if isinstance(op_info, BinaryOpInfo):
        op_sample_input.op_args = (op_sample_input.op_input,)
    elif isinstance(op_info, ReductionOpInfo):
        op_sample_input.op_kwargs = {"dim": None, "keepdim": False} | op_sample_input.op_kwargs
        op_sample_input.sample_name = f"{op_sample_input.sample_name}_default"
    return  op_sample_input

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
        op_error_info=f'dim error type is not supported for {op_info.name}',
    )

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

def basic_sample_inputs_reduction_count_nonzero(op_info, dtype, device=None, **kwargs):
    """count_nonzero does not have keepdim parameter"""
    for sample_input in basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs):
        sample_input.op_kwargs.pop('keepdim', None)
        yield sample_input

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

def basic_reference_inputs_mint_special_log_softmax(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    yield OpSampleInput(
        op_input=ms.Tensor([[1, 2, 3], [4, 5, 6]]),
        op_args=(),
        op_kwargs={'dim': 1, 'dtype': ms.float32},
        sample_name=f"{op_info.name}_{dtype}_{device}_mint_special_log_softmax"
    )
    yield OpSampleInput(
        op_input=ms.Tensor([[1, 2, 3], [4, 5, 6]]),
        op_args=(),
        op_kwargs={'dim': 0, 'dtype': ms.float32},
        sample_name=f"{op_info.name}_{dtype}_{device}_mint_special_log_softmax"
    )

def error_inputs_unary_op_mint_special_log_softmax_func(op_info:OpInfo, dtype=None, device=None, **kwargs):
    test_cases = [
        # input, dim, dtype
        ("1", 0, ms.float32),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), "None", ms.float32),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), 0, "False"),
        (ms.Tensor([[1, 2, 3], [4, 5, 6]]), 6, ms.float32)
    ]
    for op_inputs, dims, dtypes in test_cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(),
                op_kwargs={'dim': dims, 'dtype': dtypes},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info=f"{op_info.name}_{dtype}_{device}_error_mint_special_log_softmax"
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

def basic_sample_inputs_normalize(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((4, 8), 1, 1, 0.01,),
        ((4, 6, 5), 2, 0, 1e-6),
        ((3, 8, 4, 8), 4, -1, 1e-12),
        ((9, 14, 6, 2, 3), 3, -2, 2.0),
    )

    for input_shape, p, dim, eps in cases:
        slen = len(input_shape)
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_kwargs={'p':p, 'dim':dim, 'eps':eps},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_normalize(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    #empty tensors
    _input = make_func(shape=(1, 0, 4))
    yield OpSampleInput(
        _input,
        op_args=(),
        sample_name=f"{op_info.name}_{dtype}_empty_tensor",
    )

    # large/small/discontiguous values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(),
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # nan/inf/-inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full([1024, 1024], t), dtype=dtype, device=device),
                op_args=(),
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_other_op_normalize(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # p not float
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_kwargs={'p':'p',},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info='p should be float ',
    )

    # dim not int
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_kwargs={'dim':1.0,},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info='dim should be int',
    )

    # eps not float
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_kwargs={'eps':'1',},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info='eps should be float',
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
        atol = random.choice([0, 1, 1e-5, 1e-16, 31415626, 0.5])
        rtol = random.choice([0, 1, 1e-5, 1e-16, 31415626, 0.5])
        sample_input.op_kwargs=({'atol':atol, 'rtol':rtol})
        if 'nan' in sample_input.sample_name:
            sample_input.op_kwargs.update({"equal_nan":False})
            yield sample_input
            sample_input.op_kwargs.update({"equal_nan":True})
            yield sample_input
        else:
            yield sample_input

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
                    # set_inputs fixes non-Tensor scalars to the compile-time value.
                    op_args=(1,),
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
                    # Keep dim aligned with compile input while varying only rank/shape.
                    op_args=(1,),
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
                # A separate dynamic-rank group is needed for inputs whose valid dim is 0.
                op_args=(0,),
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
            op_error_type=(TypeError, ValueError, AttributeError),
            op_error_info='other ops mint.cumsum error values'
        )

def extra_sample_inputs_mint_cumprod(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate extra sample inputs for mint.cumprod.
    """
    yield from extra_sample_inputs_mint_cumsum(op_info, dtype=dtype, device=device, **kwargs)
    if dtype == ms.int8:
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.array([2, 3, 27], dtype=np.int8), dtype=dtype, device=device),
            op_args=(0,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_promote_to_int64",
        )

def error_inputs_other_op_mint_cumprod_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate error inputs for mint.cumprod.
    """
    yield from error_inputs_other_op_mint_cumsum_func(op_info, dtype=dtype, device=device, **kwargs)
    if device == 'ascend' and MSContext.get_instance().get_ascend_soc_version() == 'ascend910b':
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_tensor(shape=(2, 3, 4), dtype=ms.bfloat16, device=device),
                op_args=(1,),
                op_kwargs={},
                sample_name=f"{op_info.name}_bfloat16_unsupported",
            ),
            op_error_type=RuntimeError,
            op_error_info='DT_BF16|unsupported|Launch kernel failed',
        )

def dynamic_sample_inputs_mint_cumprod(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic-shape/rank inputs for mint.cumprod.
    """
    yield from dynamic_sample_inputs_mint_cumsum(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_diff(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = [
        (ms.Tensor(np.random.randn(5, 9), dtype=dtype), 1, 1, ms.Tensor(np.random.randn(5, 9), dtype=dtype),
         ms.Tensor(np.random.randn(5, 9), dtype=dtype)),
        (ms.Tensor(np.random.randn(3, 3, 3, 5), dtype=dtype), 1, -1, None, None),
        (ms.Tensor(np.random.randn(25, 123, 48), dtype=dtype), 1, 2,
         ms.Tensor(np.random.randn(25, 123, 48), dtype=dtype),
         ms.Tensor(np.random.randn(25, 123, 48), dtype=dtype))
    ]
    for op_inputs, n_p, dims, prepend, append in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(n_p, dims, prepend, append),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_diff(op_info, dtype, device=None, **kwargs):
    cases = [
        # discontinuous
        (ms.Tensor(np.random.randn(10, 15), dtype=dtype)[::2, ::3], 1, 0, None, None),
        (ms.Tensor(np.random.randn(8, 12), dtype=dtype), 2, 1, None, None),
        # broadcast
        (ms.Tensor(np.random.randn(1, 1, 3), dtype=dtype), 1, 1, None, None),
        #shape 0
        (ms.Tensor(np.random.randn(0, 5), dtype=dtype), 0, 0, None, None)

    ]
    for op_inputs, n_p, dims, prepend, append in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(n_p, dims, prepend, append),
            op_kwargs={},
            sample_name=op_info.name,
        )
    #small/large values
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
                op_args=(1, 0, None, None),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )
    # nan.inf/np.nan.inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((5, 5), t), dtype=dtype, device=device),
                op_args=(1, 0, None, None),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )


def error_inputs_mint_diff(op_info, dtype, device=None, **kwargs):
    cases = [
        (0, 1, 0, None, None),#input scalar
        (ms.Tensor(np.random.randn(8, 12), dtype=dtype), "2", 1, None, None),# n_p str
        (ms.Tensor(np.random.randn(5, 1, 3), dtype=dtype), 1, "1",
         ms.Tensor(np.random.randn(1, 9, 3), dtype=dtype),
         ms.Tensor(np.random.randn(5, 1, 1), dtype=dtype)),#dim str
        (ms.Tensor(np.random.randn(0, 5), dtype=dtype), 1, 0, 0, None),#prepend int
        (ms.Tensor(np.random.randn(0, 5), dtype=dtype), 1, 0, None, 0),#append int
        # prepend dim not match
        (ms.Tensor(np.random.randn(5, 9), dtype=dtype), 1, 0, ms.Tensor(np.random.randn(3, 9), dtype=dtype), None),
        # append dim not match
        (ms.Tensor(np.random.randn(5, 9), dtype=dtype), 1, 0, None, ms.Tensor(np.random.randn(7, 9), dtype=dtype)),
    ]
    for op_inputs, n_p, dims, prepend, append in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_inputs,
                op_args=(n_p, dims, prepend, append),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, RuntimeError, TypeError),
            op_error_info='other ops mint.diff error values',
        )

def dynamic_sample_inputs_mint_diff(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate dynamic inputs for mint.diff ops.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpDynamicInput objects.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    n_p, dims, prepend, append = 1, 0, None, None
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 3, None), dtype=dtype),
                op_args=(n_p, dims, prepend, append),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 3, 7)),
                    op_args=(n_p, dims, prepend, append),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 9)),
                    op_args=(n_p, dims, prepend, append),
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
                op_args=(n_p, dims, prepend, append),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 6)),
                    op_args=(n_p, dims, prepend, append),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 4)),
                    op_args=(n_p, dims, prepend, append),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

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

    for inputs, sorts, return_inverse, return_counts, dims in cases:
        yield OpSampleInput(
            op_input=inputs,
            op_args=(sorts, return_inverse, return_counts, dims),
            op_kwargs={},
            sample_name=op_info.name,
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

def extra_sample_inputs_mint_log_softmax_or_softmax(op_info: OpInfo, dtype=None, device=None, **kwargs):

    yield from make_extra_tensor(op_info, (3, 6, 1, 1), (), {'dim': -3}, dtype=dtype,
                                 device=device, )

    yield from make_extra_tensor(op_info, (2, 3, 4, 5, 6, 7, 8, 9), (), {'dim': 3}, dtype=dtype,
                                 device=device, support_multi_dim=True)

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

def basic_sample_inputs_mint_cummax(op_info, dtype, device=None, **kwargs):
    """Generate sample inputs for mit.cummax operations.

    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device to create tensors on.
        **kwargs: Additional keyword arguments.

    Yields:
        OpSampleInput: Sample input with tensor and dims int.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    test_cases = [
        ((2, 3), 1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((4,), 0),
        ((2, 1, 3), 2),
        ((100,), 0),
        ((2, 3), 0),
        ((2, 3, 4), 2),
    ]

    for shape, dims in test_cases:
        yield OpSampleInput(
            op_input=make_func(shape),
            op_args=(dims,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_shape{shape}_dims{dims}"
        )

def extra_sample_inputs_mint_cummax(op_info, dtype, device=None, **kwargs):
    cases = [
        # discontinuous
        (ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32)[:, ::2], dtype=dtype), 1),
        (ms.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), dtype=dtype), 1),
        # broadcast
        (ms.Tensor(np.array([[1, 3, 2], [4, 1, 5]]), dtype=dtype), 0),
    ]
    for op_inputs, dims in cases:
        yield OpSampleInput(
            op_input=op_inputs,
            op_args=(dims,),
            op_kwargs={},
            sample_name=op_info.name,
        )
    #small/large values
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
    # empty shape
    yield OpSampleInput(
        op_input=make_tensor(shape=(1, 0, 3), dtype=dtype, device=device),
        op_args=(1,),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_shape",
    )

def error_inputs_mint_cummax(op_info: OpInfo, dtype=None, device=None, **kwargs):

    cases = [
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), -3),#dim valueerror
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), "1"),
        (ms.Tensor(np.random.randn(7, 8), dtype=ms.float32), True),
        (ms.Tensor(np.random.randn(9, 8), dtype=ms.float32), (1, 2)),
        (ms.Tensor(np.random.randn(9, 8), dtype=ms.float32), 1.5)
    ]
    for op_input, dims in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=op_input,
                op_args=(dims,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info='other ops mint.cummax error values'
        )

def basic_reference_inputs_tensor_amax_or_amin(op_info: OpInfo, dtype=None, device=None, **kwargs):
    is_kbk = 'kbk' in op_info.name
    for sample_name in basic_reference_inputs_reduction_op_common_func(op_info=op_info, dtype=dtype, device=device):
        yield from convert_params_name(sample_name, args_param=(is_kbk, ))

def extra_reference_inputs_tensor_amax_or_amin(op_info: OpInfo, dtype=None, device=None, **kwargs):
    is_kbk = 'kbk' in op_info.name
    for sample_name in extra_reference_inputs_reduction_op_common_func(op_info=op_info, dtype=dtype, device=device):
        yield from convert_params_name(sample_name, args_param=(is_kbk, ))

def dynamic_reference_inputs_tensor_amax_or_amin(op_info: OpInfo, dtype=None, device=None, **kwargs):
    is_kbk = 'kbk' in op_info.name
    for sample_name in dynamic_inputs_reduction_op_common_func(op_info=op_info, dtype=dtype, device=device):
        yield from convert_params_name(sample_name, args_param=(is_kbk, ))

def basic_sample_inputs_tensor_count_nonzero(op_info: OpInfo, dtype=None, device=None, **kwargs):
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

def extra_sample_inputs_tensor_count_nonzero(op_info: OpInfo, dtype=None, device=None, **kwargs):
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

def error_inputs_other_op_tensor_count_nonzero(op_info: OpInfo, dtype=None, device=None, **kwargs):
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
