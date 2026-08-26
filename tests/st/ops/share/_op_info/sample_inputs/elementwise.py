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
# pylint: skip-file
"""
Custom sample input functions for operation testing.
This module provides sample input builders for unary/binary/logical/clamp various ops, including
basic, extra, dynamic, and error case generators.
"""
import functools
import numpy as np
import random
import warnings
import math
import itertools
import mindspore as ms
from mindspore import mint, mutable
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo,  \
    extra_reference_inputs_unary_op_common_func
from tests.st.ops.share._op_info.op_info import (
    _generate_binary_op_tensors_sample_inputs_func,
    _generate_binary_op_small_value_tensor_inputs_func,
    _generate_binary_op_large_value_tensor_inputs_func,
    _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func,
    _generate_binary_op_scalar_inputs_func,
    _generate_binary_op_extremal_value_tensor_inputs_func,
    basic_reference_inputs_binary_op_common_func,
    basic_reference_inputs_unary_op_common_func,
    extra_reference_inputs_binary_op_common_func,
    dynamic_inputs_binary_op_common_func,
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
    dtypes_extra_values,
    small_large_values,
    dtypes_as_torch
)
from tests.st.ops.share._op_info.sample_inputs.common import (
    _generate_binary_inplace_op_broadcasting_and_discontiguous_tensor_inputs_func,
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
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )

    # empty tensors/small medium large size
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
    values = small_large_values(dtype)
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

    extremal_cases = ([
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
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, )

    # empty tensors/small medium large size
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
    values = small_large_values(dtype)
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

    extremal_cases = ([
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
    #torch has error with input value 3
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9, exclude_value=(3, ))

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
    values = small_large_values(dtype)
    if values and dtype not in(ms.complex64, ms.complex128):
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
    if dtype not in(ms.complex64, ms.complex128):
        for extremal_val, name_suffix in extremal_cases:
            t = make_func((5,))
            t[2] = extremal_val
            yield OpSampleInput(
                op_input=t,
                op_args=(),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
            )

def basic_sample_inputs_mint_nn_softshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    input_shapes = (
        (3,), (1, 3), (1, 1, 3))
    lambd = (0, 1, 0.5, True)
    for input_shape, lambd in itertools.product(input_shapes, lambd):
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(lambd,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}"
        )

def extra_sample_inputs_mint_nn_softshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    input_shapes = (
        (), (2, 0), (9, 3, 8, 8, 7), (1, 1, 1, 1, 2, 2, 2, 2))
    lambd = (0, 15, 599.947890756524)
    for input_shape, lambd in itertools.product(input_shapes, lambd):
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(lambd,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}"
        )

    # small/large values
    values = small_large_values(dtype)
    if values:
        yield OpSampleInput(
            op_input=ms.Tensor(np.random.choice(values, size=(2, 3), replace=False), dtype=dtype),
            op_args=(0.5,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_small_large_vales"
        )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((2, 3), t), dtype=dtype),
            op_args=(1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

    # discontiguous tensor
    for discontiguous in (True,):
        yield OpSampleInput(
            op_input=make_arg((2, 3), discontiguous=discontiguous),
            op_args=(0,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def dynamic_sample_inputs_mint_nn_softshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.Softshrink.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 9, 8, 8, 3)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 8, 9, 7, 4)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9,)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 7, 7)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 9)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_softshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # input is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1.0,
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_not_tensor"
        ),
        op_error_type=TypeError,
        op_error_info=f'{op_info.name}: input is not Tensor'
    )
    # lambd is smaller than zero
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((3, 4)),
            op_args=(-1,),
            op_kwargs={},
            sample_name=f"{op_info.name}_neg_lambd"
        ),
        op_error_type=RuntimeError,
        op_error_info="lambd is small than zero"
    )
    # lambd is tuple
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(3, 4)),
            op_args=((1,),),
            op_kwargs={},
            sample_name=f"{op_info.name}_lambd_is_tuple"
        ),
        op_error_type=TypeError,
        op_error_info=f'lambd tuple value is not supported for {op_info.name}'
    )

def basic_sample_inputs_mint_nn_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((3,), 0, -3, False),
        ((7, 5), -3, 0.1, True),
        ((4, 9, 8, 7), -2, -2, False),
    )
    for input_shape, threshold, value, inplace in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(threshold, value),
            op_kwargs={'inplace': inplace},
            sample_name=f"{op_info.name}_{dtype}"
        )

def extra_sample_inputs_mint_nn_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((), 6, 6, False),
        ((2, 0), 2.0, -6, True),
        ((9, 8, 5, 9, 8), 0.4, 0.4, False),
        ((7, 5, 9, 4, 7, 4, 3, 8), -3, 0, True),
    )
    for input_shape, threshold, value, inplace in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(threshold, value),
            op_kwargs={'inplace': inplace},
            sample_name=f"{op_info.name}_{dtype}"
        )
    yield from extra_sample_inputs_mint_nn_threshold_small_large_values(op_info, dtype=dtype, device=device, **kwargs)

def extra_sample_inputs_mint_nn_threshold_small_large_values(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    # small/large values
    values = small_large_values(dtype)
    if values:
        yield OpSampleInput(
            op_input=ms.Tensor(np.random.choice(values, size=(2, 3), replace=False), dtype=dtype),
            op_args=(12.3, 12.3),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_small_large_vales"
        )
    # inf/nan values
    for t in [np.inf, -np.inf, np.nan]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((2, 3), t), dtype=dtype),
            op_args=(6, 6),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_{t}"
        )

    # discontiguous tensor
    for discontiguous in (True,):
        if dtype == ms.uint8:
            # for torch, negative value cannot be converted to type uint8 without overflow
            yield OpSampleInput(
                op_input=make_arg((2, 3), discontiguous=discontiguous),
                op_args=(0, 2.1),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
            )
        else:
            yield OpSampleInput(
                op_input=make_arg((2, 3), discontiguous=discontiguous),
                op_args=(0, -2.1),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
            )

def dynamic_sample_inputs_mint_nn_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    """
    Generate dynamic shape/rank for mint.nn.Threshold.
    """
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),
                op_args=(-3, 6),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5, 7, 3, 4)),
                    op_args=(-3, 6),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 3, 4, 5, 5)),
                    op_args=(-3, 6),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(0.4, 0.4),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 7)),
                    op_args=(0.4, 0.4),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 7, 9)),
                    op_args=(0.4, 0.4),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(None, dtype=dtype),
                op_args=(0, 0),
                op_kwargs={},
                sample_name=f'{op_info.name}_{dtype}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 8, 3)),
                    op_args=(0, 0),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 5)),
                    op_args=(0, 0),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_{dtype}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_mint_nn_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    cases = (
        ((2, 3), 2.0),
        (0, [2, 3]),
    )
    # threshold is tuple
    # value is list
    for threshold, value in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg(shape=(2, 3)),
                op_args=(threshold, value),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=TypeError,
            op_error_info=f'params error values is not supported for {op_info.name}',
        )
    # input is not Tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1.0,
            op_args=(0, -3),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_not_tensor"
        ),
        op_error_type=(AttributeError, TypeError),
        op_error_info=f'{op_info.name}: input is not Tensor'
    )

def basic_sample_inputs_functional_relu_(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    shapes = (
        (M,),
        (M, S),
        (S, S, L),
        (3, 1, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for input_shape in shapes:
        slen = len(input_shape)
        _input = make_arg(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_functional_relu_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    #empty tensors
    _input = make_func(shape=(1, 4, 0))
    yield OpSampleInput(
        _input,
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_tensor",
    )

    # large/small/discontiguous values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # nan/inf/-inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full([1024, 1024], t), dtype=dtype, device=device),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def basic_sample_inputs_functional_softplus(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    shapes = (
        ((M,), 1, 10),
        ((M, S), 0.4, 20),
        ((S, S, L), 0.5, 6),
        ((3, 1, 1), 2, 15),
        ((2, 1, 3, 2), 3, 30),
        ((2, 3, 4, 1, 2), 0, 1),
        ((2, 1, 2, 2, 1, 2), 1, 20),
        ((2, 1, 2, 2, 1, 2, 1), 1, 1),
        ((2, 1, 2, 2, 1, 2, 1, 2), 2, 8),
    )

    for input_shape, input_beta, input_threshold in shapes:
        slen = len(input_shape)
        _input = make_arg(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'beta':input_beta, 'threshold':input_threshold},
            sample_name=f"{op_info.name}_{dtype}_{slen}d",
        )

def extra_sample_inputs_functional_softplus(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    #empty tensors
    _input = make_func(shape=(1, 4, 0))
    yield OpSampleInput(
        _input,
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_empty_tensor",
    )

    # large/small/discontiguous values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # nan/inf/-inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full([1024, 1024], t), dtype=dtype, device=device),
                op_kwargs={},
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_other_op_functional_softplus(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # input not tensor
    cases = (2, None)
    for _input in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=_input,
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(ValueError, TypeError),
            op_error_info='input should be tensor ',
        )

def basic_reference_inputs_mint_nn_functional_softshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=9)

    shapes = (
        ((), 0),
        ((S,), 0.1),
        ((M, S), 0.5),
        ((S, S, L), 1),
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

def extra_sample_inputs_functional_softshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    #empty tensors
    _input = make_func(shape=(1, 0, 4))
    yield OpSampleInput(
        _input,
        op_args=(3,),
        sample_name=f"{op_info.name}_{dtype}_empty_tensor",
    )

    # large/small/discontiguous values
    values = small_large_values(dtype)
    if values and dtype not in(ms.complex64, ms.complex128):
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_args=(3,),
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # nan/inf/-inf  input is nan case failed
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full([1024, 1024], t), dtype=dtype, device=device),
                op_args=(2,),
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def basic_sample_inputs_mint_nn_functional_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.threshold
    yield from basic_sample_inputs_mint_nn_threshold(op_info, dtype=dtype, device=device, **kwargs)

def extra_sample_inputs_mint_nn_functional_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.threshold
    yield from extra_sample_inputs_mint_nn_threshold(op_info, dtype=dtype, device=device, **kwargs)

def dynamic_sample_inputs_mint_nn_functional_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.threshold
    yield from dynamic_sample_inputs_mint_nn_threshold(op_info, dtype=dtype, device=device, **kwargs)

def error_sample_inputs_mint_nn_functional_threshold(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.threshold
    yield from error_sample_inputs_mint_nn_threshold(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_nn_functional_threshold_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((3,), 0, -3),
        ((7, 5), -3, 0.1),
        ((4, 9, 8, 7), -2, -2),
    )
    for input_shape, threshold, value in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(threshold, value),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}"
        )

def extra_sample_inputs_mint_nn_functional_threshold_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((), 6, 6),
        ((2, 0), 2.0, -6),
        ((9, 8, 5, 9, 8), 0.4, 0.4),
        ((7, 5, 9, 4, 7, 4, 3, 8), -3, 0),
    )
    for input_shape, threshold, value in cases:
        yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(threshold, value),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}"
        )
    yield from extra_sample_inputs_mint_nn_threshold_small_large_values(op_info, dtype=dtype, device=device, **kwargs)

def dynamic_sample_inputs_mint_nn_functional_threshold_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.threshold
    yield from dynamic_sample_inputs_mint_nn_threshold(op_info, dtype=dtype, device=device, **kwargs)

def error_sample_inputs_mint_nn_functional_threshold_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # Reuse the same sample inputs as mint.nn.threshold
    yield from error_sample_inputs_mint_nn_threshold(op_info, dtype=dtype, device=device, **kwargs)

def basic_sample_inputs_mint_where(op_info, dtype=None, device=None, **kwargs):
    cases = (
        (ms.Tensor(np.random.randn(), dtype=ms.bool_), ms.Tensor(np.random.randn(), dtype),
         ms.Tensor(np.random.randn(), dtype),),
        (ms.Tensor(np.random.randn(4, 10), dtype=ms.bool_), ms.Tensor(np.random.randint(-2, 2, (4, 10)), dtype),
         ms.Tensor(np.random.randint(-2, 2, (4, 10)), dtype),),
        (ms.Tensor(np.random.randn(7, 9, 3), dtype=ms.bool_),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),),
        (ms.Tensor(np.random.randint(0, 255, (7, 4, 8, 3)), dtype) > 125,
         ms.Tensor(np.random.randint(0, 255, (7, 4, 8, 3)), dtype),
         ms.Tensor(np.random.randint(0, 255, (7, 4, 1, 3)), dtype),),
        (ms.Tensor(np.random.randn(10, 8, 7, 10, 3, 3), dtype=ms.bool_),
         ms.Tensor(np.random.randint(-125, 125, (10, 8, 7, 10, 3, 3)), dtype),
         ms.Tensor(np.random.randint(-125, 125, (10, 8, 7, 10, 3, 3)), dtype),),
        (ms.Tensor(np.random.randn(3, 4, 4, 4, 5, 5, 5, 5), dtype=ms.bool_),
         ms.Tensor(np.random.randn(3, 4, 4, 4, 5, 5, 5, 5), dtype),
         ms.Tensor(np.random.randn(3, 4, 4, 4, 5, 5, 5, 5), dtype),),
        (ms.Tensor(np.random.randn(10, 8, 7), dtype=ms.bool_),
         ms.Tensor(np.random.randn(10, 8, 7), dtype), 3.0,),
    )
    for condition, inputs, other in cases:
        yield OpSampleInput(
            op_input=condition,
            op_args=(inputs, other,),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}"
        )

def error_inputs_other_op_mint_where_func(op_info, dtype=None, device=None, **kwargs):
    cases = (
        (True,
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),),#condition is not bool
        (ms.Tensor(np.random.randn(7, 9, 3), dtype=ms.bool_),
         "[1.0, 2,0]",
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),),#input not tensor
        (ms.Tensor(np.random.randn(7, 9, 3), dtype=ms.bool_),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),
         "[1.0, 2,0]",),#other not tensor
        (ms.Tensor(np.random.randn(8,), dtype=ms.bool_),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),),# cant broadcast with other
        (ms.Tensor(np.random.randn(7, 9, 3), dtype=ms.int8),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),
         ms.Tensor(np.random.randint(0, 255, (7, 9, 3)), dtype),),#condition not bool_
    )
    for condition, inputs, other in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=condition,
                op_args=(inputs, other,),
                op_kwargs={},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info='other ops mint.where error values'
        )

def extra_sample_inputs_mint_where(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.randn(10, 0, 7), dtype=ms.bool_),
        op_args=(ms.Tensor(np.random.randn(10, 0, 7), dtype=dtype),
                 ms.Tensor(np.random.randn(10, 0, 7), dtype=dtype),),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}"
    )
    #small/large values
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(2, 3, 4)), dtype=dtype) > 10,
        op_args=(ms.Tensor(np.random.choice(values, size=(2, 3, 4)), dtype=dtype),
                 ms.Tensor(np.random.choice(values, size=(2, 3, 4)), dtype=dtype),),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    #inf/nan values
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(3, 3, 4)), dtype=dtype) > 10,
        op_args=(ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(3, 3, 4)), dtype=dtype),
                 ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(3, 3, 4)), dtype=dtype)),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_extremal_values"
    )
    #discontiguous value
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg(shape=(4, 3), discontiguous=discontiguous) > 3,
            op_args=(make_arg(shape=(4, 3), discontiguous=discontiguous),
                     make_arg(shape=(4, 3), discontiguous=discontiguous),),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}"
        )

def dynamic_sample_inputs_mint_where(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=ms.bool_),
                op_args=(ms.Tensor(shape=None, dtype=dtype),
                         ms.Tensor(shape=None, dtype=dtype)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 4, 8, 8, 7), dtype=ms.bool_),
                    op_args=(ms.Tensor(shape=(8, 4, 8, 8, 7), dtype=dtype),
                             ms.Tensor(shape=(8, 4, 8, 8, 7), dtype=dtype)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4, 8), dtype=ms.bool_),
                    op_args=(ms.Tensor(shape=(2, 3, 4, 8), dtype=dtype), ms.Tensor(shape=(2, 3, 4, 8), dtype=dtype)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=ms.bool_),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype),
                         ms.Tensor(shape=(None, None, None), dtype=dtype)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 9, 7), dtype=ms.bool_),
                    op_args=(ms.Tensor(shape=(8, 9, 7), dtype=dtype), ms.Tensor(shape=(8, 9, 7), dtype=dtype)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4), dtype=ms.bool_),
                    op_args=(ms.Tensor(shape=(2, 3, 4), dtype=dtype), ms.Tensor(shape=(1, 3, 4), dtype=dtype)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def basic_sample_inputs_functional_prelu(op_info: OpInfo, dtype, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    shapes = (
        (M, S),
        (S, S, L),
        (3, 1, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for input_shape in shapes:
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        _input = make_arg(input_shape)
        yield OpSampleInput(
            _input,
            op_kwargs={'weight':weight,},
            sample_name=op_info.name,
        )

def extra_sample_inputs_functional_prelu(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype, low=0)

    #empty tensors
    _input = make_func(shape=(1, 4, 0))
    yield OpSampleInput(
        _input,
        op_kwargs={'weight': make_func(1), },
        sample_name=f"{op_info.name}_{dtype}_empty_tensor",
    )

    # large/small/discontiguous values
    values = small_large_values(dtype)
    if values:
        np_array = np.random.choice(values, size=(2, 3), replace=False)
        for discontiguous in [True, False]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np_array, dtype=dtype, discontiguous=discontiguous),
                op_kwargs={'weight': make_func(1), },
                sample_name=f"{op_info.name}_{dtype}_repeats_large_mix_small_values_discontiguous_{discontiguous}",
            )

    # nan/inf/-inf
    if dtype.is_floating_point or dtype.is_complex:
        for t in [np.nan, np.inf, -np.inf]:
            yield OpSampleInput(
                op_input=make_tensor_with_np_array(np.full([1024, 1024], t), dtype=dtype, device=device),
                op_kwargs={'weight': make_func(1), },
                sample_name=f"{op_info.name}_{dtype}_{t}",
            )

def error_inputs_other_op_functional_prelu(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # weight not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_kwargs={'weight':1,},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info='weight should be tensor ',
    )
    # weight not 1-d
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_kwargs={'weight':make_arg((2, 3)),},
            sample_name=op_info.name,
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info='weight should be 0-1or1-d ',
    )

def basic_sample_inputs_mint_index_add(op_info: OpInfo, dtype=None, device=None):
    """
    Generate basic sample inputs for mint.narrow op.
    """
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((5,), 0, ms.Tensor([0, 1, 2], dtype=ms.int32), make_input((3,)), 1.0),
        ((2, 4), 1, ms.Tensor([0, 2, 3], dtype=ms.int32), make_input((2, 3)), 1.0),
        ((3, 2, 2), 0, ms.Tensor([0, 2], dtype=ms.int32), make_input((2, 2, 2)), -1.0),
        ((2, 3, 4), 2, ms.Tensor([1, 3], dtype=ms.int32), make_input((2, 3, 2)), 2.0),
        ((4,), 0, ms.Tensor([0, 0, 2], dtype=ms.int32), make_input((3,)), 1.0),
    )
    for shape, dims, index, sources, alpha in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(dims, index, sources,),
            op_kwargs={"alpha": alpha},
            sample_name=op_info.name,
        )

def error_inputs_other_op_mint_index_add_func(op_info, dtype=None, device=None, **kwargs):
    #make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    make_input = functools.partial(make_tensor, device=device, dtype=ms.float32)

    cases = (
        ((4,), 0, [0, 1], make_input((2,)), 1.0), #index not tensor
        ((2, 4), 5, ms.Tensor([0], dtype=ms.int32), make_input((2, 3)), 1.0),#dim out of range
        ((3, 2, 2), 0, ms.Tensor([0, 2], dtype=ms.int32), [2.0, 2.0, 2.0], -1.0),#source is not tensor
        ((4,), 0, ms.Tensor([0, 1, 2], dtype=ms.int32), make_input((2,)), 1.0),#index length != source size at dim
        ((4, 4), 0, ms.Tensor([[0, 1]], dtype=ms.int32), make_input((2, 4)), 1.0), #index is not 1d
        ((2, 4, 5), 1, ms.Tensor([[0, 1]], dtype=ms.int32),
         make_input((2, 2, 6)), 1.0), #input and source shape mismatch in non-dim dimensions
    )
    for shape, dims, index, sources, alpha in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_input(shape),
                op_args=(dims, index, sources,),
                op_kwargs={"alpha": alpha},
                sample_name=op_info.name,
            ),
            op_error_type=(TypeError, ValueError, RuntimeError),
            op_error_info='other ops mint.index_add error values'
        )

def extra_sample_inputs_mint_index_add(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mindspore.mint.index_add
    Reference torch's module tests for index_add.
    '''
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((0, 2), 0, ms.Tensor([], dtype=ms.int32), make_input((0, 2)), 1.0),#shape 0
        ((2, 3), 1, ms.Tensor([0, 2], dtype=ms.int32), make_input((2, 2)), 1.0),#shape mem continuous
        ((3, 4), 0, ms.Tensor([0, 1], dtype=ms.int32), make_input((2, 4)), 1.0),
    )
    for shape, dims, index, sources, alpha in cases:
        yield OpSampleInput(
            op_input=make_input(shape),
            op_args=(dims, index, sources,),
            op_kwargs={"alpha": alpha},
            sample_name=op_info.name,
        )

        # ---- small value tensors ----
    if dtype is not None and dtype.is_floating_point:
        _float_small_values = dtypes_extra_values['_float_small_values']
        input_values = np.array(_float_small_values).reshape(1, len(_float_small_values), 1)
        index_values = np.array(_float_small_values[:1])
        source_values = np.array(_float_small_values).reshape(1, len(_float_small_values), 1)
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(0, make_tensor_with_np_array(index_values, dtype=ms.int32, device=device),
                     make_tensor_with_np_array(source_values, dtype=dtype, device=device)),
            op_kwargs={"alpha": 1.0},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_small_value',
        )

        # ---- large value tensors ----
    if dtype is not None and dtype.is_floating_point:
        if dtype == ms.float16:
            _large_values = dtypes_extra_values['_fp16_large_values']
        else:
            _large_values = dtypes_extra_values['_fp32_large_values']
        input_values = np.array(_large_values).reshape(1, len(_large_values))
        index_values = np.zeros(len(_large_values))
        source_values = np.array(_large_values).reshape(1, len(_large_values))
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
            op_args=(1, make_tensor_with_np_array(index_values, dtype=ms.int64, device=device),
                     make_tensor_with_np_array(source_values, dtype=dtype, device=device)),
            op_kwargs={"alpha": 1.0},
            sample_name=f'{op_info.name}_{dtype}_default_extremal_large_value',
        )

        # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            for t in [np.inf, -np.inf, np.nan]:
                input_values = np.full((7, 9, 3), t)
                index = ms.Tensor(np.random.randint(0, 3, (3,)), dtype=ms.int32)
                source_values = np.full((7, 9, 3), t)
                yield OpSampleInput(
                    op_input=make_tensor_with_np_array(input_values, dtype=dtype, device=device),
                    op_args=(2, index, make_tensor_with_np_array(source_values, dtype=dtype, device=device),),
                    op_kwargs={"alpha": float('inf')},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{t}',
                )

def dynamic_sample_inputs_mint_index_add(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic-shape/rank inputs for mint.index_ADD.

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
                op_args=(0, ms.Tensor(shape=(None,), dtype=ms.int32), ms.Tensor(shape=(None, 5, None, None), dtype=dtype),),
                op_kwargs={"alpha": 1.0},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 5, 4, 9)),
                    op_args=(0, ms.Tensor(shape=(2,), dtype=ms.int32), make_func(shape=(2, 5, 4, 9))),
                    op_kwargs={"alpha": 1.0},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 5, 9, 7)),
                    op_args=(0, ms.Tensor(shape=(3,), dtype=ms.int32), make_func(shape=(3, 5, 9, 7))),
                    op_kwargs={"alpha": 1.0},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0, ms.Tensor(shape=(None,), dtype=ms.int32), ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={"alpha": 1.0},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 4, 7, 7)),
                    op_args=(0, ms.Tensor(shape=(3,), dtype=ms.int32), make_func(shape=(3, 4, 7, 7)),),
                    op_kwargs={"alpha": 1.0},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 4, 7, 3, 4, 9)),
                    op_args=(0, ms.Tensor(shape=(2,), dtype=ms.int32), make_func(shape=(2, 8, 4, 7, 3, 4, 9)),),
                    op_kwargs={"alpha": 1.0},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(1, ms.Tensor(shape=(None,), dtype=ms.int32), ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={"alpha": 1.0},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input_groups2",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 5)),
                    op_args=(1, ms.Tensor(shape=(3,), dtype=ms.int32), make_func(shape=(4, 3)),),
                    op_kwargs={"alpha": 1.0},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(1, 6, 3, 4, 1)),
                    op_args=(1, ms.Tensor(shape=(5,), dtype=ms.int32), make_func(shape=(1, 5, 3, 4, 1)),),
                    op_kwargs={"alpha": 1.0},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input_groups2",
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

    values = small_large_values(dtype)
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
            op_error_info='other ops mint.isclose error values'
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

def basic_sample_inputs_tensor_clamp(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for tensor.clamp_ op.
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
        op_error_type=(ValueError,RuntimeError),
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
        op_error_type=(ValueError, RuntimeError),
        op_error_info=f'min error values is not supported for {op_info.name}',
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

def basic_reference_inputs_tensor_astype(op_info, dtype=None, device=None, **kwargs):
    for sample_input in basic_reference_inputs_unary_op_common_func(op_info, dtype=dtype, device=device, **kwargs):
        arg_dtype = random.choice(dtypes_as_torch)
        # loss error, arg_dtype is uint8
        if dtype is ms.float64 and ms.ops.any(sample_input.op_input < 0) and arg_dtype is ms.uint8:
            # any() does not support complex, determining whether there are negative numbers.
            arg_dtype = ms.int8
        elif dtype.is_complex and arg_dtype is ms.uint8:
            arg_dtype = ms.int8
        sample_input.op_args=(arg_dtype,)
        sample_input.op_kwargs.update({'copy': random.choice([True, False])})
        # int do not support backward
        if not arg_dtype.is_complex and not arg_dtype.is_floating_point:
            sample_input.grad_position=()
        yield sample_input

def extra_reference_inputs_tensor_astype(op_info, dtype=None, device=None, **kwargs):
    for sample_input in extra_reference_inputs_unary_op_common_func(op_info, dtype=dtype, device=device, **kwargs):
        arg_dtype = random.choice(dtypes_as_torch)
        # loss error, arg_dtype is uint8
        if dtype is ms.float64 and ms.ops.any(sample_input.op_input < 0) and arg_dtype is ms.uint8:
            # any() does not support complex, determining whether there are negative numbers.
            arg_dtype = ms.int8
        elif dtype.is_complex and arg_dtype is ms.uint8:
            arg_dtype = ms.int8
        sample_input.op_args=(arg_dtype,)
        sample_input.op_kwargs.update({'copy': random.choice([True, False])})
        # int do not support backward
        if not arg_dtype.is_complex and not arg_dtype.is_floating_point:
            sample_input.grad_position=()
        yield sample_input
    # dtype is str
    yield OpSampleInput(
        op_input=make_tensor(shape=(1, 6), dtype=dtype),
        op_args=('float32',),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_output_dtype_str_extra",
    )

def dynamic_inputs_unary_op_tensor_astype(op_info, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # int do not support backward
    arg_dtype = random.choice((ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.bfloat16))
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(arg_dtype,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(arg_dtype,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(arg_dtype,),
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
                op_args=(arg_dtype,),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(arg_dtype,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(arg_dtype,),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_inputs_unary_op_tensor_astype(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (
        # dtype is str of mindspore.dtype
        ('ms.int32', False),
        # copy is not bool
        # (ms.int32, 'False')
    )
    for dtype, copy in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_tensor(shape=(2, 2, 1), dtype=ms.float32),
                op_args=(dtype,),
                op_kwargs={'copy': copy},
                sample_name=f"{op_info.name}_error_dtype_copy",
            ),
            op_error_type=TypeError,
            op_error_info=f'error dtype or copy is not supported for {op_info.name}',
        )

def basic_reference_inputs_leaky_relu(op_info, dtype=None, device=None, **kwargs):
    yield from basic_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    yield OpSampleInput(
        op_input=make_tensor((2,), dtype=dtype, device=device,),
        op_kwargs={'negative_slope': random.choice([0.01, 1, -0.01])},
        sample_name=f'{op_info.name}_{dtype}_minus-dim_2'
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
        op_error_info=f'negative_slope error type is not supported for {op_info.name}',
    )

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
        alpha = -2.5 if (dtype.is_floating_point or dtype.is_complex) else -2
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': alpha},
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

def basic_reference_inputs_tensor_bitwise_and(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=9)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    shapes = (
        (M,),
        (M, S),
        (S, S, L),
        (3, 1, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )
    for input_shape in shapes:
        yield OpSampleInput(
             op_input=make_arg(input_shape),
             op_args=(make_arg(input_shape),),
             sample_name=f"{op_info.name}_{dtype}_{input_shape}",
        )

def extra_reference_inputs_tensor_bitwise_and(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=1)
    yield OpSampleInput(
        op_input=make_arg((256, 120, 10, 10)),
        op_args=(make_arg((256, 120, 10, 10)),),
        sample_name=f"{op_info.name}_{dtype}_input_with_big_input",
    )
    # case with small/large values
    float_vals = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8)), dtype=dtype),
        op_args=(make_arg((6, 7, 8)),),
        sample_name=f"{op_info.name}_{dtype}_small_large_values",
    )
    #case with inf/nan values
    for t in [np.nan, np.inf, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((6, 7, 8), t), dtype=dtype),
            op_args=(make_arg((6, 7, 8)),),
            sample_name=f"{op_info.name}_{dtype}_{t}",
        )

    #case with discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8), discontiguous=discontiguous),
            op_args=(make_arg((6, 7, 8)),),
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}",
        )

def basic_reference_inputs_tensor_bitwise_not(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=9)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    shapes = (
        (M,),
        (M, S),
        (S, S, L),
        (3, 1, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )
    for input_shape in shapes:
        yield OpSampleInput(
             op_input=make_arg(input_shape),
             op_args=(),
             sample_name=f"{op_info.name}_{dtype}_{input_shape}",
        )

def extra_reference_inputs_tensor_bitwise_not(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=0, high=1)
    yield OpSampleInput(
        op_input=make_arg((256, 120, 10, 10)),
        op_args=(),
        sample_name=f"{op_info.name}_{dtype}_input_with_big_input",
    )
    # case with small/large values
    float_vals = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8)), dtype=dtype),
        op_args=(),
        sample_name=f"{op_info.name}_{dtype}_small_large_values",
    )
    # case with inf/nan values
    for t in [np.nan, np.inf, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((6, 7, 8), t), dtype=dtype),
            op_args=(),
            sample_name=f"{op_info.name}_{dtype}_{t}",
        )

    # case with discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8), discontiguous=discontiguous),
            op_args=(),
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}",
        )

def dynamic_sample_inputs_tensor_bitwise_xor(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # dynamic shape
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
                op_args=(make_func(shape=(2, 2, 3), ),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 4, 5), ),
                op_args=(make_func(shape=(2, 4, 5), ),),
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
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2, 3), ),
                    op_args=(make_func(shape=(2, 2, 3), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 5), ),
                    op_args=(make_func(shape=(2, 4, 5), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def dynamic_sample_inputs_tensor_bitwise_not(op_info:OpInfo, dtype=None, device=None, **kwargs):
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

def basic_reference_inputs_tensor_copysign(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE
    shapes = (
        (M,),
        (M, S),
        (S, S, L),
        (3, 1, 1),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (1, 1, 2, 2, 1, 2, 1, 2),
    )
    for input_shape in shapes:
        yield OpSampleInput(
             op_input=make_arg(input_shape),
             op_args=(make_arg(input_shape),),
             sample_name=f"{op_info.name}_{dtype}_{input_shape}",
        )

def extra_reference_inputs_tensor_copysign(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    yield OpSampleInput(
        op_input=make_arg((256, 120, 10, 10)),
        op_args=(make_arg((256, 120, 10, 10)),),
        sample_name=f"{op_info.name}_{dtype}_input_with_big_input",
    )
    # case with small/large values
    float_vals = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(float_vals, size=(6, 7, 8)), dtype=dtype),
        op_args=(make_arg((6, 7, 8)),),
        sample_name=f"{op_info.name}_{dtype}_small_large_values",
    )
    #case with inf/nan values
    for t in [np.nan, np.inf, -np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.full((6, 7, 8), t), dtype=dtype),
            op_args=(make_arg((6, 7, 8)),),
            sample_name=f"{op_info.name}_{dtype}_{t}",
        )

    #case with discontiguous values
    for discontiguous in [True, False]:
        yield OpSampleInput(
            op_input=make_arg((6, 7, 8), discontiguous=discontiguous),
            op_args=(make_arg((6, 7, 8)),),
            sample_name=f"{op_info.name}_{dtype}_discontiguous_{discontiguous}",
        )

def dynamic_sample_inputs_tensor_copysign(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # dynamic shape
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
                op_args=(make_func(shape=(2, 2, 3), ),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
            OpSampleInput(
                op_input=make_func(shape=(2, 4, 5), ),
                op_args=(make_func(shape=(2, 4, 5), ),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_running_input",
            ),
        ),
    )
    if not kwargs.get("only_dynamic_shape", False):
        # dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2, 3), ),
                    op_args=(make_func(shape=(2, 2, 3), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 5), ),
                    op_args=(make_func(shape=(2, 4, 5), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def dynamic_sample_inputs_tensor_copy_(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    # dynamic shape
    if not kwargs.get("only_dynamic_rank", False):
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
                    op_args=(make_func(shape=(2, 2, 3), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 5), ),
                    op_args=(make_func(shape=(2, 4, 5), ),),
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
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 2, 3), ),
                    op_args=(make_func(shape=(2, 2, 3), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 4, 5), ),
                    op_args=(make_func(shape=(2, 4, 5), ),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def basic_reference_inputs_mint_polar(op_info, dtype=None, device=None, **kwargs):
    for sample_input in basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs):
        input_shape = sample_input.op_input.shape
        other_shape = sample_input.op_args[0].shape
        if input_shape != other_shape:
            continue
        yield sample_input

def extra_reference_inputs_mint_polar(op_info, dtype=None, device=None, **kwargs):
    for sample_input in extra_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs):
        input_shape = sample_input.op_input.shape
        other_shape = sample_input.op_args[0].shape
        if input_shape != other_shape:
            continue
        yield sample_input

def basic_reference_inputs_tensor_lerp(op_info, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device)
    for sample_input in basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs):
        input_shape = sample_input.op_input.shape
        weight=(make_arg(input_shape, dtype=dtype), )
        if dtype not in (ms.float16, ms.bfloat16,):
            weight =weight + (-0.2, 0.8, )
        if '_kbk' not in op_info.name:
            weight = weight + (-10, 0, 2)
        sample_input.op_args = sample_input.op_args + (random.choice(weight),)
        yield sample_input

def extra_reference_inputs_tensor_lerp(op_info, dtype=None, device=None, **kwargs):
    values = small_large_values(dtype=dtype)
    for sample_input in extra_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs):
        if dtype in (ms.float16, ms.bfloat16,):
            sample_input.op_args = sample_input.op_args + (ms.Tensor(2, dtype=dtype), )
        else:
            if any(s in sample_input.sample_name for s in ('small', 'large',)):
                sample_input.op_args = sample_input.op_args + (np.random.choice(values),)
            else:
                sample_input.op_args = sample_input.op_args + (0.5,)
        yield sample_input

def dynamic_reference_inputs_tensor_lerp(op_info, dtype=None, device=None, **kwargs):
    for dynamic_input in dynamic_inputs_binary_op_common_func(op_info, dtype, device, **kwargs):
        compile_input = dynamic_input.op_compile_input
        compile_input.op_args = compile_input.op_args + (1.5, )
        for running_input in dynamic_input.op_running_inputs:
            running_input.op_args = running_input.op_args + (1.5,)
        yield dynamic_input

def error_reference_inputs_tensor_lerp(op_info, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    #shape not match
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(2, )),
            op_args=(make_func(shape=(3, )), 1.0),
            op_kwargs={},
            sample_name=f"{op_info.name}_shape_not_match",
        ),
        op_error_type=ValueError,
        op_error_info=f'shape not match',
    )

def basic_reference_inputs_binary_op_common_func_tensor_divide(op_info: OpInfo,  dtype=None, device=None, **kwargs):

    for rounding_mode in (None, 'floor', 'trunc'):
        for sample_input in basic_reference_inputs_binary_op_common_func(op_info=op_info, dtype=dtype, device=device,
                                                                         **kwargs):
            sample_input.op_kwargs = {'rounding_mode': rounding_mode}
            sample_input.sample_name = sample_input.sample_name + f'_rounding_mode_{rounding_mode}'
            yield sample_input

def extra_reference_inputs_binary_op_common_func_tensor_divide(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    for rounding_mode in (None, 'floor', 'trunc'):
        for sample_input in extra_reference_inputs_binary_op_common_func(op_info=op_info, dtype=dtype, device=device, **kwargs):
            sample_input.op_kwargs = {'rounding_mode': rounding_mode}
            sample_input.sample_name = sample_input.sample_name + f'_rounding_mode_{rounding_mode}'
            yield sample_input

def error_inputs_binary_op_tensor_divide(op_info:OpInfo, dtype=None, device=None, **kwargs):

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2, 3), dtype=ms.float32),
            op_args=(make_tensor(shape=(2, 4), dtype=ms.float32), ),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='input and other must be equal or broadcasted',
    )

def error_inputs_tensor_digamma(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=ms.float32)
    # input shape is (0,)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(0,)),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_element_null",
        ),
        op_error_type=ValueError,
        op_error_info=f'input shape (0,) is not supported for {op_info.name}',
    )

def error_inputs_tensor_div(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from error_inputs_binary_op_mint_div(op_info, dtype=dtype, device=device, **kwargs)
    cases = (
        # other is not tensor/number/bool
        ((1, False), 'trunc'),
        # rounding_mode is not None/floor/trunc
        (1, 'trun'),
    )
    for other, rounding_mode in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_tensor(shape=(2,), dtype=ms.float32),
                op_args=(other,),
                op_kwargs={'rounding_mode': rounding_mode},
                sample_name=f"{op_info.name}_params_error"
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f'error other {other} or rounding_mode {rounding_mode} not supported for {op_info.name} ',
        )

def basic_sample_inputs_tensor_div_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    XS = EXTRA_SMALL_DIM_SIZE
    S = EXTRA_SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE

    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    shapes = (
        ((), ()),
        ((S,), ()),
        ((S, 1), (1,)),
        ((M, S), ()),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, M, S), (M, S)),
        ((M, 1, S), (1, 1, S)),
        ((0, M, XS), (0, M, 1)),
    )

    for input_shape, other_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low,
                           high=op_info.input_high, exclude_value=op_info.input_exclude_value)
        _other = make_func(other_shape, low=op_info.other_low,
                           high=op_info.other_high, exclude_value=op_info.other_exclude_value)

        _splice = "empty_" if 0 in _input.shape + _other.shape else ""
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=f"{op_info.name}_{dtype}_{_splice}{len(_input.shape)}d_x_{len(_other.shape)}d",
        )
    # Add kwarg rounding_mode
    _input = make_func((S, S), low=op_info.input_low,
                       high=op_info.input_high, exclude_value=op_info.input_exclude_value)
    _other = make_func((S, S), low=op_info.other_low,
                       high=op_info.other_high, exclude_value=op_info.other_exclude_value)

    others = [_other, 1, -0.93, 6.436]
    rounding_mode = ['floor', 'trunc']
    for other, rounding_mode in itertools.product(others, rounding_mode):
        if not dtype.is_complex:
            yield OpSampleInput(
                op_input=_input,
                op_args=(other,),
                op_kwargs={'rounding_mode': rounding_mode},
                sample_name=f"{op_info.name}_{dtype}_rounding_mode_{rounding_mode}"
            )

def extra_sample_inputs_tensor_div_(op_info:OpInfo, dtype=None, device=None, **kwargs):
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
    # tensor with extremal value
    if not op_info.disable_extremal_value_tensor_inputs and (dtype.is_floating_point or dtype.is_complex):
        yield from _generate_binary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)

def error_sample_inputs_tensor_gcd(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.int64, device=device)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((1, 2, 1)),
            op_args=(2,),
            op_kwargs={},
            sample_name=f"{op_info.name}_params_error"
        ),
        op_error_type=TypeError,
        op_error_info=f'other not tensor not supported for {op_info.name}',
    )

def dynamic_sample_inputs_tensor_hardshrink(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(),
                op_kwargs={"lambd": mutable(input_data=1, dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(4, 8, 5)),
                    op_args=(),
                    op_kwargs={"lambd": mutable(input_data=2, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 9, 4)),
                    op_args=(),
                    op_kwargs={"lambd": mutable(input_data=8, dynamic_len=False)},
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
                op_kwargs={"lambd": mutable(input_data=1.4, dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 8, 4)),
                    op_args=(),
                    op_kwargs={"lambd": mutable(input_data=1.1, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 7, 5)),
                    op_args=(),
                    op_kwargs={"lambd": mutable(input_data=0.4, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_sample_inputs_tensor_heaviside(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device)
    cases = (
        # input is not tensor
        (2, make_arg((1, 2, 1))),
        # values is not tensor
        (make_arg((1, 2, 1)), 2),
    )
    for _input, _values in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=_input,
                op_args=(_values,),
                op_kwargs={},
                sample_name=f"{op_info.name}_params_error"
            ),
            op_error_type=(AttributeError, TypeError),
            op_error_info=f'error values not supported for {op_info.name}',
        )

def _get_broadcast_index(input_idx, input_shape, other_shape):

    max_len = max(len(other_shape),len(input_shape))

    padded_input_shape = [1] * (max_len - len(input_shape)) + list(input_shape)
    padded_other_shape = [1] * (max_len - len(other_shape)) + list(other_shape)

    padded_input_index = [0] * (max_len - len(input_idx)) + list(input_idx)

    other_idx=[]
    for i in range(max_len):
        if padded_other_shape[i] == 1:
            other_idx.append(0)
        elif padded_input_shape[i] == 1:
            other_idx.append(random.randint(0, padded_input_shape[i]-1))
        else:
            other_idx.append(padded_input_index[i])
    return tuple(other_idx[-len(other_shape):])

def _generate_random_index(shape,):
    random_index=[]

    if len(shape) == 0:
        return ()

    for index in range(len(shape)):
        random_index.append(random.randint(0, shape[index] - 1) if shape[index] > 1 else 0)

    return tuple(random_index)

def construnt_tensor_with_equal_values(input_tensor, other_tensor, input_revise_value, other_revise_value):
    input_shape = input_tensor.shape
    other_shape = other_tensor.shape

    if 0 in input_shape:
        return input_tensor, other_tensor

    if len(input_shape) == 0:
        input_tensor = input_tensor.fill(input_revise_value)
        if len(other_shape) != 0:
            other_tensor[_generate_random_index(other_shape)] = other_revise_value
        else:
            other_tensor = other_tensor.fill(other_revise_value)
    else:
        input_equal_index = _generate_random_index(input_shape)
        input_tensor[input_equal_index] = input_revise_value
        if len(other_shape) != 0:
            other_tensor[_get_broadcast_index(input_equal_index, input_shape, other_shape)] = other_revise_value
        else:
            other_tensor = other_tensor.fill(other_revise_value)

    return  input_tensor, other_tensor


def basic_reference_inputs_tensor_isclose(op_info: OpInfo,  dtype=None, device=None, **kwargs):
    cases = (
        (1e-05, 1e-08, 2, 2, False),
        (1e-05, 1e-08, np.nan, np.nan, False),
        (1e-05, 1e-08, np.nan, np.nan, True),
    )

    for rtol, atol, input_revise_value, other_revise_value, equal_nan in cases:

        for sample_input in basic_reference_inputs_binary_op_common_func(op_info=op_info,  dtype=dtype, device=device, **kwargs):

            if not (np.isnan(input_revise_value) and equal_nan and sample_input.op_input.shape != sample_input.op_args[0].shape):

                input_tensor, other_tensor = construnt_tensor_with_equal_values(sample_input.op_input, sample_input.op_args[0], input_revise_value, other_revise_value)
                sample_input.op_input = input_tensor
                sample_input.op_args = (other_tensor, )
                sample_input.op_kwargs = {'rtol': rtol, 'atol': atol, 'equal_nan': equal_nan}

                yield sample_input

def error_reference_inputs_tensor_isclose(op_info: OpInfo,  dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, dtype=ms.float32, device=device)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3)),
            op_args=(make_arg((2, 5)), ),
            op_kwargs={},
            sample_name=f"{op_info.name}_value_error"
        ),
        op_error_type=ValueError,
        op_error_info=f'input.shape and other.shape should equal or broadcasted',
    )

def basic_reference_inputs_tensor_isnan(op_info: OpInfo,  dtype=None, device=None, **kwargs):

    for sample_input in basic_reference_inputs_unary_op_common_func(op_info=op_info,  dtype=dtype, device=device, **kwargs):

        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = np.nan

        yield sample_input


def basic_reference_inputs_tensor_isinf(op_info: OpInfo, dtype=None, device=None, **kwargs):

    for sample_input in basic_reference_inputs_unary_op_common_func(op_info=op_info, dtype=dtype, device=device,
                                                                    **kwargs):
        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = np.inf

        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = -np.inf

        yield sample_input

def basic_reference_inputs_tensor_isneginf(op_info: OpInfo, dtype=None, device=None, **kwargs):

    for sample_input in basic_reference_inputs_unary_op_common_func(op_info=op_info, dtype=dtype, device=device,
                                                                    **kwargs):
        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = -np.inf

        yield sample_input

def basic_reference_inputs_tensor_isposinf(op_info: OpInfo, dtype=None, device=None, **kwargs):

    for sample_input in basic_reference_inputs_unary_op_common_func(op_info=op_info, dtype=dtype, device=device,
                                                                    **kwargs):
        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = np.inf

        yield sample_input

def basic_reference_inputs_tensor_isfinite(op_info: OpInfo, dtype=None, device=None, **kwargs):

    for sample_input in basic_reference_inputs_unary_op_common_func(op_info=op_info, dtype=dtype, device=device,
                                                                    **kwargs):
        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = np.inf

        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = -np.inf

        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = np.nan

        yield sample_input

def basic_reference_inputs_tensor_isreal(op_info: OpInfo, dtype=None, device=None, **kwargs):

    for sample_input in basic_reference_inputs_unary_op_common_func(op_info=op_info, dtype=dtype, device=device,
                                                                    **kwargs):
        random_index = _generate_random_index(sample_input.op_input.shape)

        sample_input.op_input[random_index] = ms.Tensor(1+0j, dtype=dtype)

        yield sample_input

def basic_sample_inputs_tensor_divide_(op_info: BinaryOpInfo, dtype=None, device=None, **kwargs):
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
        ((S, 5), (S,)),
        ((M, S), ()),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, M, S), (M, S)),
        ((M, M, S), (1, M, S)),
        ((0, M, XS), (0, M, XS)),
    )

    for input_shape, other_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low, high=op_info.input_high,
                           exclude_value=op_info.input_exclude_value)
        _other = make_func(other_shape, low=op_info.other_low, high=op_info.other_high,
                           exclude_value=op_info.other_exclude_value)
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )

def extra_sample_inputs_tensor_divide_(op_info: BinaryOpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    shapes = (
        ((2, 2, 3, 2), (2, 2, 3, 2)),
        ((2, 2, 2, 3, 2), (2, 2, 2, 3, 2)),
        ((3, 2, 2, 2, 2, 2), (3, 2, 2, 2, 2, 2)),
        ((2, 2, 2, 2, 2, 2, 3), (2, 2, 2, 2, 2, 2, 3)),
        ((2, 2, 2, 2, 2, 4, 2, 2), (2, 2, 2, 2, 2, 4, 2, 2)),
    )
    for input_shape, other_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low, high=op_info.input_high,
                           exclude_value=op_info.input_exclude_value)
        _other = make_func(other_shape, low=op_info.other_low, high=op_info.other_high,
                           exclude_value=op_info.other_exclude_value)
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )
