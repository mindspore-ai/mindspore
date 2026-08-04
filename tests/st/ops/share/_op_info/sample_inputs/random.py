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
- sample_inputs for random-types ops
"""
import functools

import numpy as np
import warnings
import itertools
import mindspore as ms
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_info import (
    basic_reference_inputs_unary_op_common_func,
    extra_reference_inputs_unary_op_common_func,
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
    dtypes_extra_values,
    small_large_values
)

def basic_sample_inputs_mint_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=(0,))
    cases = ((10, 10), (10,), ())
    # Other values will bring random number of output, skip it.
    p_vals = [0.5, 0.0, 1.0]
    # This is to handle special case for feature_alpha_dropout which has different
    # supported dtypes depending on `train` parameter
    training_vals = [True, False]
    inplaces = [True, False]
    for case, p, training, inplace in itertools.product(cases, p_vals, training_vals, inplaces):
        if not case and dtype.is_floating_point:
            continue
        sample_input = OpSampleInput(
            op_input=make_arg(case),
            op_args=(p, training, inplace),
            op_kwargs={},
            sample_name=op_info.name,
        )
        if inplace:
            sample_input.grad_position=()
        yield sample_input

def extra_sample_inputs_mint_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=(0,))
    #large shape:
    yield OpSampleInput(
        op_input=make_arg(shape=(256, 128)),
        op_args=(),
        op_kwargs={'p': 0.5, 'training': True},
        sample_name=op_info.name,
    )
    #empty shape
    yield OpSampleInput(
        op_input=make_arg(shape=(2, 0, 4)),
        op_args=(),
        op_kwargs={'p': 0.5, 'training': True},
        sample_name=op_info.name,
    )
    # ---- small/large value tensors ----
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(2, 4))),
        op_args=(),
        op_kwargs={'p': 0.5, 'training': True},
        sample_name=op_info.name,
    )
    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            yield OpSampleInput(
                op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(2, 4))),
                op_args=(),
                op_kwargs={'p': 0.5, 'training': True},
                sample_name=op_info.name,
            )

    # case fail at kbk mode while tensor is discontiguous skip...

def dynamic_sample_inputs_mint_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, exclude_value=(0,))
    #dynamic shape
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_args=(0.39423099706762, False,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 9, 7)),
                    op_args=(0.39423099706762, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 7, 8)),
                    op_args=(0.39423099706762, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(8, 7, 9)),
                    op_args=(0.39423099706762, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    #dynamic rank
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0.7625463192876073, False,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 4, 8, 9)),
                    op_args=(0.7625463192876073, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3, 5, 4, 9)),
                    op_args=(0.7625463192876073, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 8, 3)),
                    op_args=(0.7625463192876073, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
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

def basic_sample_inputs_mint_dropout2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=(0,))
    cases = ((10, 10), (3, 5, 10), (2, 3, 4, 5))
    # Other values will bring random number of output, skip it.
    p_vals = [0.5, 0.0, 1.0]
    # This is to handle special case for feature_alpha_dropout which has different
    # supported dtypes depending on `train` parameter
    training_vals = [True, False]
    inplaces = [True, False]
    for case, p, training, inplace in itertools.product(cases, p_vals, training_vals, inplaces):
        if not case and dtype.is_floating_point:
            continue
        sample_input = OpSampleInput(
            op_input=make_arg(case),
            op_args=(p, training, inplace),
            op_kwargs={},
            sample_name=op_info.name,
        )
        if inplace:
            sample_input.grad_position = ()
        yield sample_input

def extra_sample_inputs_mint_dropout2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, exclude_value=(0,))
    # large shape:
    yield OpSampleInput(
        op_input=make_arg(shape=(256, 128, 10)),
        op_args=(),
        op_kwargs={'p': 0.5, 'training': True},
        sample_name=op_info.name,
    )
    # empty shape
    yield OpSampleInput(
        op_input=make_arg(shape=(2, 0, 4)),
        op_args=(),
        op_kwargs={'p': 0.5, 'training': True},
        sample_name=op_info.name,
    )
    # ---- small/large value tensors ----
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values, size=(2, 4, 5, 6))),
        op_args=(),
        op_kwargs={'p': 0.5, 'training': True},
        sample_name=op_info.name,
    )
    # ---- extremal value tensors (inf, -inf, nan) ----
    if dtype is not None and dtype.is_floating_point:
        if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
            yield OpSampleInput(
                op_input=ms.Tensor(np.random.choice([np.inf, -np.inf, np.nan, 1024], size=(2, 4, 5, 6))),
                op_args=(),
                op_kwargs={'p': 0.5, 'training': True},
                sample_name=op_info.name,
            )

    # case fail at kbk mode while tensor is discontiguous skip...

def dynamic_sample_inputs_mint_dropout2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, exclude_value=(0,))
    # dynamic shape
    if not kwargs.get("only_dynamic_shape", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                op_args=(0.39423099706762, False,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 9, 7, 10)),
                    op_args=(0.39423099706762, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 7, 8, 10)),
                    op_args=(0.39423099706762, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(10, 8, 7, 9)),
                    op_args=(0.39423099706762, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    # dynamic rank
    if not kwargs.get("only_dynamic_rank", False):
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(0.7625463192876073, False,),
                op_kwargs={},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(8, 4, 8, 9)),
                    op_args=(0.7625463192876073, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 3, 5)),
                    op_args=(0.7625463192876073, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(7, 8)),
                    op_args=(0.7625463192876073, False,),
                    op_kwargs={},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

def error_inputs_other_op_with_dropout2d(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4)),
            op_args=(2,),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=''' 'p' must be in [0,1]  ''',
    )
    #input is not 2d/3d/4d
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, )),
            op_args=(0.5, True, True),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info=''' input must be 2D/3D/4D  ''',
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
        op_error_info=f'mean error type is not supported for {op_info.name}',
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
        op_error_info=f'std error type is not supported for {op_info.name}',
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
    # maybe CANN error
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
        ((2, 3), None, ms.float32, 'Ascend'),
        ((2, 3), None, ms.float32, 'npu'),
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
    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None,), dtype=dtype),
                op_args=(),
                op_kwargs={},
                grad_position=(),
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
                grad_position=(),
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
    )

    # size is float
    # dtype is float
    # size is negative
    # size is nan
    # device value is GPU
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
        ((3, 3), ms.float32, 'Ascend'),
        ((3, 3), ms.float32, 'npu'),
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
    )

    # dtype is float
    # device value is GPU
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
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(3,), dtype=ms.int32, device=device),
            op_args=(),
            op_kwargs={'dtype': None, 'device': None},
            sample_name=f"{op_info.name}_integral_input_without_dtype",
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info=f'integral input without float dtype is not supported for {op_info.name}',
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func((3,)),
            op_args=(),
            op_kwargs={'dtype': ms.int32, 'device': None},
            sample_name=f"{op_info.name}_integral_dtype",
        ),
        op_error_type=(TypeError, ValueError),
        op_error_info=f'integral dtype is not supported for {op_info.name}',
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
        (0, 1, (3,), None, ms.float32, 'Ascend'),
        (0, 1, (3,), None, ms.float32, 'npu'),
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
    )

    # low is float
    # high is float
    # low is larger than high
    # dtype is float
    # size is int
    # size is neg value
    # device value is GPU
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
        ((2, 3, 8 ,2), 0, 1, ms.float32, 'Ascend'),
        ((2, 3, 8 ,2), 0, 1, ms.float32, 'npu'),
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
        ((4, 4, 8, 5, 5, 9, 8, 7), 12, 50, ms.float32, 'Ascend'),
        ((4, 4, 8, 5, 5, 9, 8, 7), 12, 50, ms.float32, 'npu'),
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
    )

    # low isnot int
    # high isnot int
    # dtype is float
    # low is larger than high
    # device value is GPU
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
        (0, None, dtype),
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
        (1.0, ms.int64),
        (1, 1.0),
        (ms.Tensor(np.random.randint(1, 5, (1, 2))), ms.int64),
        (ms.Tensor(np.random.randint(1, 5, (2, 0))), ms.int64),
    )

    # n is float
    # dtype is float
    # n is 2d
    # n is empty tensor
    for n, case_dtype in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=n,
                op_args=(),
                op_kwargs={'dtype': case_dtype},
                sample_name=f"{op_info.name}_params_value_error",
            ),
            op_error_type=(TypeError, ValueError),
            op_error_info=f'params error values is not supported for {op_info.name}',
        )

def dynamic_sample_inputs_mint_bernoulli(op_info: OpInfo, dtype, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=1)
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

def error_sample_inputs_mint_bernoulli(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device, low=2)
    # the element values in input is not in [0, 1]
    # when the element is negative or not in [0, 1], multiple cases are executed together,
    # it may cause program exceptions that affect the execution of other cases.
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_func(shape=(1, 2)),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_element_greater_than_1",
        ),
        op_error_type=RuntimeError,
        op_error_info=f'the element values in input is not in [0, 1] is not supported by {op_info.name}.',
    )

def basic_sample_inputs_tensor_bernoulli_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=1)
    cases = (
        (),
        (1, 1),
    )
    if dtype.is_floating_point:
        p_dtype = dtype
    else:
        p_dtype = ms.float32
    for sample_input in basic_reference_inputs_unary_op_common_func(op_info, dtype=dtype, device=device, **kwargs):
        for p_shape in cases:
            if len(sample_input.op_input.shape) >= len(p_shape):
                p = make_arg(p_shape, dtype=p_dtype)
            else:
                p = 0.9
            sample_input.op_kwargs.update({'p': p})
            yield sample_input

def extra_sample_inputs_tensor_bernoulli_(op_info: OpInfo, dtype=None, device=None, **kwargs):
    yield from extra_reference_inputs_unary_op_common_func(op_info, dtype, device, **kwargs)
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    cases = (
        ((2, 3, 4), (1, 1, 1)),
        ((2, 3, 4, 5), (2, 3, 4, 5)),
        ((2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 6, 7)),
        ((2, 3, 4, 5, 6, 7, 8, 9), (2, 3, 4, 5, 6, 7, 8, 9)),
    )
    if dtype.is_floating_point:
        p_dtype = dtype
    else:
        p_dtype = ms.float32
    for input_shape, p_shape in cases:
        yield OpSampleInput(
            op_input=make_func(shape=input_shape),
            op_args=(),
            op_kwargs={'p': make_func(shape=p_shape, dtype=p_dtype, low=0, high=1)},
            sample_name=f"{op_info.name}_{dtype}_{len(input_shape)}d_extra",
        )
    yield OpSampleInput(
        op_input=make_func(shape=(10, 13, 11)),
        op_args=(),
        op_kwargs={'p': 0.00001},
        sample_name=f"{op_info.name}_{dtype}_p_float_extra",
    )

def dynamic_sample_inputs_tensor_bernoulli_(op_info: OpInfo, dtype, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
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
                    op_input=make_func(shape=(1, 4, 5)),
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
                    op_input=make_func(shape=(3, 4, 1)),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

def error_sample_inputs_tensor_bernoulli_(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=ms.float32, device=device, low=2)
    make_args = functools.partial(make_tensor, dtype=dtype, device=device, low=0, high=1)
    cases = (
        # input is not tensor
        (2, 0.5),
        # p is float but not in [0, 1]
        (make_func((1, 2)), 1.1),
        # p is not float
        (make_func((1, 2)), (0, 1.1)),
        # the element values in p is not in [0, 1]
        # when the element in p is negative or not in [0, 1], multiple cases are executed together,
        # it may cause program exceptions that affect the execution of other cases.
        # (make_func((1, 2)), make_func((1, 2))),
    )
    for case_input, p in cases:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=case_input,
                op_args=(),
                op_kwargs={'p': p},
                sample_name=f"{op_info.name}_error_param",
            ),
            op_error_type=(AttributeError, RuntimeError, TypeError),
            op_error_info=f'error values of input or p is not supported by {op_info.name}.',
        )
    # p does not support bool/uint/int/complex when p is Tensor
    tensor_p_not_support_dtypes = (
        ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.complex64, ms.complex128
    )
    for p_dtype in tensor_p_not_support_dtypes:
        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_func(shape=(1, 3, 2)),
                op_args=(),
                op_kwargs={'p': make_args(shape=(3, 2), dtype=p_dtype)},
                sample_name=f"{op_info.name}_tensor_p_dtype_{p_dtype}_error",
            ),
            op_error_type=RuntimeError,
            op_error_info=f'tensor p error dtype {p_dtype} is not supported for {op_info.name}',
        )
