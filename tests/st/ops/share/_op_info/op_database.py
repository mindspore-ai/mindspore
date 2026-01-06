# Copyright 2025 Huawei Technologies Co., Ltd
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
- Sample input builders for add/sub ops (including alpha cases).
- Dynamic-shape input builders for rank/shape dynamism.
- Gradient wrappers for ops with kwargs.
- The operator database (op_db) and get_op_info accessor.
"""
import functools
import numpy as np
from typing import Dict, Optional
import torch
import warnings
import itertools
import mindspore as ms
from mindspore import mint, mutable
from mindspore.common.parameter import Parameter
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo, ReductionOpInfo, UnaryOpInfo
from tests.st.ops.share._op_info.op_info import (
    _generate_unary_op_extremal_value_tensor_inputs_func,
    _generate_unary_op_large_value_tensor_inputs_func,
    _generate_unary_op_small_value_tensor_inputs_func,
    _generate_unary_op_tensors_sample_inputs_func,
    basic_reference_inputs_binary_op_common_func,
    basic_reference_inputs_reduction_op_common_func,
    dynamic_inputs_binary_op_common_func,
    extra_reference_inputs_binary_op_common_func,
    extra_reference_inputs_reduction_op_common_func,
)
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, wrap_sample_inputs, make_tensor_with_np_array
)
from tests.st.ops.share._op_info.op_common import  (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_as_torch,
    dtypes_integral,
    dtypes_extra_uint,
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
def error_inputs_add_sub_ext_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for add/sub_ext ops.
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
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='other is not tensor or number',
    )


def basic_sample_inputs_atan2_ext(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate sample inputs for atan2 ops including extra alpha cases.
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
            op_kwargs={},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={},
            sample_name=op_info.name
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


# NOTE: customize_op_func, used by gradient comparison only when there are kwargs in op!
def add_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.add(x, y, alpha=alpha)

def sub_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.sub(x, y, alpha=alpha)

def div_func_grad(op_input, other):
    return mint.div(op_input, other)

def repeat_interleave_func_grad(input_x, repeats, dim=None, output_size=None):
    return mint.repeat_interleave(input_x, repeats, dim, output_size=output_size)

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

def sample_inputs_binary_cross_entropy(op_info, dtype, device, logits=False, **kwargs):
    S = SMALL_DIM_SIZE

    make = functools.partial(make_tensor, device=device, dtype=dtype)
    # Lower bounds must be greater than 'eps' defined in gradcheck.py::gradgradcheck() -> eps
    # otherwise perturbation calculation causes Tensor value to become negative triggering
    # a device-side hardware assertion
    make_prob = functools.partial(make, low=1e-6, high=1)

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

    for shape, kwargs in shapes_and_kwargs:
        yield OpSampleInput(
            (make if logits else make_prob)(shape),
            op_args=(make_prob(shape),),
            op_kwargs=kwargs,
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

    for shape, kwargs in shapes_and_kwargs:
        yield OpSampleInput(
            make(shape),
            op_args=(make_prob(shape),),
            op_kwargs=kwargs,
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

    for shape, kwargs in shapes_and_kwargs:
        yield OpSampleInput(
            _make_tensor(shape), op_args=(_make_tensor(shape),), op_kwargs=kwargs, sample_name=op_info.name
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
    input_shape = (5, 9)

    yield OpSampleInput(
        make_func(input_shape),
        op_args=(),
        sample_name=op_info.name,
    )

def sample_inputs_batchnorm2d(op_info, dtype, device, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    input_shape = (8, 3, 5, 4)

    yield OpSampleInput(
        make_func(input_shape),
        op_args=(),
        sample_name=op_info.name,
    )

def sample_inputs_batchnorm3d(op_info, dtype, device, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)
    input_shape = (8, 3, 4, 8, 5)

    yield OpSampleInput(
        make_func(input_shape),
        op_args=(),
        sample_name=op_info.name,
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

    for input_shape, kwargs in cases:
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
            op_kwargs=kwargs,
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

# wrap tensor method for astype
def tensor_astype_ms(op_input, dtype=ms.float32, copy=False):
    return op_input.astype(dtype=dtype, copy=copy)

def tensor_astype_torch(op_input, other=torch.tensor(1.0, dtype=torch.float32)):
    return op_input.type_as(other=other)

# wrap tensor method for byte
def tensor_byte_ms(op_input):
    return op_input.byte()

def tensor_byte_torch(op_input):
    return op_input.byte()

# wrap tensor method for clone
def tensor_clone_ms(op_input):
    return op_input.clone()

def tensor_clone_torch(op_input):
    return op_input.clone()

# wrap tensor method for contiguous
def tensor_contiguous_ms(op_input):
    return op_input.contiguous()

def tensor_contiguous_torch(op_input):
    return op_input.contiguous()

# wrap tensor method for is_contiguous
def tensor_is_contiguous_ms(op_input):
    return op_input.is_contiguous()

def tensor_is_contiguous_torch(op_input):
    return op_input.is_contiguous()

# wrap tensor method for matmul
def tensor_matmul_ms(op_input, x):
    return op_input.matmul(x)

def tensor_matmul_torch(op_input, x):
    return op_input.matmul(x)

# wrap tensor method for numpy
def tensor_numpy_ms(op_input):
    return op_input.numpy()

def tensor_numpy_torch(op_input):
    return op_input.numpy()

# wrap tensor method for tanh
def tensor_tanh_ms(op_input):
    return op_input.tanh()

def tensor_ceil_ms(op_input):
    return op_input.ceil()

def tensor_exp_ms(op_input):
    return op_input.exp()

def tensor_log_ms(op_input):
    return op_input.log()

def tensor_neg_ms(op_input):
    return op_input.neg()

def tensor_sigmoid_ms(op_input):
    return op_input.sigmoid()

def tensor_sqrt_ms(op_input):
    return op_input.sqrt()

def tensor_square_ms(op_input):
    return op_input.square()

def tensor_select_ms(op_input, dim, index):
    return op_input.select(dim, index)

def tensor_floor_ms(op_input):
    return op_input.floor()

def tensor_abs_ms(op_input):
    return op_input.abs()

def tensor_floor_divide_ms(op_input, other):
    return op_input.floor_divide(other)

def tensor_tanh_torch(op_input):
    return op_input.tanh()

def tensor_eq_ms(op_input, x):
    return op_input.eq(x)

def tensor_repeat_interleave_ms(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim, output_size=output_size)

def tensor_repeat_interleave_torch(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim=dim, output_size=output_size)

def tensor_repeat_interleave_func_grad(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim, output_size=output_size)

def nn_functional_selu_ms(op_input):
    return mint.nn.functional.selu(op_input)

def nn_functional_selu_torch(op_input):
    return torch.nn.functional.selu(op_input)

def tensor_add__ms(op_input, other, alpha=1):
    """Wrapper for Tensor.add_ (in-place operation)."""
    #TODO:The is_inplace_op item of OpInfo still has problem. Should be fixed in the future.
    cloned_input = op_input.clone()
    result = cloned_input.add_(other, alpha=alpha)
    return result

def tensor_add__torch(op_input, other, alpha=1):
    """Wrapper for torch.Tensor.add_ (in-place operation)."""
    cloned_input_pt = op_input.clone()
    result = cloned_input_pt.add_(other, alpha=alpha)
    return result

def tensor_masked_scatter_ms(op_input, mask, source):
    return op_input.masked_scatter(mask, source)

def tensor_masked_scatter_torch(op_input, mask, source):
    return op_input.masked_scatter(mask, source)

def tensor_masked_scatter__ms(op_input, mask, source):
    """Wrapper for Tensor.masked_scatter_ (in-place operation)."""
    # For in-place operations, we need to clone the input to avoid modifying the original
    # in the test framework, but the actual operation modifies in-place
    cloned_input = op_input.clone()
    result = cloned_input.masked_scatter_(mask, source)
    return result

def tensor_masked_scatter__torch(op_input, mask, source):
    """Wrapper for torch.Tensor.masked_scatter_ (in-place operation)."""
    cloned_input = op_input.clone()
    result = cloned_input.masked_scatter_(mask, source)
    return result

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


# wrapper function for mindspore.mint.nn.Conv1d
def mint_nn_conv1d_ms(op_input, in_channels, out_channels, kernel_size, stride=1, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None,
                      weight_init=None, bias_init=None):
    """Wrapper for mindspore.mint.nn.Conv1d."""
    conv1d_layer = mint.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    conv1d_layer.weight = Parameter(weight_init, name='weight')
    if bias:
        conv1d_layer.bias = Parameter(bias_init, name='bias')
    return conv1d_layer(op_input)


def nn_conv1d_torch(op_input, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                         dtype=None, weight_init=None, bias_init=None):
    """Wrapper for torch.nn.Conv1d."""
    conv1d_layer = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    conv1d_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        conv1d_layer.bias = torch.nn.Parameter(bias_init)
    return conv1d_layer(op_input)


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
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, groups, dilation)
    cases = (
        ((1, 3, 4, 4), (3, 3, 3, 3), (3,),
            {'stride': (2, 2), 'padding': 2, 'groups': 1}),
        ((1, 2, 4, 3), (4, 2, 3, 4), None,
            {'stride': 2, 'padding': 1, 'groups': 1}),
        ((12, 6, 4, 6), (8, 3, 4, 7), (8,),
            {'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 2}),
        ((1, 4, 5, 5), (3, 4, 3, 3), None, {}),
    )

    for input_shape, weight_shape, bias_shape, case_kwargs in cases:
        # Create kwargs dict with proper order: bias, stride, padding, dilation, groups
        # This matches the interface: conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
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
        op_args=(0, 0),
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
        op_args=(1, 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_2d_repeat_1x2",
    )

    # 2D (S, S) with repeat(2, 1) -> (2S, S)
    yield OpSampleInput(
        op_input=make_x((S, S)),
        op_args=(2, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_2d_repeat_2x1",
    )

    # 3D (2, 1, S) with repeat(2, 3, 1) -> (4, 3, S)
    yield OpSampleInput(
        op_input=make_x((2, 1, S)),
        op_args=(2, 3, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_3d_repeat_2x3x1",
    )

def tensor_repeat_ms(op_input, *sizes):
    return op_input.repeat(*sizes)

def tensor_repeat_torch(op_input, *sizes):
    return op_input.repeat(*sizes)

def tensor_eq_torch(op_input, x):
    return op_input.eq(x)

def tensor_greater_equal_ms(op_input, x):
    return op_input.greater_equal(x)

def tensor_greater_equal_torch(op_input, x):
    return op_input.greater_equal(x)

def tensor_greater_ms(op_input, x):
    return op_input.greater(x)

def tensor_greater_torch(op_input, x):
    return op_input.greater(x)

def tensor_less_equal_ms(op_input, x):
    return op_input.less_equal(x)

def tensor_less_equal_torch(op_input, x):
    return op_input.less_equal(x)

def tensor_less_ms(op_input, x):
    return op_input.less(x)

def tensor_less_torch(op_input, x):
    return op_input.less(x)

def tensor_ne_ms(op_input, x):
    return op_input.ne(x)

def tensor_ne_torch(op_input, x):
    return op_input.ne(x)

def tensor_atan2_ms(op_input, x):
    return op_input.atan2(x)

def tensor_atan2_torch(op_input, x):
    return op_input.atan2(x)

def tensor_gt_ms(op_input, x):
    return op_input.gt(x)

def tensor_gt_torch(op_input, x):
    return op_input.gt(x)

def tensor_le_ms(op_input, x):
    return op_input.le(x)

def tensor_le_torch(op_input, x):
    return op_input.le(x)

def tensor_lt_ms(op_input, x):
    return op_input.lt(x)

def tensor_lt_torch(op_input, x):
    return op_input.lt(x)

def tensor_maximum_ms(op_input, other):
    return op_input.maximum(other)

def tensor_maximum_torch(op_input, other):
    return op_input.maximum(other)

def tensor_minimum_ms(op_input, other):
    return op_input.minimum(other)

def tensor_minimum_torch(op_input, other):
    return op_input.minimum(other)

def tensor_mul_ms(op_input, other):
    return op_input.mul(other)

def tensor_mul_torch(op_input, other):
    return op_input.mul(other)

def tensor_ceil_torch(op_input):
    return op_input.ceil()

def tensor_exp_torch(op_input):
    return op_input.exp()

def tensor_log_torch(op_input):
    return op_input.log()

def tensor_neg_torch(op_input):
    return op_input.neg()

def tensor_select_torch(op_input, dim, index):
    return op_input.select(dim, index)

def tensor_sigmoid_torch(op_input):
    return op_input.sigmoid()

def tensor_sqrt_torch(op_input):
    return op_input.sqrt()

def tensor_square_torch(op_input):
    return op_input.square()

def tensor_floor_torch(op_input):
    return op_input.floor()

def tensor_abs_torch(op_input):
    return op_input.abs()

def tensor_floor_divide_torch(op_input, other):
    return op_input.floor_divide(other)

# wrap tensor method for to
def tensor_to_ms(op_input, dtype=ms.uint8):
    return op_input.to(dtype=dtype)

def tensor_to_torch(op_input, dtype=torch.uint8):
    return op_input.to(dtype=dtype)

# wrap method for empty
def empty_ms(op_input):
    return mint.empty(op_input.shape, dtype=op_input.dtype).shape

def empty_torch(op_input):
    return torch.empty(op_input.shape, dtype=op_input.dtype).shape

# wrap method for empty_like
def empty_like_ms(op_input):
    return mint.empty_like(op_input).shape

def empty_like_torch(op_input):
    return torch.empty_like(op_input).shape

# wrap nn method for batchnorm1d
def nn_batchnorm1d_ms(op_input):
    op = mint.nn.BatchNorm1d(
        num_features=9, eps=2.0, momentum=-2.0, affine=True, track_running_stats=False, dtype=ms.float32
    )
    op.set_train()
    return op(op_input)

def nn_batchnorm1d_torch(op_input):
    return torch.nn.BatchNorm1d(
        num_features=9, eps=2.0, momentum=-2.0, affine=True, track_running_stats=False, dtype=torch.float32
    )(op_input)

# wrap nn method for batchnorm2d
def nn_batchnorm2d_ms(op_input):
    op = mint.nn.BatchNorm2d(
        num_features=3, eps=565.73012560178, momentum=-2.0, affine=False, track_running_stats=True, dtype=ms.float32
    )
    op.set_train()
    return op(op_input)

def nn_batchnorm2d_torch(op_input):
    return torch.nn.BatchNorm2d(
        num_features=3, eps=565.73012560178, momentum=-2.0, affine=False, track_running_stats=True, dtype=torch.float32
    )(op_input)

# wrap nn method for batchnorm3d
def nn_batchnorm3d_ms(op_input):
    op = mint.nn.BatchNorm3d(
        num_features=3, eps=121.30250066286163, momentum=-2.0, affine=False, track_running_stats=True, dtype=ms.float32
    )
    op.set_train()
    return op(op_input)

def nn_batchnorm3d_torch(op_input):
    return torch.nn.BatchNorm3d(
        num_features=3,
        eps=121.30250066286163,
        momentum=-2.0,
        affine=False,
        track_running_stats=True,
        dtype=torch.float32,
    )(op_input)

# wrap nn method for gelu
def nn_gelu_ms(op_input):
    return mint.nn.GELU()(op_input)

def nn_gelu_torch(op_input):
    return torch.nn.GELU()(op_input)

# wrap nn method for glu
def nn_glu_ms(op_input):
    return mint.nn.GLU()(op_input)

def nn_glu_torch(op_input):
    return torch.nn.GLU()(op_input)

# wrap nn method for identity
def nn_identity_ms(op_input):
    return mint.nn.Identity()(op_input)

def nn_identity_torch(op_input):
    return torch.nn.Identity()(op_input)

# wrap nn method for logsigmoid
def nn_logsigmoid_ms(op_input):
    return mint.nn.LogSigmoid()(op_input)

def nn_logsigmoid_torch(op_input):
    return torch.nn.LogSigmoid()(op_input)

# wrap nn method for prelu
def nn_prelu_ms(op_input):
    return mint.nn.PReLU(dtype=op_input.dtype)(op_input)

def nn_prelu_torch(op_input):
    return torch.nn.PReLU(dtype=op_input.dtype)(op_input)

# wrap nn method for relu
def nn_relu_ms(op_input):
    return mint.nn.ReLU()(op_input)

def nn_relu_torch(op_input):
    return torch.nn.ReLU()(op_input)

# wrap nn method for relu6
def nn_relu6_ms(op_input):
    return mint.nn.ReLU6()(op_input)

def nn_relu6_torch(op_input):
    return torch.nn.ReLU6()(op_input)

def tensor_tile_ms(op_input, dims):
    return op_input.tile(dims)

def tensor_tile_torch(op_input, dims):
    return op_input.tile(dims)

def tensor_reciprocal_ms(op_input):
    return op_input.reciprocal()

def tensor_reciprocal_torch(op_input):
    return op_input.reciprocal()

def tensor_unsqueeze_ms(op_input, *args, **kwargs):
    return op_input.unsqueeze(*args, **kwargs)

def tensor_unsqueeze_torch(op_input, *args, **kwargs):
    return op_input.unsqueeze(*args, **kwargs)

def tensor_topk_ms(op_input, *args, **kwargs):
    return op_input.topk(*args, **kwargs)

def tensor_topk_torch(op_input, *args, **kwargs):
    return op_input.topk(*args, **kwargs)

def tensor_transpose_ms(op_input, *args, **kwargs):
    return op_input.transpose(*args, **kwargs)

def tensor_transpose_torch(op_input, *args, **kwargs):
    return op_input.transpose(*args, **kwargs)

# wrap nn method for tanh
def nn_tanh_ms(op_input):
    return mint.nn.Tanh()(op_input)

def nn_tanh_torch(op_input):
    return torch.nn.Tanh()(op_input)

# wrap tensor method for expand_as
def tensor_expand_as_ms(op_input, other):
    return op_input.expand_as(other)

def tensor_expand_as_torch(op_input, other):
    return op_input.expand_as(other)

# wrap tensor method for sin
def tensor_sin_ms(op_input):
    return op_input.sin()

def tensor_sin_torch(op_input):
    return op_input.sin()

# wrap tensor method for add
def tensor_add_ms(op_input, args):
    return op_input.add(args)

def tensor_add_torch(op_input, args):
    return op_input.add(args)

# wrap tensor method for sub
def tensor_sub_ms(op_input, args):
    return op_input.sub(args)

def tensor_sub_torch(op_input, args):
    return op_input.sub(args)

# wrap tensor method for logical_and
def tensor_logical_and_ms(op_input, args):
    return op_input.logical_and(args)

def tensor_logical_and_torch(op_input, args):
    return op_input.logical_and(args)

# wrap tensor method for logical_not
def tensor_logical_not_ms(op_input):
    return op_input.logical_not()

def tensor_logical_not_torch(op_input):
    return op_input.logical_not()

# wrap tensor method for bfloat16
def tensor_bfloat16_ms(op_input):
    return op_input.bfloat16()

def tensor_bfloat16_torch(op_input):
    return op_input.bfloat16()

# wrap tensor method for bool
def tensor_bool_ms(op_input):
    return op_input.bool()

def tensor_bool_torch(op_input):
    return op_input.bool()

# wrap tensor method for double
def tensor_double_ms(op_input):
    return op_input.double()

def tensor_double_torch(op_input):
    return op_input.double()

# wrap tensor method for float
def tensor_float_ms(op_input):
    return op_input.float()

def tensor_float_torch(op_input):
    return op_input.float()

# wrap tensor method for half
def tensor_half_ms(op_input):
    return op_input.half()

def tensor_half_torch(op_input):
    return op_input.half()

# wrap tensor method for int
def tensor_int_ms(op_input):
    return op_input.int()

def tensor_int_torch(op_input):
    return op_input.int()

# wrap tensor method for long
def tensor_long_ms(op_input):
    return op_input.long()

def tensor_long_torch(op_input):
    return op_input.long()

# wrap tensor method for logical_or
def tensor_logical_or_ms(op_input, args):
    return op_input.logical_or(args)

def tensor_logical_or_torch(op_input, args):
    return op_input.logical_or(args)

# wrap tensor method for split
def tensor_split_ms(op_input, args):
    return op_input.split(args)

def tensor_split_torch(op_input, args):
    return op_input.split(args)

# wrap tensor method for index_select
def tensor_index_select_ms(op_input, axis, index):
    return op_input.index_select(axis, index)

def tensor_index_select_torch(op_input, axis, index):
    return op_input.index_select(axis, index)

# wrap tensor method for cos
def tensor_cos_ms(op_input):
    return op_input.cos()

def tensor_cos_torch(op_input):
    return op_input.cos()

def nn_group_norm_ms(op_input, num_groups, num_channels, eps=1e-5, affine=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = mint.nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    return module(op_input)

def nn_group_norm_torch(op_input, num_groups, num_channels, eps=1e-5, affine=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = torch.nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    return module(op_input)

def nn_layer_norm_ms(op_input, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = mint.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, dtype=dtype)
    return module(op_input)

def nn_layer_norm_torch(op_input, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = torch.nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, dtype=dtype)
    return module(op_input)

# wrap nn method for AdaptiveAvgPool2d
def nn_adaptiveavgpool2d_ms(op_input, output_size):
    return mint.nn.AdaptiveAvgPool2d(output_size)(op_input)

def nn_adaptiveavgpool2d_torch(op_input, output_size):
    return torch.nn.AdaptiveAvgPool2d(output_size)(op_input)

# wrap nn method for AdaptiveAvgPool3d
def nn_adaptiveavgpool3d_ms(op_input, output_size):
    return mint.nn.AdaptiveAvgPool3d(output_size)(op_input)

def nn_adaptiveavgpool3d_torch(op_input, output_size):
    return torch.nn.AdaptiveAvgPool3d(output_size)(op_input)

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


# wrap tensor method for gather
def tensor_gather_ms(input, dim, index):
    return input.gather(dim, index)

def tensor_gather_overload_ms(input, input_indices, axis):
    return input.gather(input_indices, axis)

def tensor_gather_torch(input, dim, index):
    return input.gather(dim, index)

# wrap tensor method for unique
def tensor_unique_ms(op_input, *op_args, **op_kwargs):
    return op_input.unique(*op_args, **op_kwargs)

def tensor_unique_torch(op_input, *op_args, **op_kwargs):
    return op_input.unique(*op_args, **op_kwargs)

# wrap tensor method for clamp
def tensor_clamp_ms(op_input, *op_args, **op_kwargs):
    return op_input.clamp(*op_args, **op_kwargs)

def tensor_clamp_torch(op_input, *op_args, **op_kwargs):
    return op_input.clamp(*op_args, **op_kwargs)

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
                zero_np_dtype = {
                    ms.bool_: np.bool_,
                    ms.int8: np.int8,
                    ms.int16: np.int16,
                    ms.int32: np.int32,
                    ms.int64: np.int64,
                    ms.uint8: np.uint8,
                    ms.float16: np.float16,
                    ms.float32: np.float32,
                    ms.float64: np.float64,
                    ms.bfloat16: np.float32,  # construct in fp32 then to bfloat16 by dtype=
                    ms.complex64: np.complex64,
                    ms.complex128: np.complex128,
                }[dtype]
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

    # Use a valid dim for 1D inputs: dim can be 0 or -1
    for dim in (0, -1):
        yield OpSampleInput(
            op_input=tensors,
            op_args=(dim,),
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

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-1, high=1)

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
                            "size": size_tuple,
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
                                "size": size_tuple,
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
                                "size": size_tuple,
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
                        "size": size_tuple,
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
                            "size": size_tuple,
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
                            "size": size_tuple,
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

        yield OpSampleInput(
            _input,
            op_args=(),
            sample_name=op_info.name,
        )



def tensor_sum_ms(op_input, *op_args, **op_kwargs):
    return op_input.sum(*op_args, **op_kwargs)

def tensor_sum_torch(op_input, *op_args, **op_kwargs):
    return op_input.sum(*op_args, **op_kwargs)

def tensor_mean_ms(op_input, *op_args, **op_kwargs):
    return op_input.mean(*op_args, **op_kwargs)

def tensor_mean_torch(op_input, *op_args, **op_kwargs):
    return op_input.mean(*op_args, **op_kwargs)

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


def basic_sample_inputs_mint_pow(op_info: OpInfo, dtype=None, device=None, **kwargs):
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
        _input = make_func(input_shape, low=op_info.input_low, high=op_info.input_high, random_method='randn')
        _other = make_func(other_shape, low=op_info.other_low, high=op_info.other_high, random_method='randn')

        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_pow(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    shapes = (
        ((2, 2, 3, 2), (2, 2, 3, 2)),
        ((2, 2, 2, 3, 2), (2, 2, 2, 3, 2)),
        ((3, 2, 2, 2, 2, 2), (3, 2, 2, 2, 2, 2)),
        ((2, 2, 2, 2, 2, 2, 3), (2, 2, 2, 2, 2, 2, 3)),
        ((2, 2, 2, 2, 2, 4, 2, 2), (2, 2, 2, 2, 2, 4, 2, 2)),
    )
    for input_shape, other_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low, high=op_info.input_high, random_method='randn')
        _other = make_func(other_shape, low=op_info.other_low, high=op_info.other_high, random_method='randn')

        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )


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


def basic_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    shapes = ((S, S, S), (S, S), (S,), (),)
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


def extra_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # shape x start_dim x end_dim
    cases = (
        ((5, 4, 0, 1, 3, 7), 1, 3),
        ((5, 4, 0, 1, 3, 7), 4, 5),
        ((5, 4, 1, 1, 3, 7), 2, 3),
        ((), 0, -1),
        ((1,), 0, -1),
        ((3, 7, 5), 1, 2),
        ((4, 5), 1, 1),
        ((1, 5, 5, 1, 5, 1, 5, 1), 0, 2),
        ((1, 5, 5, 1, 5, 1, 5, 1), 3, -1),
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


def basic_sample_inputs_mint_dropout(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE

    cases = ((S, S), (S,), ())
    # Other values will bring random number of output, skip it.
    p_vals = [0.0, 1.0]
    # This is to handle special case for feature_alpha_dropout which has different
    # supported dtypes depending on `train` parameter
    training_vals = [True, False]

    for case, p, training in itertools.product(cases, p_vals, training_vals):
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
    )
    for shape, dim in cases:
        make_index = functools.partial(make_tensor, device=device, dtype=ms.int32, low=0, high=shape[dim]-1)
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(dim, make_index((shape[dim]-1,))),
            sample_name=op_info.name,
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
            op_kwargs={'split_size_or_sections': args},
            sample_name=op_info.name,
        )

# sample inputs functions for chunk
def basic_sample_inputs_tensor_split(op_info: OpInfo, dtype=None, device=None):
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
            op_args=(args, ),
            sample_name=op_info.name,
        )


def _is_ascend910(device):
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        return ascend_name == 'ascend910'
    return False

# Todo: mint.sum scalar tensor test case runtime error, need to fix.
def basic_sample_inputs_mint_sum(op_info: OpInfo, dtype=None, device=None, **kwargs):
    skip_zero_dim = _is_ascend910(device)
    for sample_input in basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs):
        if skip_zero_dim:
            op_input = sample_input.op_input
            if isinstance(op_input, ms.Tensor) and op_input.shape == ():
                continue
        yield sample_input


def basic_sample_inputs_tile(op_info, dtype, device=None, **kwargs):
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
            sample_name=f"shape{shape}_dims{dims}"
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
    for input_shape_1, input_shape_2, kwargs in cases:
        yield OpSampleInput(
            op_input=tuple([make_arg(input_shape_1), make_arg(input_shape_2)]),
            op_kwargs=kwargs,
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
    for input_shape_1, input_shape_2, kwargs in cases:
        yield OpSampleInput(
            op_input=tuple([make_arg(input_shape_1), make_arg(input_shape_2)]),
            op_kwargs=kwargs,
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


def basic_sample_inputs_mint_topk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE
    M = MEDIUM_DIM_SIZE

    def get_tensor_input(size):
        inputs_x = make_tensor(size, dtype=dtype, device=device, random_method='randn')
        # get no repeat value
        input_sum = 1
        for num in inputs_x.shape:
            input_sum *= num
        input_np = np.arange(0, input_sum)
        np.random.shuffle(input_np)
        return ms.tensor(input_np.reshape(*inputs_x.shape))

    yield OpSampleInput(get_tensor_input((S, )), op_args=(1, 0, True, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, ))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, 1, True, True))
    yield OpSampleInput(get_tensor_input((S, M, S)), op_args=(3, -2, True, True))


def basic_sample_inputs_mint_normal(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    samples = (
        ((3, ), (1, )),
    )
    for mean_shape, std_shape in samples:
        yield OpSampleInput(make_arg(mean_shape, random_method='randn'),
                            op_args=(make_arg(std_shape, high=100, low=0, random_method='uniform'), ))

    yield OpSampleInput(-2.0, op_args=(make_arg((4, ), high=100, low=0, random_method='uniform'), ))

    yield OpSampleInput(make_arg((9, ), random_method='randn'), op_args=(312.1199124773063, ))

    std = 929.6270736417483
    size = (2, 5, 13)
    yield OpSampleInput(-2.0, op_args=(std, size))


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
    yield OpSampleInput(make_arg((1, 3, 9, 9)), op_args=((3, 3)), op_kwargs={}, sample_name=op_info.name)


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


# op database
op_db: Dict[str, OpInfo] = {
    'Tensor.astype': UnaryOpInfo(
        name='Tensor.astype',
        op=tensor_astype_ms,
        ref=tensor_astype_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.byte': UnaryOpInfo(
        name='Tensor.byte',
        op=tensor_byte_ms,
        ref=tensor_byte_torch,
        # int32, int64, float32, float64, complex has precision issue
        dtypes_ascend=(ms.bool_, ms.int8, ms.int16, ms.uint8, ms.float16),
        # float64, complex has precision issue
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.clone': UnaryOpInfo(
        name='Tensor.clone',
        op=tensor_clone_ms,
        ref=tensor_clone_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.contiguous': UnaryOpInfo(
        name='Tensor.contiguous',
        op=tensor_contiguous_ms,
        ref=tensor_contiguous_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    # supported only in pynative mode
    'Tensor.is_contiguous': UnaryOpInfo(
        name='Tensor.is_contiguous',
        op=tensor_is_contiguous_ms,
        ref=tensor_is_contiguous_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
        op_extra_reference_inputs_func=extra_sample_inputs_tensor_is_contiguous_func,
    ),
    'Tensor.matmul': OpInfo(
        name='Tensor.matmul',
        op=tensor_matmul_ms,
        ref=tensor_matmul_torch,
        # 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend=(ms.float16,),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_matmul,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    # supported only in pynative mode
    'Tensor.numpy': UnaryOpInfo(
        name='Tensor.numpy',
        op=tensor_numpy_ms,
        ref=tensor_numpy_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.to': UnaryOpInfo(
        name='Tensor.to',
        op=tensor_to_ms,
        ref=tensor_to_torch,
        # int32, int64, float32, float64 and complex have precision issue
        dtypes_ascend=(ms.bool_, ms.int8, ms.int16, ms.uint8, ms.float16),
        # float64 and complex have precision issue
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'mint.add': BinaryOpInfo(
        name='mint.add',
        op=mint.add,
        customize_op_func=add_ext_func_grad_without_kwargs,
        ref=torch.add,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        op_basic_reference_inputs_func=basic_sample_inputs_add_sub_ext,
        op_dynamic_inputs_func=dynamic_sample_inputs_add_sub_ext,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
    ),
    'mint.broadcast_to': OpInfo(
        name='mint.broadcast_to',
        op=mint.broadcast_to,
        ref=torch.broadcast_to,
        # bfloat16 has precision issue
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch if (not d.is_complex and d != ms.int16 and d != ms.bfloat16 and d != ms.float64)
        ),
        # bfloat16 has precision issue
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch if (not d.is_complex and d != ms.int16 and d != ms.bfloat16 and d != ms.float64)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_broadcast_to,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.clone': UnaryOpInfo(
        name='mint.clone',
        op=mint.clone,
        ref=torch.clone,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.empty': UnaryOpInfo(
        name='mint.empty',
        op=empty_ms,
        ref=empty_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'mint.empty_like': UnaryOpInfo(
        name='mint.empty_like',
        op=empty_like_ms,
        ref=empty_like_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'mint.matmul': OpInfo(
        name='mint.matmul',
        op=mint.matmul,
        ref=torch.matmul,
        # 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend=(ms.float16,),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_matmul,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.BatchNorm1d': OpInfo(
        name='mint.nn.BatchNorm1d',
        op=nn_batchnorm1d_ms,
        ref=nn_batchnorm1d_torch,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_batchnorm1d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.BatchNorm2d': OpInfo(
        name='mint.nn.BatchNorm2d',
        op=nn_batchnorm2d_ms,
        ref=nn_batchnorm2d_torch,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_batchnorm2d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.BatchNorm3d': OpInfo(
        name='mint.nn.BatchNorm3d',
        op=nn_batchnorm3d_ms,
        ref=nn_batchnorm3d_torch,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_batchnorm3d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.batch_norm': OpInfo(
        name='mint.nn.functional.batch_norm',
        op=mint.nn.functional.batch_norm,
        ref=torch.nn.functional.batch_norm,
        # `running_mean` and `running_variance` in  batch_norm cannot have requires_grad True.
        grad_position=(0, 3, 4),
        # float16 has precision issue
        dtypes_ascend=(ms.float32,),
        # bfloat16 and float16 has precision issue
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_batch_norm,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.binary_cross_entropy': OpInfo(
        name='mint.nn.functional.binary_cross_entropy',
        op=mint.nn.functional.binary_cross_entropy,
        ref=torch.nn.functional.binary_cross_entropy,
        # float16 has occasional precision issue
        dtypes_ascend=(ms.float32,),
        # float16 has precision issue
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_binary_cross_entropy,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.binary_cross_entropy_with_logits': OpInfo(
        name='mint.nn.functional.binary_cross_entropy_with_logits',
        op=mint.nn.functional.binary_cross_entropy_with_logits,
        ref=torch.nn.functional.binary_cross_entropy_with_logits,
        # float16 has precision issue
        dtypes_ascend=(ms.float32,),
        # bfloat16 and float16 has precision issue
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_binary_cross_entropy_with_logits,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.gelu': UnaryOpInfo(
        name='mint.nn.functional.gelu',
        op=mint.nn.functional.gelu,
        ref=torch.nn.functional.gelu,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.GELU': UnaryOpInfo(
        name='mint.nn.GELU',
        op=nn_gelu_ms,
        ref=nn_gelu_torch,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.glu': UnaryOpInfo(
        name='mint.nn.functional.glu',
        op=mint.nn.functional.glu,
        ref=torch.nn.functional.glu,
        # float16 precision is not supported for `glu` since precision issue in 910A
        dtypes_ascend=(ms.float32, ms.float64),
        # bfloat16 precision is not supported for `glu` since precision issue in 910B
        dtypes_ascend910b=(ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_glu_func,
        op_extra_reference_inputs_func=extra_sample_inputs_glu_func,
        op_dynamic_inputs_func=dynamic_sample_inputs_glu_func,
    ),
    'mint.nn.GLU': UnaryOpInfo(
        name='mint.nn.GLU',
        op=nn_glu_ms,
        ref=nn_glu_torch,
        # float16 precision is not supported for `glu` since precision issue in 910A
        dtypes_ascend=(ms.float32, ms.float64),
        # bfloat16 precision is not supported for `glu` since precision issue in 910B
        dtypes_ascend910b=(ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_glu_func,
        op_extra_reference_inputs_func=extra_sample_inputs_glu_func,
        op_dynamic_inputs_func=dynamic_sample_inputs_glu_func,
    ),
    'mint.nn.functional.hardsigmoid': UnaryOpInfo(
        name='mint.nn.functional.hardsigmoid',
        op=mint.nn.functional.hardsigmoid,
        ref=torch.nn.functional.hardsigmoid,
        # Remove int32 from the dtypes since it's not supported by torch in cpu backend.
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.hardswish': UnaryOpInfo(
        name='mint.nn.functional.hardswish',
        op=mint.nn.functional.hardswish,
        ref=torch.nn.functional.hardswish,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.l1_loss': OpInfo(
        name='mint.nn.functional.l1_loss',
        op=mint.nn.functional.l1_loss,
        ref=torch.nn.functional.l1_loss,
        dtypes_ascend=(ms.float16, ms.float32),
        # bfloat16 has occasional accuracy issue
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_l1_loss,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.leaky_relu': UnaryOpInfo(
        name='mint.nn.functional.leaky_relu',
        op=mint.nn.functional.leaky_relu,
        ref=torch.nn.functional.leaky_relu,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.log_softmax': OpInfo(
        name='mint.nn.functional.log_softmax',
        op=mint.nn.functional.log_softmax,
        ref=torch.nn.functional.log_softmax,
        # float64 has precision issue
        dtypes_ascend=(ms.float16, ms.float32),
        # bfloat16 and float64 has precision issue
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_softmax_variant,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.logsigmoid': UnaryOpInfo(
        name='mint.nn.functional.logsigmoid',
        op=mint.nn.functional.logsigmoid,
        ref=torch.nn.functional.logsigmoid,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.mse_loss': OpInfo(
        name='mint.nn.functional.mse_loss',
        op=mint.nn.functional.mse_loss,
        ref=torch.nn.functional.mse_loss,
        dtypes_ascend=(ms.float16, ms.float32),
        # bfloat16 is not supported in torch cpu
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_loss,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.softmax': OpInfo(
        name='mint.nn.functional.softmax',
        op=mint.nn.functional.softmax,
        ref=torch.nn.functional.softmax,
        # float64 has precision issue
        dtypes_ascend=(ms.float16, ms.float32),
        # float64 has precision issue
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=sample_inputs_softmax_variant,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.Identity': UnaryOpInfo(
        name='mint.nn.Identity',
        op=nn_identity_ms,
        ref=nn_identity_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.LogSigmoid': UnaryOpInfo(
        name='mint.nn.LogSigmoid',
        op=nn_logsigmoid_ms,
        ref=nn_logsigmoid_torch,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.PReLU': UnaryOpInfo(
        name='mint.nn.PReLU',
        op=nn_prelu_ms,
        ref=nn_prelu_torch,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_prelu_func,
    ),
    'mint.nn.functional.relu': UnaryOpInfo(
        name='mint.nn.functional.relu',
        op=mint.nn.functional.relu,
        ref=torch.nn.functional.relu,
        dtypes_ascend=(ms.float16, ms.float32, ms.int8, ms.int32, ms.int64, ms.uint8),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.int8, ms.int32, ms.int64, ms.uint8),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.ReLU': UnaryOpInfo(
        name='mint.nn.ReLU',
        op=nn_relu_ms,
        ref=nn_relu_torch,
        dtypes_ascend=(ms.float16, ms.float32, ms.int8, ms.int32, ms.int64, ms.uint8),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.int8, ms.int32, ms.int64, ms.uint8),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.relu6': UnaryOpInfo(
        name='mint.nn.functional.relu6',
        op=mint.nn.functional.relu6,
        ref=torch.nn.functional.relu6,
        dtypes_ascend=(ms.float16, ms.float32, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.ReLU6': UnaryOpInfo(
        name='mint.nn.ReLU6',
        op=nn_relu6_ms,
        ref=nn_relu6_torch,
        dtypes_ascend=(ms.float16, ms.float32, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.add': BinaryOpInfo(
        name='Tensor.add',
        op=tensor_add_ms,
        ref=tensor_add_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        supports_left_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.sub': BinaryOpInfo(
        name='mint.sub',
        op=mint.sub,
        customize_op_func=sub_ext_func_grad_without_kwargs,
        ref=torch.sub,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d != ms.bool_),
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        # On Ascend 910B, the bf16 results match PTA bitwise, but specified value has large deviation
        # when using random inputs, so we temporarily override the default loss value for bf16.
        ascend910b_forward_loss_override={ms.bfloat16: 4e-2},
        ascend910b_backward_loss_override={ms.bfloat16: 4e-2},
        op_basic_reference_inputs_func=basic_sample_inputs_add_sub_ext,
        op_dynamic_inputs_func=dynamic_sample_inputs_add_sub_ext,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
    ),
    'mint.equal': BinaryOpInfo(
        name='mint.equal',
        op=mint.equal,
        ref=torch.equal,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.eq': BinaryOpInfo(
        name='mint.eq',
        op=mint.eq,
        ref=torch.eq,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.greater_equal': BinaryOpInfo(
        name='mint.greater_equal',
        op=mint.greater_equal,
        ref=torch.greater_equal,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.greater': BinaryOpInfo(
        name='mint.greater',
        op=mint.greater,
        ref=torch.greater,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.less_equal': BinaryOpInfo(
        name='mint.less_equal',
        op=mint.less_equal,
        ref=torch.less_equal,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.less': BinaryOpInfo(
        name='mint.less',
        op=mint.less,
        ref=torch.less,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.ne': BinaryOpInfo(
        name='mint.ne',
        op=mint.ne,
        ref=torch.ne,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.maximum': BinaryOpInfo(
        name='mint.maximum',
        op=mint.maximum,
        ref=torch.maximum,
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.minimum': BinaryOpInfo(
        name='mint.minimum',
        op=mint.minimum,
        ref=torch.minimum,
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.div': BinaryOpInfo(
        name='mint.div',
        op=mint.div,
        customize_op_func=div_func_grad,
        ref=torch.div,
        # Skip FP16 fwd/bwd on Ascend 910A/910B due to out-of-tolerance numerics vs PyTorch-CPU.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float16),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=True,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.mul': BinaryOpInfo(
        name='mint.mul',
        op=mint.mul,
        ref=torch.mul,
        # Skip FP16 fwd/bwd on Ascend 910A/910B due to out-of-tolerance numerics vs PyTorch-CPU.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float16),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=True,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
        disable_large_value_tensor_inputs=True,
        # mul has precision problem when comparing with torch_cpu, but its result matches torch_npu bitwise.
        ascend910b_forward_loss_override={ms.float32: 3e-2},
    ),
    'mint.repeat_interleave': OpInfo(
        name='mint.repeat_interleave',
        op=mint.repeat_interleave,
        customize_op_func=repeat_interleave_func_grad,
        ref=torch.repeat_interleave,
        dtypes_ascend=tuple(),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_repeat_interleave,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
    ),
    'mint.arange': OpInfo(
        name='mint.arange',
        op=mint.arange,
        ref=torch.arange,
        dtypes_ascend=tuple([ms.float32, ms.int32]),
        dtypes_ascend910b=tuple([ms.float32, ms.int32]),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_arange,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=False,
    ),
    'Tensor.repeat_interleave': OpInfo(
        name='Tensor.repeat_interleave',
        op=tensor_repeat_interleave_ms,
        customize_op_func=tensor_repeat_interleave_func_grad,
        ref=tensor_repeat_interleave_torch,
        dtypes_ascend=tuple(),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_repeat_interleave,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
    ),
    'Tensor.repeat': OpInfo(
        name='Tensor.repeat',
        op=tensor_repeat_ms,
        ref=tensor_repeat_torch,
        # Repeat is shape-only; allow wide dtype sets. Exclude bf16 on generic Ascend for compatibility.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_tensor_repeat,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
    ),
    'Tensor.maximum': BinaryOpInfo(
        name='Tensor.maximum',
        op=tensor_maximum_ms,
        ref=tensor_maximum_torch,
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'Tensor.minimum': BinaryOpInfo(
        name='Tensor.minimum',
        op=tensor_minimum_ms,
        ref=tensor_minimum_torch,
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'Tensor.mul': BinaryOpInfo(
        name='Tensor.mul',
        op=tensor_mul_ms,
        ref=tensor_mul_torch,
        # Skip FP16 fwd/bwd on Ascend 910A/910B due to out-of-tolerance numerics vs PyTorch-CPU.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float16),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
        disable_large_value_tensor_inputs=True,
    ),
    'mint.floor_divide': BinaryOpInfo(
        name='mint.floor_divide',
        op=mint.floor_divide,
        ref=torch.floor_divide,
        is_differentiable=False,
        # torch donen't support bool
        # ms.float16: Precision assertion failed.
        # ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8: Error message EH9999.
        dtypes_ascend=(ms.float32, ms.float64,),
        dtypes_ascend910b=(ms.float32, ms.float64,),
        domain=((None, None), (1, None)),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.pow': BinaryOpInfo(
        name='mint.pow',
        op=mint.pow,
        ref=torch.pow,
        dtypes_ascend=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64,)),
        dtypes_ascend910b=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64, ms.bfloat16)),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_pow,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_pow,
        op_dynamic_inputs_func=None,
        domain=((1e-4, None), (1e-4, None)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.xlogy': BinaryOpInfo(
        name='mint.xlogy',
        op=mint.xlogy,
        ref=torch.xlogy,
        op_basic_reference_inputs_func = wrap_sample_inputs(
            basic_reference_inputs_binary_op_common_func,
            '',
        ),
        op_extra_reference_inputs_func = wrap_sample_inputs(
            extra_reference_inputs_binary_op_common_func,
            '',
        ),
        dtypes_ascend=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.bool_)),
        dtypes_ascend910b=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.bfloat16, ms.bool_)),
        op_dynamic_inputs_func=None,
        supports_both_python_scalar = False,
        domain=((None, None), (1e-4, None)),
        ascend910b_forward_loss_override={ms.bfloat16: 1e-2},
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.sub': BinaryOpInfo(
        name='Tensor.sub',
        op=tensor_sub_ms,
        ref=tensor_sub_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d != ms.bool_),
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        supports_left_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.bitwise_and': BinaryOpInfo(
        name='mint.bitwise_and',
        op=mint.bitwise_and,
        ref=torch.bitwise_and,
        dtypes_ascend=tuple(d for d in dtypes_integral if d != ms.bfloat16 and d not in dtypes_extra_uint),
        dtypes_ascend910b=tuple(d for d in dtypes_integral if d not in dtypes_extra_uint),
        supports_left_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.bitwise_or': BinaryOpInfo(
        name='mint.bitwise_or',
        op=mint.bitwise_or,
        ref=torch.bitwise_or,
        dtypes_ascend=tuple(d for d in dtypes_integral if d != ms.bfloat16 and d not in dtypes_extra_uint),
        dtypes_ascend910b=tuple(d for d in dtypes_integral if d not in dtypes_extra_uint),
        supports_left_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.bitwise_xor': BinaryOpInfo(
        name='mint.bitwise_xor',
        op=mint.bitwise_xor,
        ref=torch.bitwise_xor,
        dtypes_ascend=tuple(d for d in dtypes_integral if d != ms.bfloat16 and d not in dtypes_extra_uint),
        dtypes_ascend910b=tuple(d for d in dtypes_integral if d not in dtypes_extra_uint),
        supports_left_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.logical_and': BinaryOpInfo(
        name='mint.logical_and',
        op=mint.logical_and,
        ref=torch.logical_and,
        # 1j problem: torch.logical_and(1j) != ms.logical_and(1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'Tensor.logical_and': BinaryOpInfo(
        name='Tensor.logical_and',
        op=tensor_logical_and_ms,
        ref=tensor_logical_and_torch,
        # 0+1j problem: torch.logical_and(1j, 1j) != ms.logical_and(1j, 1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.logical_not': UnaryOpInfo(
        name='mint.logical_not',
        op=mint.logical_not,
        ref=torch.logical_not,
        # 0+1j problem: torch.logical_not(1j, 1j) != ms.logical_not(1j, 1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
    ),
    'Tensor.logical_not': UnaryOpInfo(
        name='Tensor.logical_not',
        op=tensor_logical_not_ms,
        ref=tensor_logical_not_torch,
        # 0+1j problem: torch.logical_not(1j, 1j) != ms.logical_not(1j, 1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
    ),
    'mint.logical_or': BinaryOpInfo(
        name='mint.logical_or',
        op=mint.logical_or,
        ref=torch.logical_or,
        # 0+1j problem: torch.logical_or(1j, 1j) != ms.logical_or(1j, 1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'Tensor.logical_or': BinaryOpInfo(
        name='Tensor.logical_or',
        op=tensor_logical_or_ms,
        ref=tensor_logical_or_torch,
        # 0+1j problem: torch.logical_or(1j, 1j) != ms.logical_or(1j, 1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.logical_xor': BinaryOpInfo(
        name='mint.logical_xor',
        op=mint.logical_xor,
        ref=torch.logical_xor,
        # 0+1j problem: torch.logical_xor(1j, 1j) != ms.logical_xor(1j, 1j)
        # Issue ID: #IB99TS
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.atan2': BinaryOpInfo(
        name='mint.atan2',
        op=mint.atan2,
        ref=torch.atan2,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        #dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        #dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        op_basic_reference_inputs_func=basic_sample_inputs_atan2_ext,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        disable_broadcasting_and_discontiguous_tensor_inputs=True,
        disable_scalar_inputs=True,
        disable_extremal_value_tensor_inputs=True,
    ),
    'Tensor.atan2': BinaryOpInfo(
        name='Tensor.atan2',
        op=tensor_atan2_ms,
        ref=tensor_atan2_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        #dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        #dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        op_basic_reference_inputs_func=basic_sample_inputs_atan2_ext,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        disable_broadcasting_and_discontiguous_tensor_inputs=True,
        disable_scalar_inputs=True,
        disable_extremal_value_tensor_inputs=True,
    ),
    'mint.tanh': UnaryOpInfo(
        name='mint.tanh',
        op=mint.tanh,
        ref=torch.tanh,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16
                                                           and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # tanh has precision problem when comparing with torch_cpu, but its result matches torch_npu bitwise.
        ascend_forward_loss_override={ms.float16: 1e-3, ms.float32: 1e-4},
        ascend910b_forward_loss_override={ms.float16: 1e-3, ms.float32: 1e-4},
        ascend_backward_loss_override={ms.float16: 1e-3, ms.float32: 1e-4},
        ascend910b_backward_loss_override={ms.float16: 1e-3, ms.float32: 1e-4},
    ),
    'mint.acos': UnaryOpInfo(
        name='mint.acos',
        op=mint.acos,
        ref=torch.acos,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d == ms.float32),
        # int8 255 torch nan
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        # float16 loss accuracy issue
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.atanh': UnaryOpInfo(
        name='mint.atanh',
        op=mint.atanh,
        ref=torch.atanh,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.atan': UnaryOpInfo(
        name='mint.atan',
        op=mint.atan,
        ref=torch.atan,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.uint16
                                                           and d != ms.uint32 and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.uint16 and d != ms.uint32
                                                               and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.sinh': UnaryOpInfo(
        name='mint.sinh',
        op=mint.sinh,
        ref=torch.sinh,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d != ms.uint16 and d != ms.uint32
                                                           and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d != ms.uint16 and d != ms.uint32 and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.acosh': UnaryOpInfo(
        name='mint.acosh',
        op=mint.acosh,
        ref=torch.acosh,
        domain=(1, None),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d != ms.uint16 and d != ms.uint32
                                                           and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d != ms.uint16 and d != ms.uint32 and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
         # The acosh backward is composed of small ops, and mul op accumulates numerical error,
         # therefore a larger loss threshold is used for float16. This issue has been reviewed.
        ascend_backward_loss_override={ms.float16: 5e-3},
        ascend910b_backward_loss_override={ms.float16: 5e-3},
    ),
    'mint.asinh': UnaryOpInfo(
        name='mint.asinh',
        op=mint.asinh,
        ref=torch.asinh,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d != ms.uint16
                                                           and d != ms.uint32 and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d != ms.uint16 and d != ms.uint32 and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.asin': UnaryOpInfo(
        name='mint.asin',
        op=mint.asin,
        ref=torch.asin,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.cosh': UnaryOpInfo(
        name='mint.cosh',
        op=mint.cosh,
        ref=torch.cosh,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d != ms.uint16
                                                           and d != ms.uint32 and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d != ms.uint16 and d != ms.uint32 and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),
    'mint.cos': UnaryOpInfo(
        name='mint.cos',
        op=mint.cos,
        ref=torch.cos,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d != ms.uint16 and d != ms.uint32
                                                           and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d != ms.uint16 and d != ms.uint32 and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs=True,
    ),

    'Tensor.tanh': UnaryOpInfo(
        name='Tensor.tanh',
        op=tensor_tanh_ms,
        ref=tensor_tanh_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # tanh has precision problem when converting input from half(fp16) to float
        convert_half_to_float=False,
    ),
    'mint.imag': UnaryOpInfo(
        name='mint.imag',
        op=mint.imag,
        ref=torch.imag,
        dtypes_ascend=(ms.complex64, ms.complex128),
        dtypes_ascend910b=(ms.complex64, ms.complex128),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_small_value_tensor_inputs = True,
    ),
    'Tensor.cos': UnaryOpInfo(
        name='Tensor.cos',
        op=tensor_cos_ms,
        ref=tensor_cos_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d != ms.uint16
                                                           and d != ms.uint32 and d != ms.uint64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d != ms.uint16 and d != ms.uint32 and d != ms.uint64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
    ),
    'Tensor.reciprocal': UnaryOpInfo(
        name='Tensor.reciprocal',
        op=tensor_reciprocal_ms,
        ref=tensor_reciprocal_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_small_value_tensor_inputs=True,
    ),
    'mint.nn.Tanh': UnaryOpInfo(
        name='mint.nn.Tanh',
        op=nn_tanh_ms,
        ref=nn_tanh_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # tanh has precision problem when converting input from half(fp16) to float
        convert_half_to_float=False,
    ),
    'mint.tan': UnaryOpInfo(
        name='mint.tan',
        op=mint.tan,
        ref=torch.tan,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
    ),
    'mint.sin': UnaryOpInfo(
        name='mint.sin',
        op=mint.sin,
        ref=torch.sin,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
    ),
    'Tensor.sin': UnaryOpInfo(
        name='Tensor.sin',
        op=tensor_sin_ms,
        ref=tensor_sin_torch,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
    ),
    'mint.sinc': UnaryOpInfo(
        name='mint.sinc',
        op=mint.sinc,
        ref=torch.sinc,
        domain=(-1, 1),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        disable_large_value_tensor_inputs=True,
        disable_extremal_value_tensor_inputs=True,
        convert_half_to_float=True,
        # For float32 inputs, use 0.0004 under normal-distribution inputs for backward according to CCB conclusions.
        # For float16 inputs, use 0.06 on backward according to CCB conclusions.
        # For bfloat16 inputs, standard 2 is satisfied. Use 0.1 on backward according to CCB conclusions.
        ascend_backward_loss_override={ms.float32: 0.0004, ms.float16: 0.06, ms.bfloat16:0.1},
        ascend910b_backward_loss_override={ms.float32: 0.0004, ms.float16: 0.06, ms.bfloat16:0.1},
    ),
    'Tensor.bfloat16': UnaryOpInfo(
        name='Tensor.bfloat16',
        op=tensor_bfloat16_ms,
        ref=tensor_bfloat16_torch,
        dtypes_ascend=(),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.bool': UnaryOpInfo(
        name='Tensor.bool',
        op=tensor_bool_ms,
        ref=tensor_bool_torch,
        dtypes_ascend=(tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.bool_)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bool_),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.double': UnaryOpInfo(
        name='Tensor.double',
        op=tensor_double_ms,
        ref=tensor_double_torch,
        dtypes_ascend=(tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.double)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.double),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.float': UnaryOpInfo(
        name='Tensor.float',
        op=tensor_float_ms,
        ref=tensor_float_torch,
        dtypes_ascend=(tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and  d != ms.float),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.half': UnaryOpInfo(
        name='Tensor.half',
        op=tensor_half_ms,
        ref=tensor_half_torch,
        dtypes_ascend=(tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'Tensor.int': UnaryOpInfo(
        name='Tensor.int',
        op=tensor_int_ms,
        ref=tensor_int_torch,
        dtypes_ascend=(tuple(d for d in dtypes_as_torch if not d.is_complex and  d != ms.bfloat16 and d != ms.int)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and  d != ms.int),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
        disable_large_value_tensor_inputs=True,
        disable_extremal_value_tensor_inputs=True,
    ),
    'Tensor.long': UnaryOpInfo(
        name='Tensor.long',
        op=tensor_long_ms,
        ref=tensor_long_torch,
        dtypes_ascend=(tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.long)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.long),
        dtypes_cpu=(),
        dtypes_gpu=(),
        is_differentiable=False,
    ),
    'mint.chunk': OpInfo(
        name='mint.chunk',
        op=mint.chunk,
        ref=torch.chunk,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_chunk,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_chunk,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_chunk, # mint.chunk limitedly supports dynamic cases
    ),
    'mint.gather': OpInfo(
        name='mint.gather',
        op=mint.gather,
        ref=torch.gather,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_gather,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_gather,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_gather,
    ),
    'Tensor.gather': OpInfo(
        name='Tensor.gather',
        op=tensor_gather_ms,
        ref=tensor_gather_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_gather,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_gather,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_gather,
    ),
    'mint.nn.functional.interpolate(mode="bilinear")': OpInfo(
        name='mint.nn.functional.interpolate(mode="bilinear")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        # On Ascend 910A, the results match PTA bitwise, but the deviation vs torch_cpu is large,
        # thus it should be considered unavailable.
        dtypes_ascend=(),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="bilinear"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="bilinear"),
    ),
    'mint.nn.functional.interpolate(mode="trilinear")': OpInfo(
        name='mint.nn.functional.interpolate(mode="trilinear")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="trilinear"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="trilinear"),
    ),
    'mint.nn.functional.interpolate(mode="bicubic")': OpInfo(
        name='mint.nn.functional.interpolate(mode="bicubic")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="bicubic"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="bicubic"),
    ),
    'mint.nn.functional.interpolate(mode="linear")': OpInfo(
        name='mint.nn.functional.interpolate(mode="linear")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        # On Ascend 910A, the results match PTA bitwise, but the deviation vs torch_cpu is large,
        # thus it should be considered unavailable.
        dtypes_ascend=(),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="linear"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="linear"),
    ),
    'mint.nn.functional.interpolate(mode="nearest")-1d': OpInfo(
        name='mint.nn.functional.interpolate(mode="nearest")-1d',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.uint8, ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.uint8, ms.float16, ms.float32, ms.float64, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="nearest1d"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="nearest1d"),
    ),
    'mint.nn.functional.interpolate(mode="nearest")-2d': OpInfo(
        name='mint.nn.functional.interpolate(mode="nearest")-2d',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.uint8, ms.float16, ms.float32),
        dtypes_ascend910b=(ms.uint8, ms.float16, ms.float32, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="nearest2d"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="nearest2d"),
    ),
    'mint.nn.functional.interpolate(mode="nearest")-3d': OpInfo(
        name='mint.nn.functional.interpolate(mode="nearest")-3d',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="nearest3d"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="nearest3d"),
    ),
    'mint.mean': ReductionOpInfo(
        name='mint.mean',
        op=mint.mean,
        ref=torch.mean,
        # Todo: MindSpore additionally supports int dtype, while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if (d.is_floating_point or d.is_complex) and d != ms.bfloat16
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d.is_floating_point or d.is_complex
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: empty tensor case result not correct.
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'empty', skip=True
        ),
        op_dynamic_inputs_func=None,
    ),
    'mint.argmax': ReductionOpInfo(
        name='mint.argmax',
        op=mint.argmax,
        ref=torch.argmax,
        # Todo: MindSpore additionally supports uint16/32/64 bool and complex
        # dtype, while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases, results are not correct.
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default_keepdim', skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', 'discontiguous_default', '_extremal_nan'],
            skip=True
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'mint.argmin': ReductionOpInfo(
        name='mint.argmin',
        op=mint.argmin,
        ref=torch.argmin,
        # To do: MindSpore additionally supports uint16/32/64 dtype,
        # while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases,
        # results are not correct.
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default_keepdim', skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', 'discontiguous_default', '_extremal_nan'],
            skip=True
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'mint.count_nonzero': ReductionOpInfo(
        name='mint.count_nonzero',
        op=mint.count_nonzero,
        ref=torch.count_nonzero,
        is_differentiable=False,
        # To do: MindSpore additionally supports uint16/32/64 dtype, while PyTorch does not.
        op_basic_reference_inputs_func=basic_sample_inputs_reduction_count_nonzero,
        op_extra_reference_inputs_func=extra_sample_inputs_reduction_count_nonzero,
        op_dynamic_inputs_func=None,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.argmax': ReductionOpInfo(
        name='Tensor.argmax',
        op=tensor_argmax_ms,
        ref=tensor_argmax_torch,
        # To do: MindSpore additionally supports uint16/32/64 bool and complex dtype, while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(ms.float32,),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases,
        # results are not correct.
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func,
            ['_default', 'dim0_keepdim', 'dim_last'],
            skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', '_default', '_extremal_nan'],
            skip=True
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'Tensor.argmin': ReductionOpInfo(
        name='Tensor.argmin',
        op=tensor_argmin_ms,
        ref=tensor_argmin_torch,
        # To do: MindSpore additionally supports uint16/32/64 dtype,
        # while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases,
        # results are not correct.
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default', skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', '_default', '_extremal_nan'],
            skip=True
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'Tensor.sum': ReductionOpInfo(
        name='Tensor.sum',
        op=tensor_sum_ms,
        ref=tensor_sum_torch,
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # On Ascend 910A, the large-shape input case has about a 3% chance of
        # failing the accuracy check, so set loss to 0.001 temporarily.
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'large_2d',
            loss={"ascend": 0.001}
        ),
    ),
    'Tensor.mean': ReductionOpInfo(
        name='Tensor.mean',
        op=tensor_mean_ms,
        ref=tensor_mean_torch,
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.sum': ReductionOpInfo(
        name='mint.sum',
        op=mint.sum,
        ref=torch.sum,
        op_basic_reference_inputs_func=basic_sample_inputs_mint_sum,
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'empty', skip=True
        ),
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        convert_half_to_float=True,
    ),
    'mint.max': ReductionOpInfo(
        name='mint.max',
        op=mint.max,
        ref=torch.max,
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default', skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'empty', skip=True
        ),
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'Tensor.max': ReductionOpInfo(
        name='Tensor.max',
        op=tensor_max_ms,
        ref=tensor_max_torch,
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default', skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'empty', skip=True
        ),
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'mint.min': ReductionOpInfo(
        name='mint.min',
        op=mint.min,
        ref=torch.min,
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default', skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'empty', skip=True
        ),
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'mint.prod': ReductionOpInfo(
        name='mint.prod',
        op=mint.prod,
        ref=torch.prod,
        op_basic_reference_inputs_func=wrap_sample_inputs(
            basic_reference_inputs_reduction_op_common_func,
            ['_dim_last_dim0', '_default_keepdim'],
            skip=True
        ),
        op_extra_reference_inputs_func=wrap_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, '', skip=True,
        ),
        op_dynamic_inputs_func=None,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d != ms.bfloat16 and d!=ms.float32 and d!=ms.float16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
        supports_multiple_dims=False,
        ascend910b_forward_loss_override={ms.bfloat16: 6.6e+12},
        ascend910b_backward_loss_override={ms.bfloat16: 1.41e+14},
        is_differentiable=False,
    ),
    'mint.tile': UnaryOpInfo(
        name='mint.tile',
        op=mint.tile,
        ref=torch.tile,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_tile,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        is_differentiable=True,
    ),
    'Tensor.tile': UnaryOpInfo(
        name='Tensor.tile',
        op=tensor_tile_ms,
        ref=tensor_tile_torch,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_tile,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        is_differentiable=True,
    ),
    'Tensor.eq': BinaryOpInfo(
        name='Tensor.eq',
        op=tensor_eq_ms,
        ref=tensor_eq_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.greater_equal': BinaryOpInfo(
        name='Tensor.greater_equal',
        op=tensor_greater_equal_ms,
        ref=tensor_greater_equal_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.greater': BinaryOpInfo(
        name='Tensor.greater',
        op=tensor_greater_ms,
        ref=tensor_greater_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.less_equal': BinaryOpInfo(
        name='Tensor.less_equal',
        op=tensor_less_equal_ms,
        ref=tensor_less_equal_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.less': BinaryOpInfo(
        name='Tensor.less',
        op=tensor_less_ms,
        ref=tensor_less_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.ne': BinaryOpInfo(
        name='Tensor.ne',
        op=tensor_ne_ms,
        ref=tensor_ne_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.gt': BinaryOpInfo(
        name='Tensor.gt',
        op=tensor_gt_ms,
        ref=tensor_gt_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.le': BinaryOpInfo(
        name='Tensor.le',
        op=tensor_le_ms,
        ref=tensor_le_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.lt': BinaryOpInfo(
        name='Tensor.lt',
        op=tensor_lt_ms,
        ref=tensor_lt_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.floor': UnaryOpInfo(
        name='mint.floor',
        op=mint.floor,
        ref=torch.floor,
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch if not d.is_complex and d not in [ms.bool_, ms.bfloat16]
        ),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bool_),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.ceil': UnaryOpInfo(
        name='mint.ceil',
        op=mint.ceil,
        ref=torch.ceil,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.float64),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.exp': UnaryOpInfo(
        name='mint.exp',
        op=mint.exp,
        ref=torch.exp,
        dtypes_ascend=(
            ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_
        ),
        dtypes_ascend910b=(
            ms.bfloat16, ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_
        ),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.log': UnaryOpInfo(
        name='mint.log',
        op=mint.log,
        ref=torch.log,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(1e-5, None),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.neg': UnaryOpInfo(
        name='mint.neg',
        op=mint.neg,
        ref=torch.neg,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_])),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.sigmoid': UnaryOpInfo(
        name='mint.sigmoid',
        op=mint.sigmoid,
        ref=torch.sigmoid,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.sqrt': UnaryOpInfo(
        name='mint.sqrt',
        op=mint.sqrt,
        ref=torch.sqrt,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(0, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.abs': UnaryOpInfo(
        name='mint.abs',
        op=mint.abs,
        ref=torch.abs,
        # torch donen't support bool
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d not in [ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bool_)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.floor': UnaryOpInfo(
        name='Tensor.floor',
        op=tensor_floor_ms,
        ref=tensor_floor_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d not in [ms.bool_, ms.bfloat16]),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bool_),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.ceil': UnaryOpInfo(
        name='Tensor.ceil',
        op=tensor_ceil_ms,
        ref=tensor_ceil_torch,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.float64),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.exp': UnaryOpInfo(
        name='Tensor.exp',
        op=tensor_exp_ms,
        ref=tensor_exp_torch,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_),
        dtypes_ascend910b=(
            ms.bfloat16, ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_
        ),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.log': UnaryOpInfo(
        name='Tensor.log',
        op=tensor_log_ms,
        ref=tensor_log_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(1e-5, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.neg': UnaryOpInfo(
        name='Tensor.neg',
        op=tensor_neg_ms,
        ref=tensor_neg_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_])),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.sigmoid': UnaryOpInfo(
        name='Tensor.sigmoid',
        op=tensor_sigmoid_ms,
        ref=tensor_sigmoid_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.sqrt': UnaryOpInfo(
        name='Tensor.sqrt',
        op=tensor_sqrt_ms,
        ref=tensor_sqrt_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(0, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.abs': UnaryOpInfo(
        name='Tensor.abs',
        op=tensor_abs_ms,
        ref=tensor_abs_torch,
        # torch donen't support bool_
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d not in [ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bool_)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.square': UnaryOpInfo(
        name='Tensor.square',
        op=tensor_square_ms,
        ref=tensor_square_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.select': OpInfo(
        name='Tensor.select',
        op=tensor_select_ms,
        ref=tensor_select_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_select,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_select,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.floor_divide': BinaryOpInfo(
        name='Tensor.floor_divide',
        op=tensor_floor_divide_ms,
        ref=tensor_floor_divide_torch,
        is_differentiable=False,
        # torch donen't support bool
        # ms.float16: Precision assertion failed.
        # ms.uint8, ms.int8, ms.int16, ms.int32, ms.int64: Error message EH9999.
        dtypes_ascend=(ms.float32, ms.float64,),
        dtypes_ascend910b=(ms.float32, ms.float64,),
        domain=((None, None), (1, None)),
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.ones_like': UnaryOpInfo(
        name='mint.ones_like',
        op=mint.ones_like,
        ref=torch.ones_like,
        is_differentiable=False,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.permute': OpInfo(
        name='mint.permute',
        op=mint.permute,
        ref=torch.permute,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_permute,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.zeros_like': UnaryOpInfo(
        name='mint.zeros_like',
        op=mint.zeros_like,
        ref=torch.zeros_like,
        is_differentiable=False,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d not in [ms.uint32, ms.uint64, ms.bfloat16]),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d not in [ms.uint32, ms.uint64]),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.select': OpInfo(
        name='mint.select',
        op=mint.select,
        ref=torch.select,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_select,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_select,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.cumsum': OpInfo(
        name='mint.cumsum',
        op=mint.cumsum,
        ref=torch.cumsum,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_cumsum,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_cumsum,
        op_dynamic_inputs_func=None,
        # issue ID ICWZN8: cumsum has accumulation bias problem.
        ascend_forward_loss_override={ms.float16: 1e-2},
        ascend_backward_loss_override={ms.float16: 1e-2},
        ascend910b_forward_loss_override={ms.float16: 5e-3},
        ascend910b_backward_loss_override={ms.float16: 5e-3},
        is_differentiable=False,
    ),
    'mint.index_select': OpInfo(
        name='mint.index_select',
        op=mint.index_select,
        ref=torch.index_select,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_index_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_index_select,
        op_dynamic_inputs_func=None,
    ),
    'Tensor.index_select': OpInfo(
        name='Tensor.index_select',
        op=tensor_index_select_ms,
        ref=tensor_index_select_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_index_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_index_select,
        op_dynamic_inputs_func=None,
    ),
    'mint.masked_select': OpInfo(
        name='mint.masked_select',
        op=mint.masked_select,
        ref=torch.masked_select,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_masked_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_masked_select,
        op_dynamic_inputs_func=None,
    ),
    'mint.nonzero': OpInfo(
        name='mint.nonzero',
        op=mint.nonzero,
        ref=torch.nonzero,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_nonzero,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_nonzero,
        op_dynamic_inputs_func=None,
        is_differentiable=False,
    ),
    'mint.flatten': OpInfo(
        name='mint.flatten',
        op=mint.flatten,
        ref=torch.flatten,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_flatten,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_flatten,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.expand_as': OpInfo(
        name='Tensor.expand_as',
        op=tensor_expand_as_ms,
        ref=tensor_expand_as_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_expand_as,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.reshape': OpInfo(
        name='mint.reshape',
        op=mint.reshape,
        ref=torch.reshape,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_reshape,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_reshape,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.scatter': OpInfo(
        name='mint.scatter',
        op=mint.scatter,
        ref=torch.scatter,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_scatter,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.cat': OpInfo(
        name='mint.cat',
        op=mint.cat,
        customize_op_func=mint_cat_func_with_flatten_inputs,
        ref=torch.cat,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_cat,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_cat,
        op_dynamic_inputs_func=None,
        flatten_inputs_for_backward=True,
    ),
    'mint.nn.functional.one_hot': OpInfo(
        name='mint.nn.functional.one_hot',
        op=mint.nn.functional.one_hot,
        ref=torch.nn.functional.one_hot,
        dtypes_ascend=tuple((ms.int64,)),
        dtypes_ascend910b=tuple((ms.int64,)),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_one_hot,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_one_hot,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.unsqueeze': OpInfo(
        name='mint.unsqueeze',
        op=mint.unsqueeze,
        ref=torch.unsqueeze,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_unsqueeze,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.transpose': OpInfo(
        name='mint.transpose',
        op=mint.transpose,
        ref=torch.transpose,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_transpose,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.topk': OpInfo(
        name='mint.topk',
        op=mint.topk,
        ref=torch.topk,
        dtypes_ascend=(ms.float32, ),
        dtypes_ascend910b=(ms.float32, ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_topk,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.normal': OpInfo(
        name='mint.normal',
        op=mint.normal,
        ref=torch.normal,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_normal,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        skip_forward_comparison=True,
    ),
    'mint.nn.functional.adaptive_avg_pool3d': OpInfo(
        name='mint.nn.functional.adaptive_avg_pool3d',
        op=mint.nn.functional.adaptive_avg_pool3d,
        ref=torch.nn.functional.adaptive_avg_pool3d,
        dtypes_ascend=tuple(),
        # IARVJM 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_adaptive_avg_pool3d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.adaptive_avg_pool2d': OpInfo(
        name='mint.nn.functional.adaptive_avg_pool2d',
        op=mint.nn.functional.adaptive_avg_pool2d,
        ref=torch.nn.functional.adaptive_avg_pool2d,
        dtypes_ascend=tuple(),
        # IARVJM 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_adaptive_avg_pool2d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.adaptive_avg_pool1d': OpInfo(
        name='mint.nn.functional.adaptive_avg_pool1d',
        op=mint.nn.functional.adaptive_avg_pool1d,
        ref=torch.nn.functional.adaptive_avg_pool1d,
        dtypes_ascend=tuple(),
        # IARVJM 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_adaptive_avg_pool1d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.avg_pool2d': OpInfo(
        name='mint.nn.functional.avg_pool2d',
        op=mint.nn.functional.avg_pool2d,
        ref=torch.nn.functional.avg_pool2d,
        dtypes_ascend=tuple(),
        # IARVJM 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_avgpool2d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.functional.max_pool2d': OpInfo(
        name='mint.nn.functional.max_pool2d',
        op=mint.nn.functional.max_pool2d,
        ref=torch.nn.functional.max_pool2d,
        dtypes_ascend=tuple(),
        #skip 910a
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_max_pool,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.AdaptiveAvgPool2d': OpInfo(
        name='mint.nn.AdaptiveAvgPool2d',
        op=nn_adaptiveavgpool2d_ms,
        ref=nn_adaptiveavgpool2d_torch,
        dtypes_ascend=tuple(),
        # IARVJM 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_adaptivegpool2d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.AdaptiveAvgPool3d': OpInfo(
        name='mint.nn.AdaptiveAvgPool3d',
        op=nn_adaptiveavgpool3d_ms,
        ref=nn_adaptiveavgpool3d_torch,
        dtypes_ascend=tuple(),
        # IARVJM 910A underlying architecture does not support fp32 and will convert to fp16 operations.
        # Therefore, float32 will be skipped in 910A.
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d == ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_adaptivegpool3d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'Tensor.unsqueeze': OpInfo(
        name='Tensor.unsqueeze',
        op=tensor_unsqueeze_ms,
        ref=tensor_unsqueeze_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_unsqueeze,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'Tensor.transpose': OpInfo(
        name='Tensor.transpose',
        op=tensor_transpose_ms,
        ref=tensor_transpose_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_transpose,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'Tensor.topk': OpInfo(
        name='Tensor.topk',
        op=tensor_topk_ms,
        ref=tensor_topk_torch,
        dtypes_ascend=(ms.float32, ),
        dtypes_ascend910b=(ms.float32, ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_topk,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.unique': OpInfo(
        name='mint.unique',
        op=mint.unique,
        ref=torch.unique,
        dtypes_ascend=(ms.float32, ms.float16),
        dtypes_ascend910b=(ms.float32, ms.float16, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_unique,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        is_differentiable=False,
    ),
    'Tensor.unique': OpInfo(
        name='Tensor.unique',
        op=tensor_unique_ms,
        ref=tensor_unique_torch,
        dtypes_ascend=(ms.float32, ms.float16),
        dtypes_ascend910b=(ms.float32, ms.float16, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_unique,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        is_differentiable=False,
    ),
    'mint.clamp': OpInfo(
        name='mint.clamp',
        op=mint.clamp,
        ref=torch.clamp,
        dtypes_ascend=(ms.float32, ms.float16),
        dtypes_ascend910b=(ms.float32, ms.float16, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_clamp,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'Tensor.clamp': OpInfo(
        name='Tensor.clamp',
        op=tensor_clamp_ms,
        ref=tensor_clamp_torch,
        dtypes_ascend=(ms.float32, ms.float16),
        dtypes_ascend910b=(ms.float32, ms.float16, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_clamp,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.stack': OpInfo(
        name='mint.stack',
        op=mint.stack,
        customize_op_func=mint_stack_func_with_flatten_inputs,
        ref=torch.stack,
        dtypes_ascend=(ms.float32, ms.float16),
        dtypes_ascend910b=(ms.float32, ms.float16, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_stack,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_stack,
        op_dynamic_inputs_func=None,
        flatten_inputs_for_backward=True,
    ),
    'mint.nn.functional.dropout': OpInfo(
        name='mint.nn.functional.dropout',
        op=mint.nn.functional.dropout,
        ref=torch.nn.functional.dropout,
        dtypes_ascend=tuple((ms.float32,)),
        dtypes_ascend910b=tuple((ms.float32,)),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_dropout,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.pad(mode="constant")': OpInfo(
        name='mint.nn.functional.pad(mode="constant")',
        op=mint.nn.functional.pad,
        ref=torch.nn.functional.pad,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_pad, mode="constant"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.pad(mode="reflect")': OpInfo(
        name='mint.nn.functional.pad(mode="reflect")',
        op=mint.nn.functional.pad,
        ref=torch.nn.functional.pad,
        dtypes_ascend=tuple((ms.float32,)),
        dtypes_ascend910b=tuple((ms.float32,)),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_pad, mode="reflect"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.pad(mode="replicate")': OpInfo(
        name='mint.nn.functional.pad(mode="replicate")',
        op=mint.nn.functional.pad,
        ref=torch.nn.functional.pad,
        dtypes_ascend=tuple((ms.float32,)),
        dtypes_ascend910b=tuple((ms.float32,)),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_pad, mode="replicate"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.pad(mode="circular")': OpInfo(
        name='mint.nn.functional.pad(mode="circular")',
        op=mint.nn.functional.pad,
        ref=torch.nn.functional.pad,
        dtypes_ascend=tuple((ms.int64,)),
        dtypes_ascend910b=tuple((ms.int64,)),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_pad, mode="circular"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.split': OpInfo(
        name='mint.split',
        op=mint.split,
        ref=torch.split,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_split,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'Tensor.split': OpInfo(
        name='Tensor.split',
        op=tensor_split_ms,
        ref=tensor_split_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_tensor_split,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.Conv1d': OpInfo(
        name='mint.nn.Conv1d',
        op=mint_nn_conv1d_ms,
        ref=nn_conv1d_torch,
        dtypes_ascend=tuple([ms.float16,]),
        dtypes_ascend910b=tuple([ms.float16]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_nn_conv1d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'mint.nn.functional.conv1d': OpInfo(
        name='mint.nn.functional.conv1d',
        op=mint.nn.functional.conv1d,
        ref=torch.nn.functional.conv1d,
        dtypes_ascend=tuple([ms.float16,]),
        dtypes_ascend910b=tuple([ms.float16]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_conv1d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'mint.nn.functional.conv2d': OpInfo(
        name='mint.nn.functional.conv2d',
        op=mint.nn.functional.conv2d,
        ref=torch.nn.functional.conv2d,
        dtypes_ascend=(),
        dtypes_ascend910b=tuple([ms.float16]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_conv2d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'mint.nn.functional.conv3d': OpInfo(
        name='mint.nn.functional.conv3d',
        op=mint.nn.functional.conv3d,
        ref=torch.nn.functional.conv3d,
        dtypes_ascend=(),
        dtypes_ascend910b=tuple([ms.float32]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_conv3d,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
        # For Conv-family operators with float32 inputs, HF32 is enabled by default.
        # HiSilicon recommends a one-sided 0.1% accuracy guard.
        # For bfloat16 inputs, standard 2 rather than standard 1 is satisfied.
        ascend_forward_loss_override={ms.float32: 0.001},
        ascend_backward_loss_override={ms.float32: 0.001},
        ascend910b_forward_loss_override={ms.float32: 0.001, ms.bfloat16: 0.08},
        ascend910b_backward_loss_override={ms.float32: 0.001, ms.bfloat16: 0.08},
    ),
    'mint.nn.functional.linear': OpInfo(
        name='mint.nn.functional.linear',
        op=mint.nn.functional.linear,
        ref=torch.nn.functional.linear,
        dtypes_ascend=tuple([ms.float16,]),
        dtypes_ascend910b=tuple([ms.float32]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_linear,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'mint.nn.Linear': OpInfo(
        name='mint.nn.Linear',
        op=mint_nn_linear_ms,
        ref=nn_linear_torch,
        dtypes_ascend=tuple([ms.float16,]),
        dtypes_ascend910b=tuple([ms.float16]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_nn_linear,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'Tensor.masked_scatter': OpInfo(
        name='Tensor.masked_scatter',
        op=tensor_masked_scatter_ms,
        ref=tensor_masked_scatter_torch,
        dtypes_ascend=tuple([ms.float32,]),
        dtypes_ascend910b=tuple([ms.float32]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_tensor_masked_scatter,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'Tensor.masked_scatter_': OpInfo(
        name='Tensor.masked_scatter_',
        op=tensor_masked_scatter__ms,
        ref=tensor_masked_scatter__torch,
        dtypes_ascend=tuple([ms.float32,]),
        dtypes_ascend910b=tuple([ms.float32]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_tensor_masked_scatter_,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'Tensor.add_': OpInfo(
        name='Tensor.add_',
        op=tensor_add__ms,
        ref=tensor_add__torch,
        dtypes_ascend=tuple([ms.float32,]),
        dtypes_ascend910b=tuple([ms.float32]),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_tensor_add_,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=True,
    ),
    'Tensor.view(dtype)': OpInfo(
        name='Tensor.view(dtype)',
        op=tensor_view_dtype_ms,
        ref=lambda op_input, *op_args, **op_kwargs: op_input.view(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=dtypes_as_torch,
        dtypes_gpu=(),
        default_golden_loss_func=lambda dtype: 0,
        convert_half_to_float=False,
    ),
    'mint.nn.functional.selu': UnaryOpInfo(
        name='mint.nn.functional.selu',
        op=nn_functional_selu_ms,
        ref=nn_functional_selu_torch,
        dtypes_ascend=tuple([ms.float16, ms.float32]),
        dtypes_ascend910b=tuple([ms.bfloat16, ms.float16, ms.float32,]),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.layer_norm': OpInfo(
        name='mint.nn.functional.layer_norm',
        op=mint.nn.functional.layer_norm,
        ref=torch.nn.functional.layer_norm,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_layer_norm,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
    'mint.nn.GroupNorm': OpInfo(
        name='mint.nn.GroupNorm',
        op=nn_group_norm_ms,
        ref=nn_group_norm_torch,
        dtypes_ascend=(ms.float16,),
        dtypes_ascend910b=(ms.float16,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_GroupNorm,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
        # IBHXMW: For backward on Ascend, HiSilicon has confirmed that the naive algorithm may cause
        # precision difference.
        ascend_backward_loss_override={ms.float16: 0.009},
        ascend910b_backward_loss_override={ms.float16: 0.002},
    ),
    'mint.nn.LayerNorm': OpInfo(
        name='mint.nn.LayerNorm',
        op=nn_layer_norm_ms,
        ref=nn_layer_norm_torch,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_LayerNorm,
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=None,
    ),
}

all_op_db = list(op_db.keys())

binary_op_db = [
    'mint.add',
    'mint.sub',
    'mint.bitwise_and',
    'mint.bitwise_or',
    'mint.bitwise_xor',
    'mint.logical_and',
    'mint.logical_or',
    'mint.logical_xor',
    'Tensor.logical_and',
    'Tensor.logical_or',
    'Tensor.add',
    'Tensor.sub',
    'mint.equal',
    'mint.eq',
    'mint.greater_equal',
    'mint.greater',
    'mint.less_equal',
    'mint.less',
    'mint.ne',
    'mint.maximum',
    'mint.minimum',
    'mint.div',
    'mint.mul',
    'Tensor.eq',
    'Tensor.greater_equal',
    'Tensor.greater',
    'Tensor.less_equal',
    'Tensor.less',
    'Tensor.ne',
    'Tensor.gt',
    'Tensor.le',
    'Tensor.lt',
    'Tensor.maximum',
    'Tensor.minimum',
    'Tensor.mul',
    'mint.floor_divide',
    'Tensor.floor_divide',
    'mint.pow',
    'mint.atan2',
    'mint.xlogy',
    'Tensor.atan2',
]

unary_op_db = [
    'Tensor.astype',
    'Tensor.byte',
    'Tensor.clone',
    'Tensor.contiguous',
    'Tensor.is_contiguous',
    'Tensor.numpy',
    'Tensor.to',
    'mint.clone',
    'mint.empty',
    'mint.empty_like',
    'mint.nn.functional.gelu',
    'mint.nn.GELU',
    'mint.nn.functional.glu',
    'mint.nn.GLU',
    #'mint.nn.functional.hardsigmoid',
    'mint.nn.functional.hardswish',
    'mint.nn.functional.leaky_relu',
    'mint.nn.functional.logsigmoid',
    'mint.nn.Identity',
    'mint.nn.LogSigmoid',
    'mint.nn.PReLU',
    'mint.nn.functional.relu',
    'mint.nn.ReLU',
    #'mint.nn.functional.relu6',
    #'mint.nn.ReLU6',
    'mint.tanh',
    'Tensor.tanh',
    'mint.nn.Tanh',
    'mint.floor',
    'mint.ceil',
    'mint.exp',
    'mint.log',
    'mint.neg',
    'mint.sigmoid',
    'mint.sqrt',
    'mint.abs',
    'Tensor.floor',
    'Tensor.ceil',
    'Tensor.exp',
    'Tensor.log',
    'Tensor.neg',
    'Tensor.sigmoid',
    'Tensor.sqrt',
    'Tensor.abs',
    'Tensor.square',
    'mint.ones_like',
    'mint.zeros_like',
    'mint.tan',
    'mint.sin',
    'mint.sinc',
    'Tensor.sin',
    'Tensor.bfloat16',
    'Tensor.bool',
    'Tensor.double',
    'Tensor.float',
    'Tensor.half',
    'Tensor.int',
    'Tensor.long',
    'mint.logical_not',
    'Tensor.logical_not',
    'mint.nn.functional.selu',
    'mint.tile',
    'Tensor.tile',
    'mint.acos',
    'mint.atanh',
    'mint.atan',
    'mint.sinh',
    'mint.acosh',
    'mint.asinh',
    'mint.asin',
    'mint.cosh',
    'mint.cos',
    'Tensor.cos',
    'mint.imag',
    'Tensor.reciprocal',
]

other_op_db = [
    'mint.chunk',
    'mint.gather',
    'mint.cat',
    'mint.cumsum',
    'mint.index_select',
    'Tensor.index_select',
    'mint.masked_select',
    'mint.split',
    'Tensor.split',
    'mint.nonzero',
    'mint.flatten',
    'mint.nn.functional.interpolate(mode="bilinear")',
    'mint.nn.functional.interpolate(mode="trilinear")',
    'mint.nn.functional.interpolate(mode="bicubic")',
    'mint.nn.functional.interpolate(mode="linear")',
    'mint.nn.functional.interpolate(mode="nearest")-1d',
    'mint.nn.functional.interpolate(mode="nearest")-2d',
    'mint.nn.functional.interpolate(mode="nearest")-3d',
    'mint.repeat_interleave',
    'Tensor.repeat_interleave',
    'Tensor.repeat',
    'mint.arange',
    'mint.select',
    'Tensor.select',
    'mint.nn.functional.one_hot',
    'mint.reshape',
    'Tensor.matmul',
    'mint.broadcast_to',
    'mint.matmul',
    'mint.nn.BatchNorm1d',
    'mint.nn.BatchNorm2d',
    'mint.nn.BatchNorm3d',
    'mint.nn.functional.batch_norm',
    'mint.nn.functional.binary_cross_entropy',
    'mint.nn.functional.binary_cross_entropy_with_logits',
    'mint.nn.functional.l1_loss',
    'mint.nn.functional.log_softmax',
    'mint.nn.functional.mse_loss',
    'mint.nn.functional.softmax',
    'mint.unique',
    'Tensor.unique',
    'mint.clamp',
    'Tensor.clamp',
    'mint.stack',
    'Tensor.gather',
    'Tensor.expand_as',
    'mint.scatter',
    'mint.nn.functional.dropout',
    'mint.nn.functional.pad(mode="constant")',
    'mint.nn.functional.pad(mode="reflect")',
    'mint.nn.functional.pad(mode="replicate")',
    'mint.nn.functional.pad(mode="circular")',
    'mint.nn.functional.conv1d',
    'mint.nn.functional.conv2d',
    'mint.nn.functional.conv3d',
    'mint.nn.functional.linear',
    'mint.nn.Linear',
    'mint.nn.Conv1d',
    'Tensor.masked_scatter',
    'Tensor.masked_scatter_',
    'Tensor.add_',
    'mint.nn.functional.layer_norm',
    'mint.nn.GroupNorm',
    'mint.nn.LayerNorm',
    'mint.permute',
    'mint.transpose',
    'mint.unsqueeze',
    'mint.topk',
    'mint.normal',
    'mint.nn.functional.adaptive_avg_pool1d',
    'mint.nn.functional.adaptive_avg_pool2d',
    'mint.nn.functional.adaptive_avg_pool3d',
    'mint.nn.functional.avg_pool2d',
    'mint.nn.functional.max_pool2d',
    'mint.nn.AdaptiveAvgPool2d',
    'mint.nn.AdaptiveAvgPool3d',
    'Tensor.transpose',
    'Tensor.unsqueeze',
    'Tensor.topk',
]

reduction_op_db = [
    'mint.mean',
    'mint.argmax',
    'mint.argmin',
    'mint.count_nonzero',
    'Tensor.mean',
    'Tensor.sum',
    'Tensor.argmax',
    'Tensor.argmin',
    'mint.sum',
    'mint.max',
    'mint.min',
    'mint.prod',
    'Tensor.max',
]

binary_op_kbk_db = [
    'mint.add',
    'mint.sub',
]

unary_op_kbk_db = [
    'mint.tanh',
    'mint.nn.functional.relu',
    'mint.tile',
]

other_op_kbk_db = [
    'mint.chunk',
    'mint.gather',
]

reduction_op_kbk_db = [
    'mint.mean',
    'mint.argmax',
    'mint.count_nonzero',
    'Tensor.sum',
    'Tensor.argmin',
    'mint.min',
    'Tensor.max',
]

def get_op_info(op_name: str, *, op_database: Optional[Dict[str, OpInfo]] = None) -> OpInfo:
    """Return `OpInfo` by name from the provided or default database."""
    if op_name not in all_op_db:
        raise ValueError(f"op name {op_name} not found in op database")
    op_database = op_db if op_database is None else op_database
    return op_database[op_name]
