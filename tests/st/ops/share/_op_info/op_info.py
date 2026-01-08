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
- OpInfo dataclass describing operator metadata for tests.
- BinaryOpInfo convenience subclass with defaults for binary ops.
- ReductionOpInfo convenience subclass with defaults for reduction ops.
- basic_reference_inputs_binary_op_common_func: canonical sample input generator for binary ops.
"""
import functools
import itertools
import warnings
import math
import numpy as np
import mindspore as ms
from typing import Callable, Optional
from dataclasses import dataclass, field
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_common import (
    MEDIUM_DIM_SIZE, SMALL_DIM_SIZE, EXTRA_SMALL_DIM_SIZE, LARGE_DIM_SIZE,
    dtypes_integral, dtypes_extra_uint,
    get_default_loss,
)
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array
)

@dataclass
class OpInfo:
    """Metadata describing an operator under test.

    Attributes:
        name: Short op alias used in logs and test names.
        op: MindSpore callable implementation.
        customize_op_func: MindSpore callable used for gradient/dynamic nets;
            falls back to ``op`` if not provided (e.g., when kwargs need special handling).
        ref: Reference implementation (e.g., PyTorch/NumPy callable).

        dtypes_ascend: Supported MindSpore dtypes on Ascend devices.
        dtypes_ascend910b: Supported dtypes specifically on Ascend 910B.
        dtypes_cpu: Supported dtypes on CPU.
        dtypes_gpu: Supported dtypes on GPU.
        dtypes_intersection: Intersection of supported dtypes across all listed
            backends. Auto-populated in ``__post_init__`` if left empty.

        op_basic_reference_inputs_func: Function that generates basic reference inputs for tests.
        op_extra_reference_inputs_func: Function that generates comprehensive sample inputs for tests.
        op_dynamic_inputs_func: Function that generates dynamic-shape samples.
        op_error_inputs_func: Function that generates error/negative samples.

        grad_position: The position of the input tensor to compute gradients. Default is None, which means 
            compute gradients for all input tensors.
        domain: The [low, high) domain of the operator.
        exclude_value: values to exclude during generation, Default is None, which means no values to exclude.
        is_differentiable: Whether gradients are expected/computed for the op.
        skip_forward_comparison: Whether to skip forward comparison with the reference implementation.
            This is useful for inherently non-deterministic ops (e.g., random sampling). When enabled,
            the forward will still be executed (to catch runtime errors), but outputs will not be compared.
        is_inplace_op: Whether the op mutates its input (in-place semantics).
        convert_extra_uint: Whether to convert extra uint dtypes for references
            that do not support them (e.g., PyTorch).
        convert_half_to_float: Whether to cast the reference inputs from float16 to float32. In principle, when
            comparing cases with float16 inputs, this conversion should always be performed, so the default value
            of this option is True. This also helps avoid failures when the reference does not support float16 inputs.

        compare_method: Comparison strategy, e.g. 'default_golden',
            'single_golden', or 'double_golden'.
        default_golden_loss_func: Callable returning default numeric tolerance
            (rtol/atol) based on dtype.
        default_loss_override: A dictionary of overridden loss for specific dtype. The key is the dtype,
            the value is the loss. Only works for compare_method 'default_golden'.
        flatten_inputs_for_backward: Whether to flatten (expand) list/tuple/dict inputs into
            positional arguments for backward/grad execution. This is useful for ops whose
            forward takes a sequence of tensors as a single argument, while the customized
            backward wrapper expects flattened positional tensor arguments (e.g. mint.stack/cat).
    """
    # name of primitive, defined in xxx_op.yaml file.
    name: str
    op: Optional[Callable] = None
    customize_op_func: Optional[Callable] = None
    ref: Optional[Callable] = None

    # dtypes supported by each backend
    dtypes_ascend: tuple = field(default_factory=tuple)
    dtypes_ascend910b: tuple = field(default_factory=tuple)
    dtypes_cpu: tuple = field(default_factory=tuple)
    dtypes_gpu: tuple = field(default_factory=tuple)
    dtypes_intersection: tuple = field(default_factory=tuple)

    # function to generate basic reference inputs (basic shape/broadcasting cases) for the op.
    op_basic_reference_inputs_func: Optional[Callable] = None
    # function to generate extra reference inputs (discontiguous/special values/large or small values cases) for the op.
    op_extra_reference_inputs_func: Optional[Callable] = None
    # function to generate dynamic inputs for the op.
    op_dynamic_inputs_func: Optional[Callable] = None
    # function to generate error inputs for the op.
    op_error_inputs_func: Optional[Callable] = None

    # extra options for the op.
    grad_position: Optional[tuple] = None
    domain: Optional[tuple] = None
    exclude_value: Optional[tuple] = None
    is_differentiable: Optional[bool] = True
    # skip forward comparison against reference
    skip_forward_comparison: Optional[bool] = False
    is_inplace_op: Optional[bool] = False
    convert_extra_uint: Optional[bool] = True
    convert_half_to_float: Optional[bool] = True

    # comparison params
    compare_method: Optional[str] = 'default_golden'
    default_golden_loss_func: Optional[Callable] = get_default_loss
    ascend_forward_loss_override: Optional[dict] = None
    ascend_backward_loss_override: Optional[dict] = None
    ascend910b_forward_loss_override: Optional[dict] = None
    ascend910b_backward_loss_override: Optional[dict] = None
    cpu_forward_loss_override: Optional[dict] = None
    cpu_backward_loss_override: Optional[dict] = None
    gpu_forward_loss_override: Optional[dict] = None
    gpu_backward_loss_override: Optional[dict] = None
    flatten_inputs_for_backward: Optional[bool] = False

    def __post_init__(self):
        if not self.dtypes_intersection:
            self.dtypes_intersection = tuple(
                set(self.dtypes_ascend) & set(self.dtypes_ascend910b) & set(self.dtypes_cpu) & set(self.dtypes_gpu)
            )
        if self.customize_op_func is None:
            self.customize_op_func = self.op

# basic op_basic_reference_inputs_func for ops
def basic_reference_inputs_binary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield shape/broadcasting cases for binary ops.

    Generates a variety of tensor shape combinations, including scalars,
    vectors, broadcasting pairs, and empty-dimension cases.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    """
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
        _input = make_func(input_shape, low=op_info.input_low,
                           high=op_info.input_high, exclude_value=op_info.input_exclude_value)
        _other = make_func(other_shape, low=op_info.other_low,
                           high=op_info.other_high, exclude_value=op_info.other_exclude_value)

        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )

# op_extra_reference_inputs_func for ops
def _generate_binary_op_tensors_sample_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor sample inputs for a binary op.

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
        (1, 0, 3),   # empty tensors
        (),          # scalar tensors
        (20,),       # 1D tensors with small size
        (812,),      # 1D tensors with medium size
        (1029, 917), # 2D tensors with large size
    )

    for shape in shapes:
        yield OpSampleInput(
            op_input=make_tensor(
                shape,
                dtype,
                device=device,
                low=op_info.input_low,
                high=op_info.input_high,
                exclude_value=op_info.input_exclude_value
            ),
            op_args=(
                make_tensor(
                    shape,
                    dtype,
                    device=device,
                    low=op_info.other_low,
                    high=op_info.other_high,
                    exclude_value=op_info.other_exclude_value
                ),
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_tensor_inputs",
        )


def _generate_binary_op_small_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor inputs with small values for edge cases.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering representative small float/int/unsigned values.
    """
    _uint_values = (0, 1, 31, 63, 128, 195, 254)
    _int_values = (0, -1, 1, -63, 63, -127, 127, -128)
    _float_vals = (
        0.0,
        -0.0,
        -1.0,
        1.0,
        -1.25e-1,
        1.25e-1,
        -1e-3,
        1e-3,
        -math.pi / 2,
        math.pi / 2,
        -math.pi + 1e-5,
        math.pi - 1e-5,
        -math.pi,
        math.pi,
        -math.pi - 1e-5,
        math.pi + 1e-5,
    )

    _input_values = []
    _other_values = []

    if dtype.is_floating_point:
        values_group = itertools.product(_float_vals, _float_vals)
    elif dtype.is_complex:
        _float_values_group = itertools.product(_float_vals, _float_vals)
        _complex_values = [complex(r, i) for r, i in _float_values_group]
        values_group = itertools.product(_complex_values, _complex_values)
    elif dtype in dtypes_integral and dtype != ms.bool_:
        if dtype in dtypes_extra_uint or dtype == ms.uint8:
            values_group = itertools.product(_uint_values, _uint_values)
        else:
            values_group = itertools.product(_int_values, _int_values)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}!")

    for l_value, r_value in values_group:
        if dtype!=ms.bool_ and (
                (op_info.input_exclude_value and l_value in op_info.input_exclude_value) or
                (op_info.other_exclude_value and r_value in op_info.other_exclude_value) ):
            continue
        if (dtype.is_floating_point or dtype in dtypes_integral) and (
            (op_info.input_low is not None and op_info.input_low > l_value)
            or (op_info.input_high is not None and op_info.input_high < l_value)
            or (op_info.other_low is not None and op_info.other_low > r_value)
            or (op_info.other_high is not None and op_info.other_high < r_value)):
            continue
        _input_values.append(l_value)
        _other_values.append(r_value)

    yield OpSampleInput(
        op_input= make_tensor_with_np_array(np.array(_input_values), dtype=dtype, device=device),
        op_args=(make_tensor_with_np_array(np.array(_other_values), dtype=dtype, device=device),),
        op_kwargs={},
        sample_name=f"{op_info.name}_small_value_tensor_inputs",
    )


def _generate_binary_op_large_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor inputs with large values for stress testing.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering large-magnitude float/complex/integer values.
    """
    _int_values = (-1023, 1023, -10922, 10922)
    _fp16_values = (-521, 521, -996.9, 996.9, -13141.5, 13141.5)
    _fp32_values = _fp16_values + (-5062757.7, 5062757.7, -1e20, 1e20)

    _input_values = []
    _other_values = []

    if dtype == ms.float16:
        values_group = itertools.product(_fp16_values, _fp16_values)
    elif dtype.is_floating_point:
        values_group = itertools.product(_fp32_values, _fp32_values)
    elif dtype.is_complex:
        _float_values_group = itertools.product(_fp32_values, _fp32_values)
        _complex_values = [complex(r, i) for r, i in _float_values_group]
        values_group = itertools.product(_complex_values, _complex_values)
    elif dtype in (ms.int16, ms.int32, ms.int64):
        values_group = itertools.product(_int_values, _int_values)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}!")

    for l_value, r_value in values_group:
        if dtype!=ms.bool_ and (
                (op_info.input_exclude_value and l_value in op_info.input_exclude_value) or
                (op_info.other_exclude_value and r_value in op_info.other_exclude_value) ):
            continue
        if (dtype.is_floating_point or dtype in dtypes_integral) and (
            (op_info.input_low is not None and op_info.input_low > l_value)
            or (op_info.input_high is not None and op_info.input_high < l_value)
            or (op_info.other_low is not None and op_info.other_low > r_value)
            or (op_info.other_high is not None and op_info.other_high < r_value)):
            continue
        _input_values.append(l_value)
        _other_values.append(r_value)

    yield OpSampleInput(
        op_input= make_tensor_with_np_array(np.array(_input_values), dtype=dtype, device=device),
        op_args=(make_tensor_with_np_array(np.array(_other_values), dtype=dtype, device=device),),
        op_kwargs={},
        sample_name=f"{op_info.name}_large_value_tensor_inputs",
    )

def _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate broadcasting and (dis)contiguous tensor inputs.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering broadcasting and contiguous/non-contiguous memory layouts.
    """

    shapes = (
        ((1,), ()),
        ((3,), ()),
        ((1,), (3,)),
        ((3, 1), (3,)),
        ((1, 3), (3,)),
        ((3, 2), (2,)),
        ((1, 2, 3), (3,)),
        ((1, 2, 3), (2, 3)),
        ((2, 1, 3), (2, 3)),
        ((2, 3, 4), ()),
        ((2, 1, 2), (1, 3, 2)),
    )

    for shape_pair, discontiguous in itertools.product(shapes, [False, True]):
        _input_shape, _other_shape = shape_pair

        yield OpSampleInput(
            op_input=make_tensor(
                _input_shape,
                dtype=dtype,
                device=device,
                discontiguous=discontiguous,
                low=op_info.input_low,
                high=op_info.input_high,
                exclude_value = op_info.input_exclude_value,
            ),
            op_args=(
                make_tensor(
                    _other_shape,
                    dtype=dtype,
                    device=device,
                    discontiguous=discontiguous,
                    low=op_info.other_low,
                    high=op_info.other_high,
                    exclude_value=op_info.other_exclude_value,
                ),
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_broadcasting_and_discontiguous_tensor_inputs",
        )


def _generate_binary_op_scalar_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate scalar-tensor/tensor-scalar/scalar-scalar inputs per support flags.

    Args:
        op_info: Operator metadata (includes Python scalar support flags).
        dtype: Data type for generated tensors.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Sample inputs mixing scalars and tensors.
    """

    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    def _make_scalar(low, high, exclude_value=None):
        """make scalar in [low, high] range
        """
        if dtype.is_complex:
            return 1.0
        return make_func((), low=low, high=high, exclude_value=exclude_value).item()
    shapes = (
        (),
        (2,),
        (0, 2),
    )

    if op_info.supports_right_python_scalar:
        for shape in shapes:
            yield OpSampleInput(
                op_input=make_func(shape, low=op_info.input_low,
                                   high=op_info.input_high,exclude_value = op_info.input_exclude_value,),
                op_args=(_make_scalar(low=op_info.other_low, high = op_info.other_high,
                                      exclude_value = op_info.other_exclude_value,),),
                op_kwargs={},
                sample_name=f"{op_info.name}_tensorxscalar_inputs",
            )

    if op_info.supports_left_python_scalar:
        for shape in shapes:
            yield OpSampleInput(
                op_input=_make_scalar(low=op_info.input_low,
                                      high = op_info.input_high, exclude_value = op_info.input_exclude_value,),
                op_args=(make_func(shape, low=op_info.other_low,
                                   high=op_info.other_high, exclude_value = op_info.other_exclude_value,),),
                op_kwargs={},
                sample_name=f"{op_info.name}_scalarxtensor_inputs",
            )

    if op_info.supports_both_python_scalar:
        yield OpSampleInput(
            op_input=_make_scalar(low=op_info.input_low,
                                  high = op_info.input_high, exclude_value = op_info.input_exclude_value,),
            op_args=(_make_scalar(low=op_info.other_low,
                                  high = op_info.other_high, exclude_value = op_info.other_exclude_value,),),
            op_kwargs={},
            sample_name=f"{op_info.name}_scalarxscalar_inputs",
        )

def _generate_binary_op_extremal_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor inputs with extremal value for stress testing.

    Args:
        op_info: Operator metadata (includes Python scalar support flags).
        dtype: Data type for generated tensors.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering extremal value.
    """
    # inf and nan are unsupported on Ascend910 devices.
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    S = SMALL_DIM_SIZE
    if (not op_info.input_exclude_value or np.inf not in op_info.input_exclude_value) and (
        not op_info.other_exclude_value or np.inf not in op_info.other_exclude_value):
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((S, S), np.inf), dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(np.full((S, S), np.inf), dtype=dtype, device=device),),
            op_kwargs={},
            sample_name=op_info.name,
        )
    if (not op_info.input_exclude_value or -np.inf not in op_info.input_exclude_value) and (
            not op_info.other_exclude_value or -np.inf not in op_info.other_exclude_value):
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((S, S), -np.inf), dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(np.full((S, S), -np.inf), dtype=dtype, device=device),),
            op_kwargs={},
            sample_name=op_info.name,
        )
    if (not op_info.input_exclude_value or np.nan not in op_info.input_exclude_value) and (
            not op_info.other_exclude_value or np.nan not in op_info.other_exclude_value):
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((S, S), np.nan), dtype=dtype, device=device),
            op_args=(make_tensor_with_np_array(np.full((S, S), np.nan), dtype=dtype, device=device),),
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_reference_inputs_binary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate comprehensive reference inputs for a binary op.

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
    yield from _generate_binary_op_tensors_sample_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with small value
    if not op_info.disable_small_value_tensor_inputs and dtype != ms.bool_:
        yield from _generate_binary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if not op_info.disable_large_value_tensor_inputs and dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_binary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # broadcasting tensors and contiguous or discontiguous tensors
    if not op_info.disable_broadcasting_and_discontiguous_tensor_inputs:
        yield from _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func(
            op_info, dtype, device, **kwargs)
    # scalarxtensor, tensorxscalar and scalarxscalar
    if not op_info.disable_scalar_inputs:
        yield from _generate_binary_op_scalar_inputs_func(op_info, dtype, device, **kwargs)
    # tensor with extremal value
    if not op_info.disable_extremal_value_tensor_inputs and (dtype.is_floating_point or dtype.is_complex):
        yield from _generate_binary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)


# op_dynamic_inputs_func for ops
def dynamic_inputs_binary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate dynamic-shape/rank inputs for a binary op.

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
        # dyncamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high,exclude_value = op_info.input_exclude_value),
                    op_args=(make_func(shape=(2, 3), low=op_info.other_low,
                                       high=op_info.other_high, exclude_value = op_info.other_exclude_value),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value = op_info.input_exclude_value),
                    op_args=(make_func(shape=(4, 5), low=op_info.other_low,
                                       high=op_info.other_high, exclude_value = op_info.other_exclude_value),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dyncamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value = op_info.input_exclude_value),
                    op_args=(make_func(shape=(2, 3), low=op_info.other_low,
                                       high=op_info.other_high, exclude_value = op_info.other_exclude_value),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value = op_info.input_exclude_value),
                    op_args=(make_func(shape=(2, 3, 4), low=op_info.other_low,
                                       high=op_info.other_high, exclude_value = op_info.other_exclude_value),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


# op_error_inputs_func for ops
def error_inputs_binary_op_common_func(op_error_inputs_func = None):
    """Wrap or synthesize error-case inputs for a binary op.

    Args:
        op_error_inputs_func: Optional downstream provider to extend/override
            default error cases.

    Returns:
        Callable[..., Generator[OpErrorInput, None, None]]: A generator factory
            that yields error samples and expected exception types.
    """
    def _error_inputs_binary_op_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
        if op_error_inputs_func is not None:
            yield from op_error_inputs_func(op_info, dtype, device)

        S = SMALL_DIM_SIZE
        if not op_info.supports_left_python_scalar:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=2,
                    op_args=(make_tensor(shape=(S,), dtype=ms.int64, device=device),),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info=f'left_python_scalar is not supported for {op_info.name}',
            )
        if not op_info.supports_right_python_scalar:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_tensor(shape=(S,), dtype=ms.int64, device=device),
                    op_args=(2,),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info=f'right_python_scalar is not supported for {op_info.name}',
            )
        if not op_info.supports_both_python_scalar:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=1,
                    op_args=(2,),
                    op_kwargs={},
                    sample_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info=f'both_python_scalar is not supported for {op_info.name}',
            )

    return _error_inputs_binary_op_func


class BinaryOpInfo(OpInfo):
    """Operator meta information for binary operations.
    These operations have two tensors as input and one tensor as output usually.
    And they may have the following characteristics:
      - they are elementwise operations
      - the output shape is determined by the broadcasted input shapes usually.
      - they may have keyword arguments.

    Extra attributes:
      - support_tensor_type_promotion: Whether to support tensors' type promotion.
      - supports_left_python_scalar: Whether to support left python scalar.
      - supports_right_python_scalar: Whether to support right python scalar.
      - supports_both_python_scalar: Whether to support both python scalar.
      - op_error_inputs_func: Function that generates error inputs for the op.
      - disable_small_value_tensor_inputs: Whether to disable small value tensor inputs.
      - disable_large_value_tensor_inputs: Whether to disable large value tensor inputs.
      - disable_broadcasting_and_discontiguous_tensor_inputs: Whether to disable broadcasting and
        discontiguous tensor inputs.
      - disable_scalar_inputs: Whether to disable scalar inputs.
      - disable_extremal_value_tensor_inputs: Whether to disable extremal value tensor inputs.
    """
    def __init__(
            self,
            name: str,
            *,
            op_basic_reference_inputs_func: Optional[Callable] = basic_reference_inputs_binary_op_common_func,
            op_extra_reference_inputs_func: Optional[Callable] = extra_reference_inputs_binary_op_common_func,
            op_dynamic_inputs_func: Optional[Callable] = dynamic_inputs_binary_op_common_func,
            op_error_inputs_func: Optional[Callable] = None,
            support_tensor_type_promotion: Optional[bool] = True,
            supports_left_python_scalar: Optional[bool] = True,
            supports_right_python_scalar: Optional[bool] = True,
            supports_both_python_scalar: Optional[bool] = True,
            disable_small_value_tensor_inputs: Optional[bool] = False,
            disable_large_value_tensor_inputs: Optional[bool] = False,
            disable_broadcasting_and_discontiguous_tensor_inputs: Optional[bool] = False,
            disable_scalar_inputs: Optional[bool] = False,
            disable_extremal_value_tensor_inputs: Optional[bool] = False,
            **kwargs,
    ):
        super().__init__(
            name,
            op_basic_reference_inputs_func=op_basic_reference_inputs_func,
            op_extra_reference_inputs_func=op_extra_reference_inputs_func,
            op_dynamic_inputs_func=op_dynamic_inputs_func,
            **kwargs,
        )
        self.support_tensor_type_promotion = support_tensor_type_promotion
        self.supports_left_python_scalar = supports_left_python_scalar
        self.supports_right_python_scalar = supports_right_python_scalar
        self.supports_both_python_scalar = supports_both_python_scalar
        self.op_error_inputs_func = error_inputs_binary_op_common_func(op_error_inputs_func)
        self.disable_small_value_tensor_inputs = disable_small_value_tensor_inputs
        self.disable_large_value_tensor_inputs = disable_large_value_tensor_inputs
        self.disable_broadcasting_and_discontiguous_tensor_inputs = disable_broadcasting_and_discontiguous_tensor_inputs
        self.disable_scalar_inputs = disable_scalar_inputs
        self.disable_extremal_value_tensor_inputs = disable_extremal_value_tensor_inputs
        self.input_low = None if not self.domain else self.domain[0][0]
        self.input_high = None if not self.domain else self.domain[0][1]
        self.other_low = None if not self.domain else self.domain[1][0]
        self.other_high = None if not self.domain else self.domain[1][1]
        self.input_exclude_value = None if not self.exclude_value else self.exclude_value[0]
        self.other_exclude_value = None if (not self.exclude_value) or len(self.exclude_value) <= 1\
            else self.exclude_value[1]


# basic op_basic_reference_inputs_func for unary ops
def basic_reference_inputs_unary_op_common_func(
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

    shapes = (
        (),
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
        _input = make_func(input_shape, low=op_info.input_low,
                           high=op_info.input_high, exclude_value = op_info.input_exclude_value)

        yield OpSampleInput(
            _input,
            op_args=(),
            sample_name=op_info.name,
        )


# op_extra_reference_inputs_func for unary ops
def _generate_unary_op_tensors_sample_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate single-tensor sample inputs for a unary op.

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
        (1, 0, 3),   # empty tensors
        (),          # scalar tensors
        (20,),       # 1D tensors with small size
        (812,),      # 1D tensors with medium size
        (1029, 917), # 2D tensors with large size
    )

    for shape in shapes:
        yield OpSampleInput(
            op_input=make_tensor(
                shape,
                dtype,
                device=device,
                low=op_info.input_low,
                high=op_info.input_high,
                exclude_value=op_info.input_exclude_value
            ),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_tensor_inputs",
        )


def _generate_unary_op_small_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate single-tensor inputs with small values for edge cases.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering representative small float/int/unsigned values.
    """
    _uint_values = (0, 1, 31, 63, 128, 195, 254)
    _int_values = (0, -1, 1, -63, 63, -127, 127, -128)
    _float_vals = (
        0.0,
        -0.0,
        -1.0,
        1.0,
        -1.25e-1,
        1.25e-1,
        -1e-3,
        1e-3,
        -math.pi / 2,
        math.pi / 2,
        -math.pi + 1e-5,
        math.pi - 1e-5,
        -math.pi,
        math.pi,
        -math.pi - 1e-5,
        math.pi + 1e-5,
    )

    def _filter_range_values(values, low, high, exclude_value = None):
        return list(filter(lambda x: (low is None or x > low)
                             and (high is None or x < high) and (
                not exclude_value or x not in exclude_value ),values))
    if dtype.is_floating_point:
        values = _filter_range_values(_float_vals, op_info.input_low,
                                      op_info.input_high, op_info.input_exclude_value)
    elif dtype.is_complex:
        values = [complex(r, i) for r, i in itertools.product(_float_vals, _float_vals)]
    elif dtype in dtypes_integral and dtype != ms.bool_:
        if dtype in dtypes_extra_uint or dtype == ms.uint8:
            values = _filter_range_values(_uint_values, op_info.input_low,
                                          op_info.input_high, op_info.input_exclude_value)
        else:
            values = _filter_range_values(_int_values, op_info.input_low,
                                          op_info.input_high,op_info.input_exclude_value )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}!")

    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.array(values), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_small_value_tensor_inputs",
    )


def _generate_unary_op_large_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate single-tensor inputs with large values for stress testing.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering large-magnitude float/complex/integer values.
    """
    _int_values = (-1023, 1023, -10922, 10922)
    _fp16_values = (-521, 521, -996.9, 996.9, -13141.5, 13141.5)
    _fp32_values = _fp16_values + (-5062757.7, 5062757.7, -1e20, 1e20)

    def _filter_range_values(values, low, high, exclude_value=None):
        return list(filter(lambda x: (low is None or x > low)
                            and (high is None or x < high) and (
                            not exclude_value or x not in exclude_value), values))

    if dtype == ms.float16:
        values = _filter_range_values(_fp16_values, op_info.input_low,
                                      op_info.input_high, op_info.input_exclude_value)
    elif dtype.is_floating_point:
        values = _filter_range_values(_fp32_values, op_info.input_low,
                                      op_info.input_high, op_info.input_exclude_value)
    elif dtype.is_complex:
        values = [complex(r, i) for r, i in itertools.product(_fp32_values, _fp32_values)]
    elif dtype in (ms.int16, ms.int32, ms.int64):
        values = _filter_range_values(_int_values, op_info.input_low,
                                      op_info.input_high, op_info.input_exclude_value)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}!")

    yield OpSampleInput(
        op_input=make_tensor_with_np_array(np.array(values), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_large_value_tensor_inputs",
    )


def _generate_unary_op_discontiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate contiguous and discontiguous tensor inputs for unary ops.

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

    for shape, discontiguous in itertools.product(shapes, [False, True]):
        yield OpSampleInput(
            op_input=make_tensor(
                shape,
                dtype=dtype,
                device=device,
                discontiguous=discontiguous,
                low=op_info.input_low,
                high=op_info.input_high,
                exclude_value = op_info.input_exclude_value
            ),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_discontiguous_tensor_inputs",
        )


def _generate_unary_op_extremal_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor with extremal value for a unary op.

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

    S = SMALL_DIM_SIZE
    if not op_info.input_exclude_value or np.inf not in op_info.input_exclude_value:
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((S, S), np.inf), dtype=dtype, device=device),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )
    if not op_info.input_exclude_value or -np.inf not in op_info.input_exclude_value:
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((S, S), -np.inf), dtype=dtype, device=device),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )
    if not op_info.input_exclude_value or np.nan not in op_info.input_exclude_value:
        yield OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((S, S), np.nan), dtype=dtype, device=device),
            op_args=(),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_reference_inputs_unary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate comprehensive reference inputs for a unary op.

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
    if not op_info.disable_small_value_tensor_inputs and dtype != ms.bool_:
        yield from _generate_unary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if not op_info.disable_large_value_tensor_inputs and dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_unary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # contiguous or discontiguous tensors
    if not op_info.disable_discontiguous_tensor_inputs:
        yield from _generate_unary_op_discontiguous_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensor with extremal value
    if not op_info.disable_extremal_value_tensor_inputs and (dtype.is_floating_point or dtype.is_complex):
        yield from _generate_unary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)


# op_dynamic_inputs_func for unary ops
def dynamic_inputs_unary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate dynamic-shape/rank inputs for a unary op.

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
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
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
                    op_input=make_func(shape=(2, 3), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4), low=op_info.input_low,
                                       high=op_info.input_high, exclude_value=op_info.input_exclude_value),
                    op_args=(),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


class UnaryOpInfo(OpInfo):
    """Operator meta information for unary operations.
    These operations have one tensor as input and one tensor as output usually.
    And they may have the following characteristics:
      - they are elementwise operations
      - the output shape is determined by the input shape usually.
      - they may have keyword arguments.

    Extra attributes:
      - disable_small_value_tensor_inputs: Whether to disable small value tensor inputs.
      - disable_large_value_tensor_inputs: Whether to disable large value tensor inputs.
      - disable_discontiguous_tensor_inputs: Whether to disable discontiguous tensor inputs.
      - disable_extremal_value_tensor_inputs: Whether to disable extremal value tensor inputs.
    """
    def __init__(
            self,
            name: str,
            *,
            op_basic_reference_inputs_func: Optional[Callable] = basic_reference_inputs_unary_op_common_func,
            op_extra_reference_inputs_func: Optional[Callable] = extra_reference_inputs_unary_op_common_func,
            op_dynamic_inputs_func: Optional[Callable] = dynamic_inputs_unary_op_common_func,
            op_error_inputs_func: Optional[Callable] = None,
            disable_small_value_tensor_inputs: Optional[bool] = False,
            disable_large_value_tensor_inputs: Optional[bool] = False,
            disable_discontiguous_tensor_inputs: Optional[bool] = False,
            disable_extremal_value_tensor_inputs: Optional[bool] = False,
            **kwargs,
    ):
        super().__init__(
            name,
            op_basic_reference_inputs_func=op_basic_reference_inputs_func,
            op_extra_reference_inputs_func=op_extra_reference_inputs_func,
            op_dynamic_inputs_func=op_dynamic_inputs_func,
            **kwargs,
        )
        self.disable_small_value_tensor_inputs = disable_small_value_tensor_inputs
        self.disable_large_value_tensor_inputs = disable_large_value_tensor_inputs
        self.disable_discontiguous_tensor_inputs = disable_discontiguous_tensor_inputs
        self.disable_extremal_value_tensor_inputs = disable_extremal_value_tensor_inputs
        self.input_low = None if not self.domain else self.domain[0]
        self.input_high = None if not self.domain else self.domain[1]
        self.input_exclude_value = self.exclude_value


def basic_reference_inputs_reduction_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate basic test inputs for reduction operators.

    Generates various dim and keepdim combinations, similar to PyTorch's
    _generate_reduction_kwargs.

    Args:
        op_info: OpInfo object for the reduction operator.
        dtype: Data type of the tensors.
        device: Device to create tensors on.
        **kwargs: Additional keyword arguments.

    Yields:
        OpSampleInput: Sample input configurations.
    """
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )
    # Generate tensors of different dimensions
    shapes = [
        (),
        (S,),
        (M, S),
        (S, S, L),
        (5, 3, 4, 2),
    ]

    for shape in shapes:
        ndim = len(shape)

        # Default: full reduction
        yield OpSampleInput(
            op_input=make_func(shape),
            op_kwargs={},
            sample_name=f'{op_info.name}_default_dim{ndim}'
        )

        # dim=None with keepdim=True
        yield OpSampleInput(
            op_input=make_func(shape),
            op_kwargs={'dim': None, 'keepdim': True},
            sample_name=f'{op_info.name}_default_keepdim_dim{ndim}'
        )

        yield OpSampleInput(
            op_input=make_func(shape),
            op_kwargs={'dim': 0, 'keepdim': True},
            sample_name=f'{op_info.name}_dim0_keepdim_dim{ndim}'
        )

        yield OpSampleInput(
            op_input=make_func(shape),
            op_kwargs={'dim': -1, 'keepdim': False},
            sample_name=f'{op_info.name}_dim_last_dim{ndim}'
        )

        if ndim > 2:
            yield OpSampleInput(
                op_input=make_func(shape),
                op_kwargs={'dim': 1, 'keepdim': True},
                sample_name=f'{op_info.name}_dim1_dim{ndim}'
            )

            if ndim >= 4:
                yield OpSampleInput(
                    op_input=make_func(shape),
                    op_kwargs={'dim': 3, 'keepdim': False},
                    sample_name=f'{op_info.name}_dim3_dim{ndim}'
                )

        if getattr(op_info, 'supports_multiple_dims', True):
            # Test all dimensions
            yield OpSampleInput(
                op_input=make_func(shape),
                op_kwargs={
                    'dim': tuple(range(ndim)),
                    'keepdim': False
                },
                sample_name=f'{op_info.name}_dim_all_dim{ndim}'
            )

            # Test partial multi-dim reduction: first and last
            if ndim > 1:
                yield OpSampleInput(
                    op_input=make_func(shape),
                    op_kwargs={'dim': (0, -1), 'keepdim': True},
                    sample_name=f'{op_info.name}_dim_0_last_dim{ndim}'
                )

                if ndim >= 4:
                    yield OpSampleInput(
                        op_input=make_func(shape),
                        op_kwargs={'dim': (0, 2), 'keepdim': False},
                        sample_name=f'{op_info.name}_dim_0_2_dim{ndim}'
                    )
                    yield OpSampleInput(
                        op_input=make_func(shape),
                        op_kwargs={'dim': (1, 3), 'keepdim': True},
                        sample_name=f'{op_info.name}_dim_1_3_dim{ndim}'
                    )


def _generate_reduction_op_discontiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate contiguous and discontiguous tensor inputs for reduction ops.

    Reference: PyTorch's test_noncontiguous_* tests in test_reductions.py:
    - test_noncontiguous_innermost: reducing along noncontiguous innermost dimension
    - test_noncontiguous_outermost: reducing along noncontiguous outermost dimension
    - test_noncontiguous_all: reducing all dimensions of a noncontiguous tensor

    Args:
        op_info: OpInfo object for the reduction operator.
        dtype: Data type of the tensors.
        device: Device to create tensors on.
        **kwargs: Additional keyword arguments.

    Yields:
        OpSampleInput: Sample input configurations for noncontiguous tensors.
    """
    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )

    # 1. Noncontiguous innermost dimension (dim=-1)
    yield OpSampleInput(
        op_input=make_func((10, 10), discontiguous=True),
        op_kwargs={'dim': -1},
        sample_name=f'{op_info.name}_discontiguous_innermost'
    )

    # 2. Noncontiguous outermost dimension (dim=0)
    yield OpSampleInput(
        op_input=make_func((10, 10), discontiguous=True),
        op_kwargs={'dim': 0},
        sample_name=f'{op_info.name}_discontiguous_outermost'
    )

    # 3. Noncontiguous all dimensions (default reduction)
    yield OpSampleInput(
        op_input=make_func((5, 5, 5), discontiguous=True),
        op_kwargs={},
        sample_name=f'{op_info.name}_discontiguous_default'
    )

    # 4. Noncontiguous with keepdim=True
    yield OpSampleInput(
        op_input=make_func((10, 10), discontiguous=True),
        op_kwargs={'dim': 1, 'keepdim': True},
        sample_name=f'{op_info.name}_discontiguous_keepdim'
    )

    # 5. Noncontiguous with multiple dims (if supported)
    if getattr(op_info, 'supports_multiple_dims', True):
        yield OpSampleInput(
            op_input=make_func((5, 5, 5), discontiguous=True),
            op_kwargs={'dim': (0, 2), 'keepdim': False},
            sample_name=f'{op_info.name}_discontiguous_multi_dim'
        )


def extra_reference_inputs_reduction_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate additional edge-case test inputs for reduction operators.

    Reference: PyTorch's test_ref_* edge case tests in test_reductions.py:
    - test_ref_duplicate_values: numerical stability with repeated values
    - test_ref_extremal_values: nan, inf, -inf handling
    - test_ref_large_input_1D/2D: large tensor stability and parallelism
    - test_noncontiguous_*: noncontiguous tensor inputs

    Args:
        op_info: OpInfo object for the reduction operator.
        dtype: Data type of the tensors.
        device: Device to create tensors on.
        **kwargs: Additional keyword arguments.

    Yields:
        OpSampleInput: Sample input configurations for edge cases.
    """
    S = SMALL_DIM_SIZE
    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )

    # Empty tensors
    yield OpSampleInput(
        op_input=make_func((0, S)),
        op_kwargs={'dim': 0},
        sample_name=f'{op_info.name}_empty_dim0'
    )

    yield OpSampleInput(
        op_input=make_func((S, 0)),
        op_kwargs={'dim': 1},
        sample_name=f'{op_info.name}_empty_dim1'
    )

    # Single element tensors
    yield OpSampleInput(
        op_input=make_func((1,)),
        op_kwargs={},
        sample_name=f'{op_info.name}_single_element_1d_default'
    )

    yield OpSampleInput(
        op_input=make_func((1, 1, 1)),
        op_kwargs={'dim': 1},
        sample_name=f'{op_info.name}_single_element_3d'
    )

    # Large dimension (only meaningful for operators supporting multi-dim reductions)
    if getattr(op_info, 'supports_multiple_dims', True):
        yield OpSampleInput(
            op_input=make_func((S, S, S, S, S)),
            op_kwargs={'dim': (1, 3), 'keepdim': True},
            sample_name=f'{op_info.name}_5d_multi_dim'
        )

    # Duplicate values test - numerical stability
    t = make_func((4, 4))
    yield OpSampleInput(
        op_input=t,
        op_kwargs={},
        sample_name=f'{op_info.name}_duplicate_values_default'
    )

    yield OpSampleInput(
        op_input=make_func((4, 4)),
        op_kwargs={'dim': 0},
        sample_name=f'{op_info.name}_duplicate_values_dim0'
    )

    yield OpSampleInput(
        op_input=make_func((4, 4)),
        op_kwargs={'dim': 1},
        sample_name=f'{op_info.name}_duplicate_values_dim1'
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
                op_kwargs={},
                sample_name=f'{op_info.name}_extremal_{name_suffix}'
            )

    # Large input tests - numerical stability
    LARGE_1D_SIZE = 2 ** 20
    yield OpSampleInput(
        op_input=make_func((LARGE_1D_SIZE,)),
        op_kwargs={},
        sample_name=f'{op_info.name}_large_1d_stability'
    )

    LARGE_2D_SIZE = 2 ** 16
    yield OpSampleInput(
        op_input=make_func((32, LARGE_2D_SIZE)),
        op_kwargs={'dim': 1},
        sample_name=f'{op_info.name}_large_2d_stability'
    )

    # Noncontiguous tensor tests
    if not getattr(op_info, 'disable_discontiguous_tensor_inputs', False):
        yield from _generate_reduction_op_discontiguous_tensor_inputs_func(
            op_info, dtype, device, **kwargs
        )


def dynamic_inputs_reduction_op_common_func(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    """Generate dynamic shape/rank test inputs for reduction operators.

    Args:
        op_info: OpInfo object for the reduction operator.
        dtype: Data type of the tensors.
        device: Device to create tensors on.
        **kwargs: Additional keyword arguments including:
            - only_dynamic_shape: Only generate dynamic shape inputs.
            - only_dynamic_rank: Only generate dynamic rank inputs.

    Yields:
        OpDynamicInput: Dynamic input configurations.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None), dtype=dtype),
                op_kwargs={'dim': 1, 'keepdim': True},
                sample_name=f'{op_info.name}_dynamic_shape_compile'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 7, 9)),
                    op_kwargs={'dim': 1, 'keepdim': True},
                    sample_name=f'{op_info.name}_dynamic_shape_running_1'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4, 5)),
                    op_kwargs={'dim': 1, 'keepdim': True},
                    sample_name=f'{op_info.name}_dynamic_shape_running_2'
                ),
            )
        )

        # Dynamic shape with tuple dim (only for operators that support multiple dims)
        if getattr(op_info, 'supports_multiple_dims', True):
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=(None, None, None, None), dtype=dtype),
                    op_kwargs={'dim': (0, 2), 'keepdim': False},
                    sample_name=f'{op_info.name}_dynamic_shape_compile_tuple_dim'
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=(4, 5, 6, 7)),
                        op_kwargs={'dim': (0, 2), 'keepdim': False},
                        sample_name=f'{op_info.name}_dynamic_shape_running_tuple_1'
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=(2, 3, 4, 5)),
                        op_kwargs={'dim': (0, 2), 'keepdim': False},
                        sample_name=f'{op_info.name}_dynamic_shape_running_tuple_2'
                    ),
                )
            )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_kwargs={'dim': 0},
                sample_name=f'{op_info.name}_dynamic_rank_compile'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 7)),
                    op_kwargs={'dim': 0},
                    sample_name=f'{op_info.name}_dynamic_rank_running_1'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(3, 4, 5)),
                    op_kwargs={'dim': 0},
                    sample_name=f'{op_info.name}_dynamic_rank_running_2'
                ),
            )
        )


class ReductionOpInfo(OpInfo):
    """OpInfo for reduction operators.

    Provides default values for reduction-specific attributes:
    - supports_multiple_dims: Whether the operator supports reducing multiple dimensions (default: True)
    - promotes_int_to_float: Whether the operator promotes integral to floating point dtypes (default: False)
    """

    def __init__(
            self,
            name: str,
            *,
            op_basic_reference_inputs_func: Optional[Callable] = basic_reference_inputs_reduction_op_common_func,
            op_extra_reference_inputs_func: Optional[Callable] = extra_reference_inputs_reduction_op_common_func,
            op_dynamic_inputs_func: Optional[Callable] = dynamic_inputs_reduction_op_common_func,
            op_error_inputs_func: Optional[Callable] = None,
            supports_multiple_dims: bool = True,
            **kwargs,
    ):
        super().__init__(
            name,
            op_basic_reference_inputs_func=op_basic_reference_inputs_func,
            op_extra_reference_inputs_func=op_extra_reference_inputs_func,
            op_dynamic_inputs_func=op_dynamic_inputs_func,
            **kwargs,
        )
        self.supports_multiple_dims = supports_multiple_dims
