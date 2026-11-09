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
"""Exercise real argument conversions, including Tensor/sequence boundaries."""
# Direct private-helper calls intentionally cover the internal rules under test.
# pylint: disable=protected-access
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops._utils import arg_dtype_cast as cast
from mindspore.ops._utils import arg_handler as handler


@pytest.fixture(scope='module', autouse=True)
def cpu_pynative_context():
    """Run conversion kernels on CPU and restore the caller's context."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


@pytest.mark.parametrize(
    'data,target,expected',
    [
        (3, cast.DT_FLOAT_VAL, 3.0),
        (2, cast.DT_TUPLE_INT_VAL, (2,)),
        ([1, 2], cast.DT_TUPLE_INT_VAL, (1, 2)),
        (2, cast.DT_LIST_INT_VAL, [2]),
        ((1, 2), cast.DT_LIST_INT_VAL, [1, 2]),
        (True, cast.DT_TUPLE_BOOL_VAL, (True,)),
        ((1, 2), cast.DT_TUPLE_INT_VAL, (1, 2)),
    ],
)
def test_type_cast_scalar_and_sequence(data, target, expected):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Type cast scalar and sequence.
    Expectation: Conversions must preserve both sequence elements and container type.
    """
    result = cast.do_type_cast(data, target, 'TestOp')
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    'target,expected',
    [
        (cast.DT_TUPLE_INT_VAL, (2, -3)),
        (cast.DT_LIST_INT_VAL, [2, -3]),
    ],
)
def test_type_cast_int32_tensor_to_sequence(target, expected):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Type cast int32 tensor to sequence.
    Expectation: The int32-to-int64 compatibility path preserves signed elements.
    """
    result = cast.do_type_cast(ms.Tensor([2, -3], ms.int32), target, 'TestOp')
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize('data,dtype', [(True, ms.bool_), (3, ms.int32), (1.5, ms.float32)])
def test_type_cast_scalar_to_tensor_chooses_dtype(data, dtype):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Type cast scalar to tensor chooses dtype.
    Expectation: Bool must not accidentally follow the integer conversion branch.
    """
    result = cast.do_type_cast(data, cast.DT_TENSOR_VAL, 'TestOp')
    assert result.dtype == dtype
    assert result.shape == ()
    assert result.asnumpy().item() == data


@pytest.mark.parametrize('data', [(2, -3), [2, -3]])
def test_type_cast_sequence_to_tensor_preserves_values(data):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Type cast sequence to tensor preserves values.
    Expectation: Tuple and list arguments must produce equal signed tensor elements.
    """
    result = cast.do_type_cast(data, cast.DT_TENSOR_VAL, 'TestOp')
    assert result.shape == (2,)
    np.testing.assert_array_equal(result.asnumpy(), [2, -3])


@pytest.mark.parametrize(
    'data,dtype,target,expected',
    [
        (2.75, ms.float32, cast.DT_INT_VAL, 2),
        (-3, ms.int16, cast.DT_NUMBER_VAL, -3),
        (1.5, ms.float64, cast.DT_NUMBER_VAL, 1.5),
        (True, ms.bool_, cast.DT_BOOL_VAL, True),
    ],
)
def test_type_cast_tensor_to_python_number(data, dtype, target, expected):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Type cast tensor to python number.
    Expectation: Scalar conversions must apply the requested numeric type.
    """
    result = cast.do_type_cast(ms.Tensor(data, dtype), target, 'TestOp')
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    'data,target',
    [
        ('bad', cast.DT_TENSOR_VAL),
        (1.5, cast.DT_INT_VAL),
        ({'x': 1}, cast.DT_TUPLE_INT_VAL),
        (1, cast.DT_TYPE_VAL),
    ],
)
def test_type_cast_rejects_unsupported_conversions(data, target):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Type cast rejects unsupported conversions.
    Expectation: Unsupported conversion pairs must fail instead of silently coercing.
    """
    with pytest.raises(TypeError, match='TestOp'):
        cast.do_type_cast(data, target, 'TestOp')


@pytest.mark.parametrize('to_tuple', [True, False])
def test_normalize_int_sequence_converts_scalar_tensors(to_tuple):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Normalize int sequence converts scalar tensors.
    Expectation: Mixed integer/tensor dimensions retain values and requested container.
    """
    data = [2, ms.Tensor(-3, ms.int32), ms.Tensor(4, ms.int64)]
    result = cast.normalize_int_sequence('Reshape', 'shape', data, to_tuple)
    assert result == ((2, -3, 4) if to_tuple else [2, -3, 4])
    # Exact Python result types are part of this conversion contract.
    # pylint: disable-next=unidiomatic-typecheck
    assert type(result) is (tuple if to_tuple else list)


def test_normalize_int_sequence_leaves_invalid_elements_for_parser():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Normalize int sequence leaves invalid elements for parser.
    Expectation: An invalid element must retain the original argument for later validation.
    """
    data = [2, 'invalid']
    assert cast.normalize_int_sequence('Reshape', 'shape', data) is data


def test_normalize_int_sequence_rejects_floating_tensor():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Normalize int sequence rejects floating tensor.
    Expectation: Floating tensor dimensions must not be truncated to integers.
    """
    with pytest.raises(ValueError, match='integral tensor'):
        cast.normalize_int_sequence('Reshape', 'shape', [ms.Tensor(1.5, ms.float32)])


@pytest.mark.parametrize(
    'dtype,integral,floating',
    [
        (ms.int8, True, False),
        (ms.uint64, True, False),
        (ms.float16, False, True),
        (ms.float64, False, True),
        (ms.bool_, False, False),
        (ms.complex64, False, False),
    ],
)
def test_numeric_dtype_categories_do_not_accept_bool_or_complex(dtype, integral, floating):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Numeric dtype categories do not accept bool or complex.
    Expectation: Category tests protect scalar normalization from invalid casts.
    """
    assert cast.is_integral_mstype(dtype) is integral
    assert cast.is_floating_point_mstype(dtype) is floating


@pytest.mark.parametrize(
    'data,target,expected',
    [
        (True, cast.DT_BOOL_VAL, True),
        (3, cast.DT_BOOL_VAL, False),
        ('text', -999, False),
        ([1], cast.DT_LIST_INT_VAL, True),
    ],
)
def test_argument_instance_check_handles_bool_and_unknown_type(data, target, expected):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Argument instance check handles bool and unknown type.
    Expectation: Unknown type ids and bool mismatches must not be accepted.
    """
    assert cast.is_instance_of(data, target) is expected


@pytest.mark.parametrize(
    'convert',
    [
        handler.to_pair,
        handler.to_kernel_size,
        handler.to_strides,
        handler.to_rates,
        handler.to_dilations,
        handler.to_output_padding,
        handler.to_2d_paddings,
    ],
)
def test_argument_handlers_reject_non_numeric_values(convert):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Argument handlers reject non numeric values.
    Expectation: Handler errors identify both the operator and offending argument.
    """
    with pytest.raises(ValueError, match="TestOp.*size"):
        convert('TestOp', 'size', 'invalid')


@pytest.mark.parametrize('value,expected', [(3, (3, 3)), ((1, 1, 2, 3), (2, 3)), ([2, 3], [2, 3])])
def test_rate_handler_extracts_spatial_dimensions(value, expected):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Rate handler extracts spatial dimensions.
    Expectation: Four-dimensional dilation rates must drop the batch/channel dimensions.
    """
    assert handler.to_rates('ExtractImagePatches', 'rates', value) == expected


def test_generator_handler_only_converts_integer_arguments():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Generator handler only converts integer arguments.
    Expectation: Integer constants become int64 tensors; other operands retain identity.
    """
    tensor = ms.Tensor([7], ms.int32)
    result = handler.generator_handler('TestOp', 'inputs', (2, 1.5, tensor))
    assert isinstance(result, tuple)
    assert result[0].dtype == ms.int64
    assert result[0].asnumpy().item() == 2
    assert result[1] == 1.5
    assert result[2] is tensor


@pytest.mark.parametrize('data,dtype,expected', [(2, ms.int32, 2), (1.5, ms.float32, 1.5), (True, ms.bool_, True)])
def test_scalar_tensor_handler_preserves_python_scalar_type(data, dtype, expected):
    """Feature: Operator argument dtype conversion and normalization.
    Description: Scalar tensor handler preserves python scalar type.
    Expectation: Scalar conversion dispatch must preserve integer, float and bool types.
    """
    result = handler._scalar_tensor_to_scalar('TestOp', 'value', ms.Tensor(data, dtype))
    assert result == expected
    assert type(result) is type(expected)


def test_scalar_tensor_handler_rejects_non_scalar_tensor():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Scalar tensor handler rejects non scalar tensor.
    Expectation: A one-element vector must not be accepted as a zero-dimensional tensor.
    """
    with pytest.raises(ValueError, match='0-dim'):
        handler._scalar_tensor_to_scalar('TestOp', 'value', ms.Tensor([2], ms.int32))


def test_scalar_tensor_handler_rejects_complex_tensor():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Scalar tensor handler rejects complex tensor.
    Expectation: Unsupported complex scalar input must fail before a lossy cast.
    """
    with pytest.raises(TypeError, match='integral, floating-point and bool'):
        handler._scalar_tensor_to_scalar('TestOp', 'value', ms.Tensor(1 + 2j, ms.complex64))


def test_scalar_tensor_handler_passes_python_scalars_through():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Scalar tensor handler passes python scalars through.
    Expectation: Python scalars need no Tensor conversion.
    """
    assert handler._scalar_tensor_to_scalar('TestOp', 'value', 1.5) == 1.5


def test_tensor_to_number_converts_integer_tensor_to_float():
    """Feature: Operator argument dtype conversion and normalization.
    Description: Tensor to number converts integer tensor to float.
    Expectation: The numeric conversion helper promotes integer Tensor values to float.
    """
    result = cast.tensor_to_number(ms.Tensor(3, ms.int32), cast.DT_FLOAT_VAL, 'TestOp')
    assert result == 3.0
    assert isinstance(result, float)
