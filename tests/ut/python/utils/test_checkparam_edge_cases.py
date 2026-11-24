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
"""Boundary validation for tensor shapes, sparse metadata and model inputs."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import _checkparam as validator


@pytest.mark.parametrize('value,options,expected', [
    (2, {}, (2, 2, 2)),
    (2, {'ret_five': True, 'pad_value': 3}, (3, 3, 2, 2, 2)),
    ((2, 3, 4), {}, (2, 3, 4)),
    ((2, 3, 4), {'ret_five': True}, (1, 1, 2, 3, 4)),
    ((1, 1, 2, 3, 4), {'allow_five': True}, (2, 3, 4)),
    ((1, 1, 2, 3, 4), {'allow_five': True, 'ret_five': True}, (1, 1, 2, 3, 4)),
    ((0, 2, 3), {'greater_zero': False}, (0, 2, 3)),
    ((1, 2, 3), {'third_one': True, 'three_input': True}, (1, 2, 3)),
])
def test_3d_parameters_expand_and_validate(value, options, expected):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Three-dimensional parameters must preserve the requested padding format.
    Expectation: Three-dimensional parameters must preserve the requested padding format.
    """
    # Internal helpers are the unit coverage targets here.
    # pylint: disable=protected-access
    assert validator._check_3d_int_or_tuple('kernel', value, 'Conv3D', **options) == expected


@pytest.mark.parametrize('value,options,error', [
    ([1, 2, 3], {}, TypeError),
    ((1, 2), {}, ValueError),
    ((1, 1, 2, 3, 4), {}, ValueError),
    ((1, 1, 2, 3, 4), {'allow_five': True, 'three_input': True}, ValueError),
    ((2, 3, 4), {'third_one': True}, ValueError),
    ((True, 2, 3), {}, ValueError),
    ((1.5, 2, 3), {}, ValueError),
    ((0, 2, 3), {}, ValueError),
    ((-1, 2, 3), {'greater_zero': False}, ValueError),
])
def test_3d_parameters_reject_invalid_values(value, options, error):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Invalid length, depth, bool and non-positive dimensions must fail.
    Expectation: Invalid length, depth, bool and non-positive dimensions must fail.
    """
    # Internal helpers are the unit coverage targets here.
    # pylint: disable=protected-access
    with pytest.raises(error, match='kernel'):
        validator._check_3d_int_or_tuple('kernel', value, 'Conv3D', **options)


@pytest.mark.parametrize('check,value', [(validator.check_negative_float, -1.5),
                                        (validator.check_non_positive_float, -1.5),
                                        (validator.check_non_positive_float, 0.0)])
def test_negative_float_validators_accept_boundaries(check, value):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: The non-positive validator includes zero, the negative validator does not.
    Expectation: The non-positive validator includes zero, the negative validator does not.
    """
    assert check(value, 'value', 'TestOp') == value


@pytest.mark.parametrize('check,value,error', [
    (validator.check_negative_float, 0.0, ValueError),
    (validator.check_non_positive_float, 0.5, ValueError),
    (validator.check_negative_float, -1, TypeError),
    (validator.check_negative_float, True, TypeError),
    (validator.check_negative_float, float('nan'), ValueError),
    (validator.check_negative_float, float('-inf'), ValueError),
])
def test_negative_float_validators_reject_invalid_numbers(check, value, error):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Non-finite values and wrong numeric types cannot pass numeric validation.
    Expectation: Non-finite values and wrong numeric types cannot pass numeric validation.
    """
    with pytest.raises(error, match='value'):
        check(value, 'value', 'TestOp')


@pytest.mark.parametrize('value', [True, '1.5', 2])
def test_check_is_number_requires_exact_numeric_category(value):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Float validation must not accept bool, strings or integer arguments.
    Expectation: Float validation must not accept bool, strings or integer arguments.
    """
    with pytest.raises(TypeError, match='value'):
        validator.check_is_number(value, float, 'value', 'TestOp')


@pytest.mark.parametrize('shape,axes,expected', [
    ((1, 2, 1), 0, (2, 1)),
    ((1, 2, 1), -1, (1, 2)),
    ((1, 2, 1), [0, 0, -1], (2,)),
    ((1, 2, 1), (), (1, 2, 1)),
    ((), (), ()),
])
def test_squeeze_shape_preserves_unselected_singletons(shape, axes, expected):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Only selected singleton axes are removed, including negative axes.
    Expectation: Only selected singleton axes are removed, including negative axes.
    """
    assert validator.prepare_shape_for_squeeze(shape, axes) == expected


@pytest.mark.parametrize('axes,error,reason', [(3, ValueError, 'range'), (-4, ValueError, 'range'),
                                             ([0, 3], ValueError, 'range'), (1, ValueError, 'must be 1'),
                                             ('0', TypeError, 'axes')])
def test_squeeze_shape_rejects_invalid_axes(axes, error, reason):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Out-of-range and non-singleton axes must fail before shape construction.
    Expectation: Out-of-range and non-singleton axes must fail before shape construction.
    """
    with pytest.raises(error, match=reason):
        validator.prepare_shape_for_squeeze((1, 2, 1), axes)


@pytest.mark.parametrize('axes,expected', [(None, (0, 1, 2)), (-1, (2,)), ([-1, 0], (2, 0))])
def test_axis_validation_canonicalizes_negative_axes(axes, expected):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Axis normalization must preserve the supplied order.
    Expectation: Axis normalization must preserve the supplied order.
    """
    assert validator.check_axis_valid(axes, 3) == expected


@pytest.mark.parametrize('axes,error', [((0, -3), ValueError), ((3,), ValueError), (('x',), TypeError)])
def test_canonical_axes_reject_duplicates_and_invalid_members(axes, error):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Aliases of the same axis count as duplicates after canonicalization.
    Expectation: Aliases of the same axis count as duplicates after canonicalization.
    """
    with pytest.raises(error):
        validator.check_and_canonicalize_axes(axes, 3)


@pytest.mark.parametrize('axes', ['0', [1, 'x']])
def test_axis_type_validation_rejects_non_integers(axes):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Reduction axis validation must reject strings within containers too.
    Expectation: Reduction axis validation must reject strings within containers too.
    """
    with pytest.raises(TypeError, match='axis'):
        validator.check_axis_type(axes)


@pytest.mark.parametrize('shape,expected', [((3,), (3,)), (([2, 3],), (2, 3)), (((2, 3),), (2, 3)),
                                         ((2, 3), (2, 3))])
def test_reshape_shape_normalizes_calling_conventions(shape, expected):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Single list, single tuple and variadic dimensions produce the same shape.
    Expectation: Single list, single tuple and variadic dimensions produce the same shape.
    """
    assert validator.check_reshape_shp(shape) == expected


def test_reshape_shape_rejects_a_string():
    """
    Feature: Parameter, shape and sparse-input validation
    Description: A single non-shape argument must not be interpreted as dimensions.
    Expectation: A single non-shape argument must not be interpreted as dimensions.
    """
    with pytest.raises(TypeError, match='shape'):
        validator.check_reshape_shp(('2,3',))


@pytest.mark.parametrize('order,error', [(None, TypeError), ('A', ValueError)])
def test_flatten_order_rejects_invalid_values(order, error):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Flatten supports only explicit C/F ordering strings.
    Expectation: Flatten supports only explicit C/F ordering strings.
    """
    with pytest.raises(error, match='order'):
        validator.check_flatten_order(order)


@pytest.mark.parametrize('axes,expected', [(-1, 2), ([-1, 0], (2, 0)), ((0, 1), (0, 1))])
def test_swapaxes_validation_normalizes_axes(axes, expected):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Swapaxes accepts integer and sequence forms with negative axes.
    Expectation: Swapaxes accepts integer and sequence forms with negative axes.
    """
    assert validator.check_swapaxes_axis(axes, 3) == expected


@pytest.mark.parametrize('axes,error', [('bad', TypeError), ([0, 'bad'], TypeError), ([0, 3], ValueError)])
def test_swapaxes_validation_rejects_invalid_axes(axes, error):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Invalid members and out-of-range axes cannot pass swapaxes validation.
    Expectation: Invalid members and out-of-range axes cannot pass swapaxes validation.
    """
    with pytest.raises(error):
        validator.check_swapaxes_axis(axes, 3)


@pytest.mark.parametrize('path,error', [(1, TypeError), ('folder/', ValueError), ('folder\\', ValueError),
                                      ('bad name.ckpt', ValueError)])
def test_file_name_validation_rejects_directories_and_illegal_characters(path, error):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Checkpoint filenames must identify files and contain legal characters.
    Expectation: Checkpoint filenames must identify files and contain legal characters.
    """
    with pytest.raises(error):
        validator.check_file_name_by_regular(path, prim_name='Save')


def test_name_and_padding_validation():
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Invalid names and inconsistent padding must be rejected.
    Expectation: Invalid names and inconsistent padding must be rejected.
    """
    with pytest.raises(ValueError, match='illegal'):
        validator.check_str_and_none_by_regular('bad/name', prim_name='TestOp')
    assert validator.check_pad_value_by_mode('pad', 2, 'Conv') == 2
    assert validator.check_pad_value_by_mode('valid', 0, 'Conv') == 0
    with pytest.raises(ValueError, match='padding must be zero'):
        validator.check_pad_value_by_mode('same', 2, 'Conv')


@pytest.mark.parametrize('indptr,indices,values,shape,error,reason', [
    ((3,), (2,), (2,), (0, 3), ValueError, 'positive'),
    ((3,), (2,), (2,), (2.0, 3), TypeError, 'must be int'),
    ((3,), (0,), (0,), (2, 3), ValueError, 'positive'),
    ((3,), (7,), (7,), (2, 3), ValueError, 'too small'),
    ((3,), (1, 2), (2,), (2, 3), ValueError, 'indices must be a 1'),
    ((3, 1), (2,), (2,), (2, 3), ValueError, 'indptr must be a 1'),
    ((2,), (2,), (2,), (2, 3), ValueError, 'indptr must have length'),
    ((3,), (1,), (2,), (2, 3), ValueError, 'indices and values'),
    ((3,), (2,), (2, 1), (2, 3), ValueError, 'dimension'),
    ((3,), (2,), (2, 2), (2, 3, 3), ValueError, "shape"),
])
def test_csr_shape_validation_rejects_inconsistent_metadata(indptr, indices, values, shape, error, reason):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Sparse row pointers, ranks and trailing dense dimensions must agree.
    Expectation: Sparse row pointers, ranks and trailing dense dimensions must agree.
    """
    with pytest.raises(error, match=reason):
        validator.check_csr_tensor_shape(indptr, indices, values, shape)


@pytest.mark.parametrize('indptr,indices,values,shape', [((3,), (2,), (2,), (2, 3)),
                                                       ((0,), (0,), (0,), (0, 0)),
                                                       ((3,), (2,), (2, 3), (2, 3, 3))])
def test_csr_shape_validation_accepts_empty_and_dense_values(indptr, indices, values, shape):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Empty CSR tensors and valid dense value dimensions must remain supported.
    Expectation: Empty CSR tensors and valid dense value dimensions must remain supported.
    """
    assert validator.check_csr_tensor_shape(indptr, indices, values, shape) is None


@pytest.mark.parametrize('indices,values,shape,error,reason', [
    ((2, 2), (2,), (2, 3, 1), ValueError, 'length'),
    ((2, 2), (2,), (0, 3), ValueError, 'positive'),
    ((2, 2), (2,), (2.0, 3), TypeError, 'must be int'),
    ((7, 2), (7,), (2, 3), ValueError, 'too small'),
    ((2,), (2,), (2, 3), ValueError, '2-dimensional'),
    ((2, 2), (2, 1), (2, 3), ValueError, '1-dimensional'),
    ((1, 2), (2,), (2, 3), ValueError, 'indices.shape'),
    ((2, 3), (2,), (2, 3), ValueError, 'must be 2'),
])
def test_coo_shape_validation_rejects_inconsistent_metadata(indices, values, shape, error, reason):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: COO coordinates must have one row per value and two columns.
    Expectation: COO coordinates must have one row per value and two columns.
    """
    with pytest.raises(error, match=reason):
        validator.check_coo_tensor_shape(indices, values, shape)


@pytest.mark.parametrize('dtype', [ms.int16, ms.int32, ms.int64])
def test_sparse_index_dtype_accepts_supported_integers(dtype):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: All three supported signed index widths must remain accepted.
    Expectation: All three supported signed index widths must remain accepted.
    """
    assert validator.check_csr_tensor_dtype(dtype, dtype) is None
    assert validator.check_coo_tensor_dtype(dtype) is None


@pytest.mark.parametrize('indptr,indices', [(ms.float32, ms.int32), (ms.int32, ms.float32)])
def test_csr_index_dtype_rejects_floats(indptr, indices):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Both row pointers and column indices need integer storage.
    Expectation: Both row pointers and column indices need integer storage.
    """
    with pytest.raises(TypeError):
        validator.check_csr_tensor_dtype(indptr, indices)


def test_coo_index_dtype_rejects_float():
    """
    Feature: Parameter, shape and sparse-input validation
    Description: COO coordinates cannot use floating storage.
    Expectation: COO coordinates cannot use floating storage.
    """
    with pytest.raises(TypeError, match='indices'):
        validator.check_coo_tensor_dtype(ms.float32)


@pytest.mark.parametrize('bad_argument', ['indices', 'values', 'shape', 'indptr'])
def test_sparse_inputs_require_tensor_indices_values_and_tuple_shape(bad_argument):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Each sparse input is independently checked for the required type.
    Expectation: Each sparse input is independently checked for the required type.
    """
    args = {'indptr': ms.Tensor([0, 1], ms.int32), 'indices': ms.Tensor([0], ms.int32),
                'values': ms.Tensor([1.0], ms.float32), 'shape': (1, 1)}
    args[bad_argument] = [1]
    with pytest.raises(TypeError, match=bad_argument):
        validator.check_csr_tensor_input(**args)


@pytest.mark.parametrize('sequence', [(1,), (1, True), [1, 2.0], '12'])
def test_expand_sequence_rejects_wrong_length_and_non_int_members(sequence):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Sequence expansion must reject bools, floats and incorrect dimensions.
    Expectation: Sequence expansion must reject bools, floats and incorrect dimensions.
    """
    with pytest.raises(TypeError):
        validator.twice_sequence(sequence)


@pytest.mark.parametrize('data', [[1, (2, 3)], {'x': [1, 2], 'y': 3}, None])
def test_input_data_validation_recurses_through_containers(data):
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Nested input data may combine accepted values and optional None.
    Expectation: Nested input data may combine accepted values and optional None.
    """
    assert validator.check_input_data(data, data_class=(int, None)) is None


def test_input_data_validation_rejects_invalid_nested_value():
    """
    Feature: Parameter, shape and sparse-input validation
    Description: A nested invalid value must not be hidden by an outer dict/list.
    Expectation: A nested invalid value must not be hidden by an outer dict/list.
    """
    with pytest.raises(TypeError, match='input data'):
        validator.check_input_data({'x': [1, 'bad']}, data_class=int)


def test_input_data_validation_rejects_empty_array():
    """
    Feature: Parameter, shape and sparse-input validation
    Description: An array with a supported type must still contain data.
    Expectation: An array with a supported type must still contain data.
    """
    with pytest.raises(ValueError, match='empty'):
        validator.check_input_data(np.array([]), data_class=np.ndarray)


def test_model_input_and_output_validation():
    """
    Feature: Parameter, shape and sparse-input validation
    Description: Missing required values and invalid dataset types must be rejected.
    Expectation: Missing required values and invalid dataset types must be rejected.
    """
    assert validator.check_valid_input('x', 0, 'TestOp') == 0
    with pytest.raises(ValueError, match='None'):
        validator.check_valid_input('x', None, 'TestOp')
    assert not validator.check_input_dataset(1, dataset_type=str)
    assert validator.check_input_dataset('data', dataset_type=str)
    with pytest.raises(RuntimeError, match='output data'):
        validator.check_output_data(None)
