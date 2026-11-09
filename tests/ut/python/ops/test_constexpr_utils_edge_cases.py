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
"""Constant indexing, slice normalization and broadcast edge cases."""
# Direct private-helper calls intentionally cover the internal rules under test.
# pylint: disable=protected-access
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const


@pytest.mark.parametrize(
    'start,size,expected', [(None, 5, 0), (-8, 5, 0), (-2, 5, 3), (8, 5, 5), (2, None, 2), (-2, None, -2)]
)
def test_normalize_slice_start_clips_to_dimension(start, size, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Normalize slice start clips to dimension.
    Expectation: Negative indices normalize only when the dimension is known.
    """
    assert const.normalize_start(start, size) == expected


@pytest.mark.parametrize('stop,size,expected', [(None, 5, 5), (-8, 5, 0), (-2, 5, 3), (8, 5, 5), (2, None, 2)])
def test_normalize_slice_stop_clips_to_dimension(stop, size, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Normalize slice stop clips to dimension.
    Expectation: Stop positions must remain within a known dimension's bounds.
    """
    assert const.normalize_stop(stop, size) == expected


def test_normalize_slice_stop_requires_bound_for_dynamic_dimension():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Normalize slice stop requires bound for dynamic dimension.
    Expectation: An implicit stop cannot be inferred for an unknown dimension.
    """
    with pytest.raises(IndexError, match='dynamic'):
        const.normalize_stop(None, None)


@pytest.mark.parametrize(
    'index,expected', [(slice(None), (0, 5, 1)), (slice(-4, 99, 2), (1, 5, 2)), (slice(4, 0, -1), (4, 0, -1))]
)
def test_normalize_slice_handles_positive_and_negative_steps(index, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Normalize slice handles positive and negative steps.
    Expectation: Negative steps select the reverse normalization path.
    """
    assert const.normalize_slice(index, 5) == expected


@pytest.mark.parametrize('index,expected', [(slice(1, 5, 2), [1, 3]), (slice(4, 0, -2), [4, 2])])
def test_slice_to_element_list_preserves_step_direction(index, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Slice to element list preserves step direction.
    Expectation: Materialized indices must follow the slice's signed step.
    """
    assert const.transform_slice_to_ele_list(index, 5) == expected


def test_slice_to_element_list_rejects_empty_slice():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Slice to element list rejects empty slice.
    Expectation: An empty assignment index must be rejected explicitly.
    """
    with pytest.raises(IndexError, match='empty slice'):
        const.transform_slice_to_ele_list(slice(2, 2), 5)


@pytest.mark.parametrize(
    'index,expected',
    [
        (Ellipsis, (slice(None), slice(None), slice(None))),
        ((1, Ellipsis), (1, slice(None), slice(None))),
        ((Ellipsis, 1), (slice(None), slice(None), 1)),
        ((1, slice(None)), (1, slice(None))),
    ],
)
def test_ellipsis_expands_to_missing_dimensions(index, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Ellipsis expands to missing dimensions.
    Expectation: Ellipsis expansion must preserve explicitly indexed dimensions.
    """
    assert const.ellipsis2slice(index, (2, 3, 4)) == expected


def test_ellipsis_expansion_rejects_multiple_ellipses():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Ellipsis expansion rejects multiple ellipses.
    Expectation: Two ellipses cannot define a unique index expansion.
    """
    with pytest.raises(IndexError, match='more than one'):
        const.ellipsis2slice((Ellipsis, Ellipsis), (2, 3))


@pytest.mark.parametrize('index', [slice(None), (1, Ellipsis, slice(None))])
def test_setitem_index_accepts_basic_indices(index):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Setitem index accepts basic indices.
    Expectation: Basic slices and mixed basic-index tuples remain valid.
    """
    assert const.check_tensor_setitem_index(index) is True


@pytest.mark.parametrize('index', [None, (), ('bad',), 'bad'])
def test_setitem_index_rejects_unsupported_indices(index):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Setitem index rejects unsupported indices.
    Expectation: Invalid setitem indices must fail before tensor updates.
    """
    with pytest.raises(IndexError):
        const.check_tensor_setitem_index(index)


@pytest.mark.parametrize('element_type', [None, ms.int32])
def test_setitem_tensor_index_requires_boolean_dtype(element_type):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Setitem tensor index requires boolean dtype.
    Expectation: A tensor type used for this indexing path must be boolean.
    """
    with pytest.raises(TypeError, match='bool'):
        const.check_tensor_setitem_index(ms.TensorType(ms.int32), element_type)


def test_setitem_tensor_index_accepts_boolean_dtype():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Setitem tensor index accepts boolean dtype.
    Expectation: Boolean mask tensor metadata must pass index validation.
    """
    assert const.check_tensor_setitem_index(ms.TensorType(ms.bool_), ms.bool_) is True


@pytest.mark.parametrize('value,expected', [(-1, 4), (-5, 0), (4, 4), (True, True)])
def test_index_range_wraps_negative_integer_indices(value, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Index range wraps negative integer indices.
    Expectation: Bool masks must not be normalized as negative/positive integer indices.
    """
    result = const.check_range(value, 5)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize('value', [-6, 5])
def test_index_range_rejects_out_of_bounds_values(value):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Index range rejects out of bounds values.
    Expectation: Both integer index boundaries are checked.
    """
    with pytest.raises(IndexError, match='out of bounds'):
        const.check_range(value, 5)


@pytest.mark.parametrize('shapes,expected', [((), ()), (((2, 1), (1, 3), (3,)), (2, 3))])
def test_generate_broadcast_shape_combines_all_indices(shapes, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Generate broadcast shape combines all indices.
    Expectation: Broadcasting a tuple of index shapes must retain every dimension.
    """
    assert const.generate_broadcast_shape(shapes, 'getitem') == expected


@pytest.mark.parametrize('left,right', [(None, (2,)), ((2,), None), ((None,), (2,)), ((2,), (None,)), ((2, 3), (4, 3))])
def test_constexpr_broadcast_rejects_dynamic_and_incompatible_shapes(left, right):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Constexpr broadcast rejects dynamic and incompatible shapes.
    Expectation: Constant indexing requires known, broadcast-compatible shapes.
    """
    with pytest.raises(ValueError):
        const.get_broadcast_shape(left, right, 'getitem')


@pytest.mark.parametrize(
    'positions,fancy,expected',
    [
        ((1, 2), None, ((2, 3), (1, 2, 3, 1), (5, 2, 3, 7), 1)),
        ((0, 2), None, ((2, 3), (2, 3, 1, 1), (2, 3, 5, 7), 0)),
        ((1, 2), 2, ((2, 3), (1, 1, 2, 3), (5, 7, 2, 3), 2)),
    ],
)
def test_mixed_tensor_index_shapes_respect_fancy_index_position(positions, fancy, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Mixed tensor index shapes respect fancy index position.
    Expectation: Separated advanced indices move broadcast dimensions to the front.
    """
    assert (
        const.generate_index_info_from_tuple_of_mixed_tensors(positions, ((2, 1), (1, 3)), (5, 7), 'getitem', fancy)
        == expected
    )


@pytest.mark.parametrize('indices,values,error', [(0, 0, ValueError), (3, 2, ValueError)])
def test_assignment_value_size_rejects_empty_or_mismatched_updates(indices, values, error):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Assignment value size rejects empty or mismatched updates.
    Expectation: Assignment values must be scalar or match the number of indices.
    """
    with pytest.raises(error):
        const.check_indices_value_size(indices, values)


@pytest.mark.parametrize('indices,values', [(3, 1), (3, 3)])
def test_assignment_value_size_accepts_scalar_and_exact_size(indices, values):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Assignment value size accepts scalar and exact size.
    Expectation: Scalar updates may broadcast; non-scalar updates must match exactly.
    """
    assert const.check_indices_value_size(indices, values) == values


@pytest.mark.parametrize('origin,broadcast,expected', [((1, 3), (2, 4, 3), (2, 4, 1)), ((2, 1), (2, 5), (1, 5))])
def test_compute_multiples_retains_leading_broadcast_dimensions(origin, broadcast, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Compute multiples retains leading broadcast dimensions.
    Expectation: Tile factors must include dimensions introduced by broadcasting.
    """
    assert const.compute_multiples(origin, broadcast) == expected


@pytest.mark.parametrize('shape,flags,expected', [((2, 3), (True, False), (2,)), ((2, 3), (False, True, True), (2, 3))])
def test_filter_expanded_dims_matches_flags_to_trailing_shape(shape, flags, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Filter expanded dims matches flags to trailing shape.
    Expectation: New leading dimensions must not shift the flags for existing axes.
    """
    assert const.filter_expanded_dims(shape, flags) == expected


def test_filter_expanded_dims_rejects_too_few_flags():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Filter expanded dims rejects too few flags.
    Expectation: A shorter dimension mask cannot describe the input shape.
    """
    with pytest.raises(ValueError, match='broadcast'):
        const.filter_expanded_dims((2, 3), (True,))


@pytest.mark.parametrize(
    'left,right,expected',
    [
        (ms.int32, ms.float32, ms.float32),
        (ms.float64, ms.float32, ms.float64),
        (ms.complex64, ms.complex128, ms.complex128),
        (ms.float64, ms.complex64, ms.complex64),
        (ms.int32, ms.int32, ms.int32),
    ],
)
def test_binary_dtype_promotion_preserves_wider_numeric_type(left, right, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Binary dtype promotion preserves wider numeric type.
    Expectation: Mixed numeric operations must preserve the higher-priority result type.
    """
    assert const.promote_binary_dtype(left, right) == expected


def test_binary_dtype_promotion_rejects_unsupported_type():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Binary dtype promotion rejects unsupported type.
    Expectation: An unknown dtype must not silently select the other operand's type.
    """
    with pytest.raises(ValueError, match='not supported'):
        const.get_output_dtype(ms.string, ms.float32)


@pytest.mark.parametrize('sequence,number,expected', [([1, 2], 2, [1, 2, 1, 2]), ((1, 2), 0, ()), ((1, 2), -1, ())])
def test_sequence_repetition_preserves_container_and_empty_results(sequence, number, expected):
    """Feature: Constant indexing, shape and dtype validation.
    Description: Sequence repetition preserves container and empty results.
    Expectation: Zero/negative repetitions produce an empty sequence of the same type.
    """
    result = const.sequence_mul_int(sequence, number)
    assert result == expected
    assert type(result) is type(expected)


def test_sequence_repetition_requires_integer_count():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Sequence repetition requires integer count.
    Expectation: Fractional repetition counts must not be rounded or truncated.
    """
    with pytest.raises(TypeError, match='non-int'):
        const.sequence_mul_int((1, 2), 1.5)


def test_nested_tensor_conversion_preserves_values():
    """Feature: Constant indexing, shape and dtype validation.
    Description: Nested tensor conversion preserves values.
    Expectation: Nested tensors must be converted at every list level.
    """
    result = const._deep_tensor_to_nparray([ms.Tensor([1, 2]), [ms.Tensor([3])]])
    np.testing.assert_array_equal(result[0], [1, 2])
    np.testing.assert_array_equal(result[1][0], [3])
    assert const.deep_tuple([1, [2, (3,)]]) == (1, (2, (3,)))
