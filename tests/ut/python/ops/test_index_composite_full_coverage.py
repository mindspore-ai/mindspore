# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Numerical and contract tests for Python indexing and multitype helpers."""
# Direct private-helper calls intentionally cover the internal rules under test.
# pylint: disable=protected-access
import importlib
import operator

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore.common import dtype as dt
from mindspore.ops.composite.multitype_ops import _compile_utils as cu
from mindspore.ops.composite.multitype_ops import _constexpr_utils as co


@pytest.fixture(autouse=True)
def cpu_context():
    old = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**old)


def tensor(value, dtype=ms.float32):
    return ms.Tensor(value, dtype)


def equal(actual, expected, dtype=None):
    assert isinstance(actual, ms.Tensor)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    if dtype is not None:
        assert actual.dtype == dtype
    np.testing.assert_allclose(actual.asnumpy(), expected, rtol=1e-6, atol=1e-6)


def np_index(index):
    if isinstance(index, ms.Tensor):
        return index.asnumpy()
    if isinstance(index, tuple):
        return tuple(np_index(item) for item in index)
    return index


@pytest.mark.parametrize(
    'index',
    [
        True,
        False,
        0,
        -1,
        slice(None),
        slice(1, 3),
        slice(0, 3, 2),
        None,
        Ellipsis,
        (),
        (slice(None), -1),
        (None, Ellipsis, 1),
        (True, slice(None)),
        tensor(1, ms.int32),
        tensor(True, ms.bool_),
        tensor(False, ms.bool_),
        tensor([0, 2], ms.int64),
        (tensor([0, 2], ms.int32), slice(1, 3)),
        (slice(None), tensor([1, 3], ms.int32)),
        (Ellipsis, tensor(1, ms.int64)),
    ],
)
def test_view_getitem_matches_numpy(index, request):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: View getitem matches numpy.
    Expectation: Each basic or advanced index selects the same float32 values and shape as NumPy.
    """
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = array[np_index(index)]
    advanced = (
        isinstance(index, bool)
        or (isinstance(index, ms.Tensor) and index.dtype == ms.bool_)
        or (isinstance(index, ms.Tensor) and index.ndim > 0)
        or (
            isinstance(index, tuple)
            and any(isinstance(x, bool) or (isinstance(x, ms.Tensor) and x.ndim > 0) for x in index)
        )
    )
    if advanced:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True, reason='CPU Index kernel is not registered; numerical advanced indexing cannot execute'
            )
        )
    equal(cu._tensor_getitem(tensor(array), index), expected, ms.float32)


@pytest.mark.parametrize(
    'index,error,message',
    [
        (3, IndexError, 'bounds'),
        (-4, IndexError, 'bounds'),
        ((0, 0, 0), IndexError, 'too many'),
        ('invalid', IndexError, 'Invalid'),
        (slice(None, None, 0), ValueError, 'positive'),
        (slice(None, None, -1), ValueError, 'positive'),
    ],
)
def test_view_getitem_invalid_index(index, error, message):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: View getitem invalid index.
    Expectation: Invalid indices raise the specified exception with the expected boundary-validation message.
    """
    with pytest.raises(error, match=message):
        cu._tensor_getitem(tensor(np.zeros((3, 4))), index)


@pytest.mark.parametrize('index', [0, slice(None)])
def test_scalar_view_index_rejected(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Scalar view index rejected.
    Expectation: Indexing a zero-dimensional tensor by a scalar raises the documented TypeError.
    """
    with pytest.raises(TypeError, match='0-dim'):
        cu._tensor_getitem(tensor(2), index)
    with pytest.raises(TypeError, match='0-dim'):
        cu._tensor_setitem(tensor(2), index, 5)


def test_view_index_metadata_and_scalar_slice():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: View index metadata and scalar slice.
    Expectation: Indexed-axis counts, tuple wrapping and tensor-valued slice bounds preserve the expected
        metadata and selected values.
    """
    assert (
        cu._count_indexed_dims(
            (None, Ellipsis, True, tensor(np.ones((2, 3)), ms.bool_), tensor([1], ms.int32), 0, slice(None))
        )
        == 5
    )
    for index, expected in [
        (True, True),
        (False, True),
        (tensor(True, ms.bool_), True),
        (tensor(1, ms.int32), False),
        (tensor([1], ms.int64), True),
        (None, False),
    ]:
        assert cu._get_need_index_prim(index, False) is expected
    source = tensor(np.arange(6).reshape(2, 3))
    equal(
        cu._do_slice(source, 1, slice(tensor(0, ms.int64), tensor(2, ms.int64), tensor(1, ms.int64)), 3),
        np.arange(6).reshape(2, 3)[:, :2],
    )
    assert cu._wrap_index_to_tuple([1, 2]) == ([1, 2],)
    assert cu._wrap_index_to_tuple([slice(None), None]) == (slice(None), None)
    assert cu._wrap_index_to_tuple((1,)) == (1,)
    assert cu._wrap_index_to_tuple([None] * 32) == ([None] * 32,)


@pytest.mark.parametrize('index', [False])
def test_view_false_setitem_noop(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: View false setitem noop.
    Expectation: False boolean assignment returns the original tensor and leaves all stored values unchanged.
    """
    source = tensor(np.arange(6).reshape(2, 3))
    assert cu._tensor_setitem(source, index, 9) is source
    equal(source, np.arange(6).reshape(2, 3))


def test_view_setitem_rejects_non_numeric_value_and_invalid_index():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: View setitem rejects non numeric value and invalid index.
    Expectation: Unsupported values, excess dimensions and invalid index kinds fail before assignment.
    """
    with pytest.raises(TypeError, match='type of value'):
        cu._tensor_setitem(tensor([1, 2]), 0, [4])
    with pytest.raises(IndexError, match='too many'):
        cu._tensor_setitem(tensor([1, 2]), (0, 0), 4)
    with pytest.raises(IndexError, match='Invalid'):
        cu._tensor_setitem(tensor([1, 2]), 'bad', 4)


@pytest.mark.parametrize('index', [-1, slice(1, 3), (slice(None), 1), (tensor(1, ms.int32),)])
@pytest.mark.xfail(strict=True, reason='CPU view inplace_copy does not update the original Tensor storage')
def test_cpu_view_setitem_updates_original_storage(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: CPU view setitem updates original storage.
    Expectation: The returned tensor and original CPU storage both contain the same updates as NumPy assignment.
    """
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = array.copy()
    expected[np_index(index)] = 19
    source = tensor(array)
    result = cu._tensor_setitem(source, index, 19)
    equal(result, expected)
    equal(source, expected)


@pytest.mark.parametrize(
    'name,index',
    [
        ('tensor_index_by_number', 1),
        ('tensor_index_by_number', True),
        ('tensor_index_by_tensor', tensor([-1, 0], ms.int32)),
        ('tensor_index_by_tensor', tensor([True, False, True], ms.bool_)),
        ('tensor_index_by_list', [0, -1]),
        ('tensor_index_by_list', [True, False, True]),
        ('tensor_index_by_list', [True, 0]),
    ],
)
def test_legacy_single_axis_getitem(name, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Legacy single axis getitem.
    Expectation: Legacy single-axis handlers return the same selection as NumPy for each supported index kind.
    """
    array = np.arange(12).reshape(3, 4)
    # A list containing ints and bools follows integer sequence conversion.
    # Exact Python result types are part of this conversion contract.
    # pylint: disable-next=unidiomatic-typecheck
    idx = [int(x) for x in index] if isinstance(index, list) and any(type(x) is int for x in index) else np_index(index)
    equal(getattr(cu, name)(tensor(array), index), array[idx])


@pytest.mark.parametrize(
    'name,index,message',
    [
        ('tensor_index_by_number', 1.5, 'int or bool'),
        ('tensor_index_by_number', False, "only support 'True'"),
        ('tensor_index_by_tensor', tensor([1.0]), 'int or bool'),
        ('tensor_index_by_tensor', tensor([[True]], ms.bool_), 'shape of index'),
        ('tensor_index_by_tensor', tensor(np.ones((1, 1, 1)), ms.bool_), 'dim of index'),
        ('tensor_index_by_list', [True], 'boolean dimension'),
        ('tensor_index_by_list', [], 'boolean dimension'),
    ],
)
def test_legacy_getitem_errors(name, index, message):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Legacy getitem errors.
    Expectation: Legacy index validation raises IndexError with the expected scenario-specific message.
    """
    with pytest.raises(IndexError, match=message):
        getattr(cu, name)(tensor(np.zeros((3, 4))), index)


@pytest.mark.parametrize(
    'index,expected,dtype',
    [
        ([0, 2], [0, 2], ms.int64),
        ([True, False], [True, False], ms.bool_),
        ([True, 2, False], [1, 2, 0], ms.int64),
        ([], [], ms.int64),
    ],
)
def test_list_index_conversion(index, expected, dtype):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List index conversion.
    Expectation: List indices become tensors with the expected values and boolean or integer dtype.
    """
    equal(cu._convert_list_index_to_tensor(index), expected, dtype)


def test_misc_index_shape_helpers():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Misc index shape helpers.
    Expectation: Stride, slice-mask and dimension helpers retain the expected metadata and reject unsupported
        tuple layouts.
    """
    assert cu._convert_list_index_to_tensor([slice(None)]) is None
    assert cu.get_stride_info_from_integer(-2) == ((-2,), (-1,), (1,))
    assert cu.get_slice_stride(slice(None), 4) == (0, 4, 1)
    assert cu.get_slice_stride(slice(tensor(1, ms.int32), tensor(4, ms.int32), tensor(2, ms.int32)), 4) == (1, 4, 2)
    assert cu.cal_tuple_slice_mask((2, 3, 4), (slice(None, 2), slice(1, None))) == (5, 6)
    assert cu.parse_check_slice_index(slice(2, 2), 4) is True
    assert cu.parse_check_slice_index(slice(None), 4) is False
    assert cu.tensor_index_by_tuple(tensor([1, 2]), ()).shape == (2,)
    assert (
        cu.judge_tuple_index_dim(tensor(np.zeros((2, 3))), (None, Ellipsis, True, tensor([True, False], ms.bool_)))
        is None
    )
    with pytest.raises(IndexError, match='dim of index'):
        cu.judge_tuple_index_dim(tensor([1]), (0, 0))
    equal(cu.reshape_with_check(tensor(np.arange(6)), tensor([2, 3], ms.int64)), np.arange(6).reshape(2, 3))
    equal(cu._tensor_index_transfer(tensor(2, ms.int64), (), (2, 3), (1, 1)), np.full((2, 3), 2))
    equal(cu._tensor_index_transfer(tensor([1, 2], ms.int64), (2,), (2, 2), (1, 2)), [[1, 2], [1, 2]])
    equal(cu._tensor_index_transfer(tensor(2, ms.int64), (), (0, 2), (1, 1)), np.empty((0, 2)))
    equal(cu.handle_empty_tensor((2, 3), tensor([1], ms.int32)), np.zeros((2, 3)), ms.int32)
    equal(cu.handle_empty_tensor((0, 3), tensor([1], ms.int32)), np.empty((0, 3)), ms.int32)


@pytest.mark.parametrize(
    'index', [tensor(1, ms.int32), tensor([-1, 0], ms.int64), tensor([True, False, True], ms.bool_)]
)
@pytest.mark.parametrize('kind', ['tensor', 'number', 'sequence'])
def test_legacy_tensor_setitem(index, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Legacy tensor setitem.
    Expectation: Tensor-index assignment produces the same updated values as NumPy for each replacement kind.
    """
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = array.copy()
    value = ([21, 22, 23] if index.dtype == ms.bool_ else [21, 22, 23, 24]) if kind != 'number' else 9
    if index.dtype == ms.bool_ and kind != 'number':
        # The legacy helper pads values on the right, treating these as per-row values.
        expected[index.asnumpy()] = np.asarray(value)[index.asnumpy(), None]
    else:
        expected[np_index(index)] = value
    value_arg = tensor(value) if kind == 'tensor' else value
    equal(getattr(cu, 'tensor_setitem_by_tensor_with_' + kind)(tensor(array), index, value_arg), expected)


@pytest.mark.parametrize('index', [0, -1])
@pytest.mark.parametrize('kind', ['number', 'tensor', 'sequence'])
def test_legacy_number_setitem(index, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Legacy number setitem.
    Expectation: Number-index assignment broadcasts each replacement to the same row values as NumPy.
    """
    array = np.arange(12).reshape(3, 4)
    expected = array.copy()
    value = 17 if kind == 'number' else [4, 5, 6, 7]
    expected[index] = value
    equal(
        getattr(cu, 'tensor_setitem_by_number_with_' + kind)(
            tensor(array), index, tensor(value) if kind == 'tensor' else value
        ),
        expected,
    )


@pytest.mark.parametrize('value', [7, tensor([3, 4, 5, 6]), [3, 4, 5, 6], (3, 4, 5, 6)])
def test_ellipsis_bool_and_list_setitem(value):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Ellipsis bool and list setitem.
    Expectation: Ellipsis and true boolean assignments update all selected values, false is unchanged, and list
        indices update only their rows.
    """
    array = np.arange(12).reshape(3, 4)
    expected = np.broadcast_to(np_index(value) if isinstance(value, ms.Tensor) else value, (3, 4))
    equal(cu.tensor_setitem_by_ellipsis(tensor(array), Ellipsis, value), expected)
    equal(cu.tensor_setitem_by_bool(tensor(array), True, value), expected)
    equal(cu.tensor_setitem_by_bool(tensor(array), False, value), array)
    equal(cu.tensor_setitem_by_list(tensor(array), [0, 2], tensor(9)), [[9] * 4, [4, 5, 6, 7], [9] * 4])


def test_parameter_scatter_updates_mutate_parameter():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Parameter scatter updates mutate parameter.
    Expectation: Scatter updates retain Parameter identity and mutate the selected row to [8, 8].
    """
    for fn, index in [
        (cu.tensor_setitem_by_number_with_number, -1),
        (cu.tensor_setitem_by_tensor_with_tensor, tensor([1], ms.int32)),
    ]:
        source = ms.Parameter(tensor([[1, 2], [3, 4]]), name='independent')
        result = fn(source, index, tensor(8))
        assert result is source
        equal(source, [[1, 2], [8, 8]])


@pytest.mark.parametrize('index', [tensor([True, False, True], ms.bool_), tensor([False] * 3, ms.bool_)])
def test_bool_mask_assignment_broadcasts_selected_rows(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Bool mask assignment broadcasts selected rows.
    Expectation: The replacement row broadcasts to exactly the rows selected by the boolean mask.
    """
    array = np.arange(12).reshape(3, 4)
    expected = array.copy()
    expected[index.asnumpy()] = [20, 21, 22, 23]
    equal(cu.handle_setitem_by_bool_tensor(tensor(array), index, tensor([20, 21, 22, 23])), expected)


@pytest.mark.parametrize(
    'value,dtype',
    [
        ([1, 2], ms.float16),
        ((tensor(1), tensor(2)), ms.float32),
        ((tensor(1, ms.int32), 2), ms.int32),
        ([1, 2.5], ms.float64),
    ],
)
def test_sequence_updates_promote_and_stack(value, dtype):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Sequence updates promote and stack.
    Expectation: Mixed sequence values use the requested promoted dtype and generated replacement tensors
        preserve [3, 4].
    """
    equal(cu.sequence_to_tensor(value, dtype), [1, 2.5] if dtype == ms.float64 else [1, 2], dtype)
    equal(
        cu._generate_updates_from_sequence(
            tensor(np.zeros((2, 2))), tensor([0], ms.int64), (3, 4), co.SET_ITEM_BY_NON_TENSOR
        ),
        [3, 4],
    )


@pytest.mark.parametrize(
    'name,operation',
    [
        ('_tensor_add', np.add),
        ('_tensor_sub', np.subtract),
        ('_tensor_mul', np.multiply),
        ('_tensor_div', np.divide),
        ('_tensor_mod', np.mod),
        ('_tensor_pow', np.power),
        ('_tensor_rpow', lambda x, y: np.power(y, x)),
        ('_tensor_floordiv', np.floor_divide),
    ],
)
def test_arithmetic_helpers_accept_sequences(name, operation):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Arithmetic helpers accept sequences.
    Expectation: Tensor-sequence arithmetic in either operand order matches the independent NumPy operation.
    """
    equal(getattr(cu, name)(tensor([2, 3]), [3, 2]), operation([2, 3], [3, 2]))
    if name in ('_tensor_sub', '_tensor_div', '_tensor_mod', '_tensor_floordiv'):
        equal(getattr(cu, name)([2, 3], tensor([3, 2])), operation([2, 3], [3, 2]))


def test_scalar_item_and_itemset_contracts():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Scalar item and itemset contracts.
    Expectation: Scalar extraction and item assignment retain their intended values, while invalid shapes and
        indices raise explicit errors.
    """
    assert cu.tensor_item(tensor(3)) == 3
    assert cu.tensor_item(tensor([4])) == 4
    assert cu._check_scalar_tensor_args(()) is None
    assert cu._check_scalar_tensor_args((None,)) is None
    equal(cu.tensor_itemset(tensor([4]), 9), [9])
    equal(cu.tensor_itemset(tensor([[1, 2], [3, 4]]), 2, 9), [[1, 2], [9, 4]])
    for fn, args, error, message in [
        (cu.tensor_item, (tensor([1, 2]),), ValueError, 'only one'),
        (cu._check_scalar_tensor_args, ((1,),), ValueError, 'should not'),
        (cu.tensor_itemset, (tensor([1]),), ValueError, 'at least'),
        (cu.tensor_itemset, (tensor([1]), 1, 2, 3), ValueError, 'at most'),
        (cu.tensor_itemset, (tensor([1]), 'bad', 2), TypeError, 'integer'),
        (cu.tensor_itemset_with_number, (tensor([1]), 'bad'), IndexError, 'number'),
        (cu.tensor_itemset_with_number, (tensor([1, 2]), 3), IndexError, 'shape'),
        (cu.tensor_itemset_by_tuple_with_number, (tensor([[1]]), (0,), 9), IndexError, 'Tuple index'),
        (cu.tensor_setitem_by_number_with_number, (tensor([1, 2]), 2, 9), IndexError, 'out of bounds'),
    ]:
        with pytest.raises(error, match=message):
            fn(*args)


def test_constexpr_scalar_shape_and_type_contracts():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Constexpr scalar shape and type contracts.
    Expectation: Constant helpers retain scalar values, shapes and dtype metadata and reject invalid ranges,
        types or broadcasting.
    """
    assert co.make_empty_slice() == slice(None)
    assert co.check_equal(2, 2) == 2
    with pytest.raises(ValueError, match='2,3'):
        co.check_equal(2, 3)
    with pytest.raises(NotImplementedError, match='unsupported'):
        co.raise_unimplemented_error('unsupported')
    assert co.check_range(-3, None) == -3
    assert co._deep_list((-1, (0, -2)), 3) == [2, [0, 1]]
    equal(co.make_tensor([], ms.int32, (2, 3)), np.zeros((2, 3)), ms.int32)
    equal(co.make_tensor(-1, ms.int64, None, 3), 2, ms.int64)
    equal(co.make_tensor(1.5, ms.float32), 1.5, ms.float32)
    equal(co.make_tensor([], ms.int64), [], ms.int64)
    with pytest.raises(TypeError, match='Input data'):
        co.make_tensor(object())
    for dim in [-1, 9]:
        with pytest.raises(ValueError, match='range'):
            co.judge_data_dim(dim)
    assert co.judge_data_dim(3) is None
    assert co.get_source_shape((2, 3), (1, 2, 3)) == (2, 3)
    assert co.get_source_shape((2, 3), (3,)) == (3,)
    assert co.check_valid_dim(2, 'matrix') is None
    with pytest.raises(ValueError, match='1d or 2d'):
        co.check_valid_dim(3, 'matrix')
    assert co.judge_indexes_types((ms.int32, ms.int64), dt.int_type)
    assert not co.judge_indexes_types((ms.int32, ms.float32), dt.int_type)
    assert co.judge_indexes_types((ms.int32,), ms.int32)
    assert not co.judge_indexes_types((ms.float32,), ms.int32)
    assert co.check_type_isinstance((1, 2), int)
    assert not co.check_type_isinstance((1, 'x'), int)
    assert co.check_type_invalid(ms.float32, dt.int_type)
    assert co.check_type_valid(ms.int32, dt.int_type, 'getitem') is None
    with pytest.raises(TypeError, match='support'):
        co.check_type_valid(ms.float32, dt.int_type, 'operator')
    with pytest.raises(IndexError, match='support'):
        co.check_type_valid(ms.float32, dt.int_type, co.TENSOR_GETITEM)
    assert co.get_index_tensor_dtype(ms.int32) == co.INT_
    assert co.get_index_tensor_dtype(ms.bool_) == co.BOOL_
    with pytest.raises(IndexError, match='not supported'):
        co.get_index_tensor_dtype(ms.float32)
    assert co.check_tensors_dtype_same(ms.int32, ms.int32, 'setitem')
    with pytest.raises(TypeError, match='not consistent'):
        co.check_tensors_dtype_same(ms.int32, ms.float32, 'setitem')
    assert co.tuple_index_type_cnt((dt.TensorType(ms.int32),), 'getitem') == co.ALL_TENSOR
    assert co.tuple_index_type_cnt((ms.int64, dt.Slice()), 'getitem') == co.ALL_BASIC
    assert co.tuple_index_type_cnt((dt.TensorType(ms.int32), dt.NoneType()), 'getitem') == co.MIXED
    assert co.check_indices(2, 'x') == 2
    with pytest.raises(IndexError, match='unreasonable'):
        co.check_indices(0, 'x')
    assert co._judge_order_continuous(()) is False
    assert co.get_np_eps(ms.float32) == float(np.finfo(np.float32).eps)
    assert co.check_number_index_type(True) == co.BOOL_
    assert co.check_number_index_type(1) == co.INT_
    with pytest.raises(IndexError, match='Only support'):
        co.check_number_index_type(1.5)
    assert co.unpack(([[3]],)) == 3
    assert co.unpack((1, 2)) == (1, 2)
    assert co.get_step_from_slice(slice(None)) == 1
    assert co.get_step_from_slice(slice(None, None, -2)) == -2
    assert co.tuple_slice((1, 2, 3), 1, 3) == (2, 3)
    assert co.expanded_shape((2, 3), 2) == (1, 1, 2, 3)
    assert co.compute_slice_shape((2, 3), 2, 1, 1) == [1, 1, 1, 3]
    assert co.rem_not_expanded_dims(0, True, 1, 1, ()) == ((False, True), 0)
    assert co.rem_not_expanded_dims(1, False, 2, 0, (False,)) == ((False, True, True), 0)
    with pytest.raises(ValueError, match='Can not pad'):
        co.generate_padding_shape((2, 3), 1)


@pytest.mark.parametrize(
    'sequence,dim,expected',
    [
        ([], 3, False),
        ([False, False], 2, False),
        ([True, False, True], 3, [0, 2]),
        ([True, False], None, [True, False]),
        ([-1, 0], 3, [2, 0]),
    ],
)
def test_sequence_to_index_boolean_and_negative_indices(sequence, dim, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Sequence to index boolean and negative indices.
    Expectation: Boolean masks and negative integer sequences normalize to the expected index values or the
        empty-selection sentinel.
    """
    result = co.sequence_to_index(sequence, dim)
    if expected is False:
        assert result is False
    else:
        equal(result, expected, ms.bool_ if dim is None else ms.int64)


def test_sequence_to_index_rejects_boolean_shape_mismatch():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Sequence to index rejects boolean shape mismatch.
    Expectation: A boolean sequence with incompatible dimension raises IndexError before index generation.
    """
    with pytest.raises(IndexError, match='boolean dimension'):
        co.sequence_to_index([True], 3)


@pytest.mark.parametrize(
    'index,shape', [(slice(1, 3), (3,)), (slice(None, None, 2), (3, 2)), (slice(2, 0, -1), (3, 2))]
)
def test_slice2indices_cartesian_coordinates(index, shape):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Slice2indices cartesian coordinates.
    Expectation: Materialized int64 coordinates match the Cartesian product of the requested slice positions.
    """
    coordinates = np.stack(
        np.meshgrid(np.arange(shape[0])[index], *[np.arange(x) for x in shape[1:]], indexing='ij'), axis=-1
    )
    equal(co.slice2indices(index, shape), coordinates, ms.int64)


def test_empty_slice_indices_and_scalar_updates_shapes():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Empty slice indices and scalar updates shapes.
    Expectation: Empty slices return False and scalar replacements broadcast to the generated update shape.
    """
    assert co.slice2indices(slice(2, 2), (3, 2)) is False
    for op_type, indices, expected_shape in [
        (co.SET_ITEM_BY_ONE_TENSOR, (2,), (2, 3)),
        (co.SET_ITEM_BY_TUPLE_OF_TENSOR, (2, 1), (2, 3)),
    ]:
        assert co.generate_updates_shape((4, 3), indices, op_type) == expected_shape
        equal(co.convert_scalar_to_tensor((4, 3), ms.float32, indices, 7, op_type), np.full(expected_shape, 7))


def test_index_type_position_classification():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Index type position classification.
    Expectation: Index kinds retain their exact tuple positions, while unsupported types and multiple ellipses
        raise explicit errors.
    """
    types = (dt.Slice(), dt.Ellipsis_(), dt.NoneType(), ms.int64, ms.bool_, dt.TensorType(ms.int32), (ms.int64,))
    assert co.get_pos_of_indexes_types(types, 'getitem') == ((0,), (1,), (2,), (3,), (4,), (5,), (6,))
    with pytest.raises(TypeError, match='only support'):
        co.get_pos_of_indexes_types((ms.float32,), 'getitem')
    with pytest.raises(IndexError, match='single ellipsis'):
        co.get_pos_of_indexes_types((dt.Ellipsis_(), dt.Ellipsis_()), 'getitem')


@pytest.mark.parametrize(
    'index',
    [
        (slice(None), 1),
        (1, slice(None, None, 2)),
        (tensor([0, 2], ms.int64), slice(1, 4)),
        ([0, 2], slice(1, 4)),
        (tensor([0, 2], ms.int64), tensor([1, 3], ms.int64)),
    ],
)
def test_generated_mixed_indices_select_numpy_values(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Generated mixed indices select numpy values.
    Expectation: Generated int64 coordinates select exactly the same values as NumPy mixed indexing.
    """
    array = np.arange(12).reshape(3, 4)
    result = cu._generate_indices_from_tuple(tensor(array), index, co.TENSOR_GETITEM, None)
    coordinates = result.asnumpy()
    actual = array[tuple(coordinates[..., i] for i in range(coordinates.shape[-1]))]
    np.testing.assert_array_equal(actual, array[np_index(index)])
    assert result.dtype == ms.int64


def test_generated_indices_empty_and_invalid_tensor_dtype():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Generated indices empty and invalid tensor dtype.
    Expectation: Empty selections return False, noninteger tensor indices fail, and combined index tensors
        preserve coordinate order.
    """
    assert cu._generate_indices_from_tuple(tensor(np.zeros((2, 3))), (slice(1, 1), 0), 'getitem', None) is False
    with pytest.raises(IndexError, match='int or bool'):
        cu._generate_indices_from_tuple(tensor(np.zeros((2, 3))), (tensor([1.0]), slice(None)), 'getitem', None)
    equal(
        cu._generate_indices_from_tuple_of_tensor((tensor([0, 1], ms.int32), tensor([2, 0], ms.int64)), 'getitem'),
        [[[0, 2], [1, 0]]],
        ms.int64,
    )


def test_format_and_multi_dim_index_tensor_broadcasting():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Format and multi dim index tensor broadcasting.
    Expectation: Negative indices wrap correctly and advanced indices broadcast to the expected coordinate grid
        and expanded view.
    """
    indices = [tensor([-1, 0], ms.int64), tensor([-2, 1], ms.int64)]
    result = cu.format_index_tensor(indices, ((0, 1), (3, 4)))
    equal(result[0], [2, 0])
    equal(result[1], [2, 1])
    equal(cu.format_index_tensor(tensor([[-1, -2], [0, 1]], ms.int64), (None, (3, 4))), [[2, 2], [0, 1]])
    equal(cu.handle_multi_dim_index_tensor([np.array([0, 2]), np.array([1])], ((2,),)), [[0, 1], [2, 1]])
    arg = ((2,), (2, 3), (2, 1), (3,), (0,), 0)
    equal(
        cu.handle_multi_dim_index_tensor([[0, 2], [1, 2, 3]], arg), [[[0, 1], [0, 2], [0, 3]], [[2, 1], [2, 2], [2, 3]]]
    )
    source = tensor(np.arange(6).reshape(2, 3))
    expanded, indices = cu._expand_data_dims(source, (None, True, slice(None)))
    equal(expanded, np.arange(6).reshape(1, 1, 2, 3))
    assert indices[0] == slice(None) and indices[2] == slice(None)
    equal(indices[1], [0], ms.int64)
    with pytest.raises(IndexError, match='False'):
        cu._expand_data_dims(source, (False,))
    assert cu._transform_ellipsis_to_slice(source, (Ellipsis, 1), 'getitem') == (slice(None), 1)
    with pytest.raises(IndexError, match='out rang'):
        cu._transform_ellipsis_to_slice(source, (1, 1, 1), 'getitem')


@pytest.mark.parametrize(
    'index',
    [
        (None, slice(None), slice(None)),
        (True,),
        (False,),
        (tensor([0, 2], ms.int64), slice(None)),
        (tensor([True, False, True], ms.bool_), slice(None)),
        ([0, 2], slice(None)),
        (slice(None), tensor([-1, 0], ms.int64)),
        (slice(1, 1), slice(None)),
    ],
)
def test_remove_expanded_dimensions_preserves_assignment_shape(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Remove expanded dimensions preserves assignment shape.
    Expectation: Removing inserted dimensions preserves NumPy assignment coordinates or the explicit
        empty-selection sentinel.
    """
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    selected = array[np_index(index)]
    result, value, advanced = cu.remove_expanded_dims(index, array.shape, tensor(np.full(selected.shape, 9)))
    assert isinstance(advanced, int)
    if selected.size == 0:
        assert result is False
    else:
        actual = array.copy()
        actual[np_index(result)] = value.asnumpy()
        expected = array.copy()
        expected[np_index(index)] = 9
        np.testing.assert_array_equal(actual, expected)


def test_remove_expanded_dimensions_invalid_and_membership():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Remove expanded dimensions invalid and membership.
    Expectation: Invalid assignments fail before updates, and membership compares complete tensor values with
        compatible dtypes.
    """
    with pytest.raises(IndexError, match='invalid index'):
        cu.remove_expanded_dims(('bad',), (2,), tensor(1))
    with pytest.raises(IndexError, match='unable to broadcast'):
        cu.remove_expanded_dims((False, tensor([0, 1], ms.int64)), (2,), tensor(1))
    equal(
        cu.tensor_in_sequence(tensor([1, 2]), [None, tensor([1, 2], ms.int64), tensor([1, 2]), tensor([0])]),
        True,
        ms.bool_,
    )
    equal(cu.tensor_in_sequence(tensor([1, 2]), [None, tensor([2, 1])]), False, ms.bool_)
    for fn, size in [(cu.check_coo_tensor_input_length, 3), (cu.check_csr_tensor_input_length, 4)]:
        args = tuple(range(size))
        assert fn(args) is args
        with pytest.raises(ValueError, match='inputs'):
            fn(())
    with pytest.raises(ValueError, match='zero-size'):
        cu._check_shape_mul((0, 2))
    with pytest.raises(NotImplementedError, match='not implemented'):
        cu.check_indices(3, tensor([0], ms.int32), 'raise')


@pytest.mark.parametrize('index', [0, Ellipsis, None, True, ()])
def test_view_setitem_contiguous_storage(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: View setitem contiguous storage.
    Expectation: Contiguous view assignment updates both the returned tensor and original storage exactly as
        NumPy.
    """
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = array.copy()
    expected[index] = 19
    source = tensor(array)
    result = cu._tensor_setitem(source, index, 19)
    equal(result, expected)
    equal(source, expected)


@pytest.mark.xfail(strict=True, reason='_scalar_to_tensor calls tensor-only ops.dtype on a Python scalar')
def test_scalar_to_tensor_accepts_python_integer():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Scalar to tensor accepts python integer.
    Expectation: The Python integer three becomes an int64 scalar tensor without changing its value.
    """
    equal(cu._scalar_to_tensor(3), 3, ms.int64)


@pytest.mark.parametrize(
    'kind,arg,index,expected',
    [
        (cu.ValueTransferType.kStrideSliceWithMask, (((0, 0), (2, 3), (1, 2)), (0, 0, 0)), None, [[0, 2], [3, 5]]),
        (cu.ValueTransferType.kGatherND, None, tensor([[0, 1], [1, 2]], ms.int64), [1, 5]),
        (cu.ValueTransferType.kTensorScatterUpdate, None, tensor([[0, 1], [1, 2]], ms.int64), [[0, 9, 2], [3, 4, 9]]),
        (cu.ValueTransferType.kSelect, None, [[True, False, True], [False, True, False]], [[9, 1, 9], [3, 9, 5]]),
        (cu.ValueTransferType.kReshape, (3, 2), None, [[0, 1], [2, 3], [4, 5]]),
        (cu.ValueTransferType.kGather, None, tensor([1, 0], ms.int64), [[3, 4, 5], [0, 1, 2]]),
        (cu.ValueTransferType.kExpandDims, None, None, [[[0, 1, 2], [3, 4, 5]]]),
        (cu.ValueTransferType.kUnsqueeze, 1, None, [[[0, 1, 2]], [[3, 4, 5]]]),
        (cu.ValueTransferType.kStrideSlice, ((0, 0), (2, 3), (1, 2)), None, [[0, 2], [3, 5]]),
    ],
)
def test_data_transfer_ops_numerical(kind, arg, index, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Data transfer ops numerical.
    Expectation: Each transfer operation produces its independently specified tensor values and shape.
    """
    source = tensor(np.arange(6).reshape(2, 3))
    value = tensor([9, 9]) if kind == cu.ValueTransferType.kTensorScatterUpdate else tensor(np.full((2, 3), 9))
    equal(cu.data_update_by_ops(kind, arg, source, index, source, value), expected)


def test_data_transfer_unknown_raises_and_parameter_scatter():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Data transfer unknown raises and parameter scatter.
    Expectation: Parameter scatter mutates the selected element; unsupported transfer kinds and out-of-range
        indices raise IndexError.
    """
    source = ms.Parameter(tensor([[1, 2], [3, 4]]), name='transfer')
    equal(
        cu.data_update_by_ops(
            cu.ValueTransferType.kScatterNdUpdate, None, source, tensor([[1, 0]], ms.int64), source, tensor([9])
        ),
        [[1, 2], [9, 4]],
    )
    equal(source, [[1, 2], [9, 4]])
    with pytest.raises(IndexError, match='Inlvaid'):
        cu.data_update_by_ops(cu.ValueTransferType.kUnknown, None, tensor([1]), None, None)
    with pytest.raises(IndexError, match='Inlvaid'):
        cu.data_update((cu.ValueTransferType.kUnknown,), (None,), tensor([1]), None)
    with pytest.raises(IndexError, match='out of bounds'):
        cu.data_update((cu.ValueTransferType.kRaiseIndexError,), ((5, 2),), tensor([1]), None)


@pytest.mark.parametrize(
    'kind,arg,index,value,expected',
    [
        (cu.ValueTransferType.kJustReturn, tensor([7]), None, None, [7]),
        (cu.ValueTransferType.kSetItemByBool, None, True, tensor(9), [[9, 9, 9], [9, 9, 9]]),
        (cu.ValueTransferType.kCopySlice, ((0, 0), (1, 3), (1, 1)), None, tensor([[8, 8, 8]]), [[8, 8, 8], [3, 4, 5]]),
        (cu.ValueTransferType.kSetItemByEllipsis, None, Ellipsis, tensor([7, 8, 9]), [[7, 8, 9], [7, 8, 9]]),
        (cu.ValueTransferType.kReSetItemByIndex, None, 0, tensor(7), [[7, 7, 7], [3, 4, 5]]),
        (cu.ValueTransferType.kEmptyTensor, (0, 3), None, None, np.empty((0, 3))),
        (cu.ValueTransferType.kGetitemByBoolTensor, None, tensor([False, True], ms.bool_), None, [[3, 4, 5]]),
        (
            cu.ValueTransferType.kSetitemByBoolTensor,
            None,
            tensor([False, True], ms.bool_),
            tensor(8),
            [[0, 1, 2], [8, 8, 8]],
        ),
    ],
)
def test_data_transfer_control_operations(kind, arg, index, value, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Data transfer control operations.
    Expectation: The selected control operation produces the expected tensor update without altering index
        semantics.
    """
    equal(cu.data_update((kind,), (arg,), tensor(np.arange(6).reshape(2, 3)), index, value), expected)


@pytest.mark.parametrize(
    'kinds,args,value,expected',
    [
        ((cu.ValueTransferType.kByPass, cu.ValueTransferType.kNumberToTensor), (None, None), 7, 7),
        (
            (cu.ValueTransferType.kExpandDims, cu.ValueTransferType.kBroadCast),
            (0, (2, 3)),
            tensor([1, 2, 3]),
            [[1, 2, 3]] * 2,
        ),
        (
            (cu.ValueTransferType.kCast, cu.ValueTransferType.kReshape),
            (None, (3, 1)),
            tensor([1, 2, 3], ms.int64),
            [[1], [2], [3]],
        ),
        ((cu.ValueTransferType.kScatterND,), ((tensor([[0], [2]], ms.int64), (3,)),), tensor([7, 8]), [7, 0, 8]),
        (
            (cu.ValueTransferType.kHandleSequenceValue,),
            ((co.SET_ITEM_BY_ONE_TENSOR, [0, 1]),),
            (7, 8, 9),
            [[7, 8, 9]] * 2,
        ),
        ((cu.ValueTransferType.kHandleSequenceValue,), ((co.SET_ITEM_BY_NON_TENSOR, None),), (7, 8), [7, 8]),
    ],
)
def test_value_transfer_cast_broadcast_reshape_scatter(kinds, args, value, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Value transfer cast broadcast reshape scatter.
    Expectation: The transfer sequence produces the expected cast, broadcast, reshape or scatter result in
        float32.
    """
    equal(cu.value_update(kinds, args, tensor(np.zeros((2, 3))), value), expected, ms.float32)


def test_value_transfer_rejects_unknown_kind():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Value transfer rejects unknown kind.
    Expectation: An unsupported replacement-transfer kind raises the documented IndexError.
    """
    with pytest.raises(IndexError, match='Inlvaid'):
        cu.value_update((cu.ValueTransferType.kUnknown,), (None,), tensor([1]), tensor([2]))


@pytest.mark.parametrize(
    'where,initial', [(True, None), (None, 3), (tensor([[True, False, True]], ms.bool_), 2), (False, tensor(3))]
)
def test_reduce_helper_initial_and_where(where, initial):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Reduce helper initial and where.
    Expectation: Reduction results match the independent expected values for each initial value and boolean
        mask.
    """
    array = np.arange(6).reshape(2, 3)
    kwargs = {'axis': 1}
    if initial is not None:
        kwargs['initial'] = int(initial.asnumpy()) if isinstance(initial, ms.Tensor) else initial
    if where is not None:
        kwargs['where'] = np_index(where)
    expected = np.max(array, **kwargs)
    equal(cu.reduce_(tensor(array), ops.ReduceMax(), ops.maximum, axis=1, initial=initial, where=where), expected)


def test_reduce_helper_rejects_invalid_initial_and_mask_without_identity():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Reduce helper rejects invalid initial and mask without identity.
    Expectation: A missing identity under a mask or nonscalar initial value raises the documented validation
        error.
    """
    for initial in [tensor([2]), 'bad']:
        with pytest.raises(TypeError, match='initial must be scalar'):
            cu.reduce_(tensor([1, 2]), ops.ReduceMax(), ops.maximum, initial=initial)
    with pytest.raises(ValueError, match='initial value'):
        cu.reduce_(tensor([1, 2]), ops.ReduceMax(), ops.maximum, where=False)


def module(name):
    return importlib.import_module('mindspore.ops.composite.multitype_ops.' + name + '_impl')


_ARITHMETIC = [
    ('add', '_scalar_add_scalar', 3, 2, 5),
    ('add', '_string_concat_string', 'ab', 'c', 'abc'),
    ('mul', '_mul_scalar', 3, 2, 6),
    ('mul', '_number_mul_string', 3, 'ab', 'ababab'),
    ('mul', '_string_mul_number', 'ab', 2, 'abab'),
    ('sub', '_sub_scalar', 3, 2, 1),
    ('div', '_div_scalar', 3, 2, 1.5),
    ('floordiv', '_floordiv_scalar', 7, 2, 3),
    ('mod', '_mod_scalar', 7, 2, 1),
    ('pow', '_pow_scalar', 3, 2, 9),
    ('bitwise_and', '_bitwise_and_scalar', 6, 3, 2),
    ('bitwise_or', '_bitwise_or_scalar', 6, 3, 7),
    ('bitwise_xor', '_bitwise_xor_scalar', 6, 3, 5),
    ('left_shift', '_left_shift_scalar', 3, 2, 12),
    ('right_shift', '_right_shift_scalar', 12, 2, 3),
]


@pytest.mark.parametrize('mod,name,left,right,expected', _ARITHMETIC)
def test_multitype_scalar_operations(monkeypatch, mod, name, left, right, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype scalar operations.
    Expectation: Each dispatched scalar operation preserves the expected Python result type and ordered
        native-boundary arguments.
    """
    boundary_names = {
        '_string_concat_string': 'string_concat',
        '_bitwise_and_scalar': 'bit_and',
        '_bitwise_or_scalar': 'bit_or',
        '_bitwise_xor_scalar': 'bit_xor',
        '_left_shift_scalar': 'bit_left_shift',
        '_right_shift_scalar': 'bit_right_shift',
    }
    calls = []
    if name in boundary_names:

        def native_boundary(x, y):
            calls.append((x, y))
            return {
                'add': operator.add,
                'bitwise_and': operator.and_,
                'bitwise_or': operator.or_,
                'bitwise_xor': operator.xor,
                'left_shift': operator.lshift,
                'right_shift': operator.rshift,
            }[mod](x, y)

        monkeypatch.setattr(module(mod).F if mod == 'add' else module(mod).inner, boundary_names[name], native_boundary)
    result = getattr(module(mod), name)(left, right)
    assert result == expected
    assert type(result) is type(expected)
    if name in boundary_names:
        assert calls == [(left, right)]


@pytest.mark.parametrize(
    'mod,op,names',
    [
        (
            'add',
            np.add,
            (
                '_scalar_add_tensor',
                '_tensor_add_scalar',
                '_tensor_add_tensor',
                '_tuple_add_tensor',
                '_tensor_add_tuple',
                '_list_add_tensor',
                '_tensor_add_list',
            ),
        ),
        (
            'mul',
            np.multiply,
            (
                '_scalar_mul_tensor',
                '_tensor_mul_scalar',
                '_mul_tensor',
                '_tuple_mul_tensor',
                '_tensor_mul_tuple',
                '_list_mul_tensor',
                '_tensor_mul_list',
            ),
        ),
        (
            'sub',
            np.subtract,
            (
                '_scalar_sub_tensor',
                '_tensor_sub_scalar',
                '_sub_tensor',
                '_tuple_sub_tensor',
                '_tensor_sub_tuple',
                '_list_sub_tensor',
                '_tensor_sub_list',
            ),
        ),
        (
            'div',
            np.divide,
            (
                '_scalar_div_tensor',
                '_tensor_div_scalar',
                '_div_tensor',
                '_tuple_div_tensor',
                '_tensor_div_tuple',
                '_list_div_tensor',
                '_tensor_div_list',
            ),
        ),
        (
            'floordiv',
            np.floor_divide,
            (
                '_scalar_floordiv_tensor',
                '_tensor_floordiv_scalar',
                '_floordiv_tensor',
                '_tuple_floordiv_tensor',
                '_tensor_floordiv_tuple',
                '_list_floordiv_tensor',
                '_tensor_floordiv_list',
            ),
        ),
        ('mod', np.mod, ('_tuple_mod_tensor', '_tensor_mod_tuple', '_list_mod_tensor', '_tensor_mod_list')),
        ('pow', np.power, ('_scalar_pow_tensor', '_tensor_pow_scalar', '_pow_tensor')),
    ],
)
def test_multitype_tensor_and_sequence_arithmetic(mod, op, names):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype tensor and sequence arithmetic.
    Expectation: Dispatched tensor or sequence arithmetic matches the independent NumPy operation in both
        operand orders.
    """
    for name in names:
        if name.startswith('_scalar'):
            left, right = 3, tensor([2, 4])
        elif '_scalar' in name:
            left, right = tensor([2, 4]), 3
        elif name.startswith('_tuple'):
            left, right = (3, 2), tensor([2, 4])
        elif name.startswith('_list'):
            left, right = [3, 2], tensor([2, 4])
        elif name.endswith('_tuple'):
            left, right = tensor([2, 4]), (3, 2)
        elif name.endswith('_list'):
            left, right = tensor([2, 4]), [3, 2]
        else:
            left, right = tensor([2, 4]), tensor([3, 2])
        equal(getattr(module(mod), name)(left, right), op(np_index(left), np_index(right)))


@pytest.mark.parametrize(
    'mod,names,op',
    [
        (
            'bitwise_and',
            ('_bitwise_and_tensor', '_tensor_bitwise_and_scalar', '_scalar_bitwise_and_tensor'),
            np.bitwise_and,
        ),
        ('bitwise_or', ('_bitwise_or_tensor', '_tensor_bitwise_or_scalar', '_scalar_bitwise_or_tensor'), np.bitwise_or),
        (
            'bitwise_xor',
            ('_bitwise_xor_tensor', '_tensor_bitwise_xor_scalar', '_scalar_bitwise_xor_tensor'),
            np.bitwise_xor,
        ),
    ],
)
def test_multitype_bitwise_tensor_operations(mod, names, op):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype bitwise tensor operations.
    Expectation: Tensor bitwise dispatch matches NumPy elementwise results with int32 dtype.
    """
    for i, name in enumerate(names):
        left, right = (
            (tensor([6, 3], ms.int32), tensor([3, 6], ms.int32))
            if i == 0
            else ((tensor([6, 3], ms.int32), 3) if i == 1 else (3, tensor([6, 3], ms.int32)))
        )
        equal(getattr(module(mod), name)(left, right), op(np_index(left), np_index(right)), ms.int32)


@pytest.mark.parametrize('name', ['_list_mul_scalar', '_scalar_mul_list', '_tuple_mul_scalar', '_scalar_mul_tuple'])
@pytest.mark.parametrize('count', [-1, 0, 3, 1.5])
def test_multitype_sequence_repetition(name, count):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype sequence repetition.
    Expectation: Integer repetition preserves list or tuple type and values, while fractional counts raise
        TypeError.
    """
    sequence = [1, 2] if 'list' in name else (1, 2)
    args = (count, sequence) if name.startswith('_scalar') else (sequence, count)
    if not isinstance(count, int):
        with pytest.raises(TypeError, match='non-int'):
            getattr(module('mul'), name)(*args)
    else:
        result = getattr(module('mul'), name)(*args)
        assert result == sequence * count
        assert type(result) is type(sequence)


@pytest.mark.parametrize(
    'mod,op,names',
    [
        (
            'equal',
            operator.eq,
            ('_equal_bool', '_equal_scalar', '_equal_string', '_euqal_tuple', '_euqal_list', 'default_equal'),
        ),
        (
            'not_equal',
            operator.ne,
            ('_not_equal_scalar', '_not_equal_string', '_not_euqal_tuple', '_not_euqal_list', 'default_not_equal'),
        ),
        (
            'greater',
            operator.gt,
            ('_greater_scalar', '_greater_string', '_greater_than_tuple', '_greater_than_list', 'default_greater'),
        ),
        (
            'greater_equal',
            operator.ge,
            (
                '_greater_equal_scala',
                '_greater_equal_string',
                '_greater_equal_tuple',
                '_greater_equal_list',
                'default_greater_equal',
            ),
        ),
        ('less', operator.lt, ('_less_scala', '_less_string', '_less_tuple', '_less_list', 'default_less')),
        (
            'less_equal',
            operator.le,
            ('_less_equal_scala', '_less_equal_string', '_less_equal_tuple', '_less_equal_list', 'default_less_equal'),
        ),
    ],
)
def test_multitype_comparisons(monkeypatch, mod, op, names):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype comparisons.
    Expectation: Scalar comparison dispatch matches Python operators and invokes the expected boundary with
        ordered operands.
    """

    def make_boundary(recorded_calls):
        def boundary(x, y):
            recorded_calls.append((x, y))
            return operator.eq(x, y) if mod in ('equal', 'not_equal') else op(x, y)

        return boundary

    for name in names:
        if 'string' in name:
            left, right = 'ab', 'bc'
        elif 'tuple' in name:
            left, right = (1, 3), (2, 1)
        elif 'list' in name:
            left, right = [1, 3], [2, 1]
        elif 'bool' in name:
            left, right = True, False
        else:
            left, right = 2, 3
        if (
            'string' in name
            or 'bool' in name
            or (mod in ('equal', 'not_equal') and ('tuple' in name or 'list' in name))
        ):
            owner = module(mod).inner if 'string' in name and mod not in ('equal', 'not_equal') else module(mod).F
            suffix = {'greater': 'gt', 'greater_equal': 'ge', 'less': 'lt', 'less_equal': 'le'}.get(mod)
            attr = (
                ('string_' + suffix)
                if owner is getattr(module(mod), 'inner', None)
                else (
                    'bool_eq'
                    if 'bool' in name
                    else ('string_eq' if 'string' in name else ('tuple_equal' if 'tuple' in name else 'list_equal'))
                )
            )
            calls = []
            boundary = make_boundary(calls)
            with monkeypatch.context() as patch:
                patch.setattr(owner, attr, boundary)
                assert bool(getattr(module(mod), name)(left, right)) == op(left, right)
            assert calls == [(left, right)]
        else:
            assert bool(getattr(module(mod), name)(left, right)) == op(left, right)


@pytest.mark.parametrize(
    'mod,name,expected',
    [
        ('equal', '_tensor_equal_tensor', [True, False]),
        ('equal', '_tensor_equal_scalar', [True, False]),
        ('equal', '_scalar_equal_tensor', [True, False]),
        ('not_equal', '_tensor_not_equal_tensor', [False, True]),
        ('greater', '_greater_tensor', [False, True]),
        ('greater_equal', '_greater_equal_tensor', [True, True]),
        ('less', '_less_tensor', [False, False]),
        ('less', '_less_scalar_tensor', [False, True]),
        ('less_equal', '_less_equal_tensor', [True, False]),
    ],
)
def test_multitype_tensor_comparisons(mod, name, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype tensor comparisons.
    Expectation: Tensor comparisons match the expected elementwise boolean values.
    """
    left, right = (
        (2, tensor([2, 3])) if name.startswith('_scalar') or name == '_less_scalar_tensor' else (tensor([2, 3]), 2)
    )
    equal(getattr(module(mod), name)(left, right), expected, ms.bool_)


@pytest.mark.parametrize('mod', ['in', 'not_in'])
def test_multitype_membership(monkeypatch, mod):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype membership.
    Expectation: Sequence, dictionary, tensor and dtype membership agree with the expected inclusion or negation
        result and boundary arguments.
    """
    invert = mod == 'not_in'
    calls = []

    def string_in(x, y):
        calls.append((x, y))
        return x in y

    monkeypatch.setattr(module(mod).inner, 'string_in', string_in)
    dict_calls = []

    def dict_in(x, y):
        dict_calls.append((x, y))
        return x not in y if invert else x in y

    monkeypatch.setattr(module(mod).F, 'not_in_dict' if invert else 'in_dict', dict_in)
    prefix = '_number_not_in_' if invert else '_number_in_'
    for seq in [(1, 3), [1, 3]]:
        fn = getattr(module(mod), prefix + ('tuple' if isinstance(seq, tuple) else 'list'))
        assert bool(fn(1, seq)) is (not invert)
        assert bool(fn(2, seq)) is invert
    stem = '_string_not_in_' if invert else '_string_in_'
    for kind, sequence in [('string', 'abc'), ('tuple', ('a', 'b')), ('list', ['a', 'b'])]:
        assert bool(getattr(module(mod), stem + kind)('a', sequence)) is (not invert)
    assert bool(getattr(module(mod), '_str_not_in_dict' if invert else '_str_in_dict')('a', {'a': 1})) is (not invert)
    for kind, sequence in [('list', [tensor([1, 2])]), ('tuple', (tensor([1, 2]),))]:
        assert bool(
            getattr(module(mod), '_tensor_not_in_' + kind if invert else '_tensor_in_' + kind)(tensor([1, 2]), sequence)
        ) is (not invert)
        fn = getattr(module(mod), '_mstype_not_in_' + kind if invert else '_mstype_in_' + kind)
        assert bool(fn(ms.int32, [ms.int32] if kind == 'list' else (ms.int32,))) is (not invert)
    assert bool(getattr(module(mod), 'default_' + mod)(2, [1, 3])) is invert
    assert calls == [('a', 'abc')]
    assert dict_calls == [('a', {'a': 1})]


def test_multitype_logical_unary_and_none_contracts(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multitype logical unary and none contracts.
    Expectation: Logical, inversion, negation and None dispatch retain the expected scalar or tensor results and
        explicit invalid-operand errors.
    """
    calls = []

    def bool_and(x, y):
        calls.append(('and', x, y))
        return x and y

    def bool_or(x, y):
        calls.append(('or', x, y))
        return x or y

    def bool_not(x):
        calls.append(('not', x))
        return not x

    def string_not(x):
        calls.append(('strnot', x))
        return not x

    monkeypatch.setattr(module('logical_and').F, 'bool_and', bool_and)
    monkeypatch.setattr(module('logical_or').F, 'bool_or', bool_or)
    monkeypatch.setattr(module('logical_not').F, 'bool_not', bool_not)
    monkeypatch.setattr(module('logical_not').inner, 'string_not', string_not)
    for mod in ['logical_and', 'logical_or']:
        op = np.logical_and if mod.endswith('and') else np.logical_or
        assert bool(getattr(module(mod), '_' + mod + '_scala')(True, False)) == bool(op(True, False))
        assert bool(getattr(module(mod), 'default_' + mod)(True, False)) == bool(op(True, False))
        equal(
            getattr(module(mod), '_' + mod + '_tensor')(
                tensor([True, False], ms.bool_), tensor([False, True], ms.bool_)
            ),
            op([True, False], [False, True]),
        )
    for suffix, value in [('scala', 0), ('string', ''), ('none', None), ('tuple', ()), ('list', [])]:
        assert bool(getattr(module('logical_not'), '_logical_not_' + suffix)(value)) is True
    assert module('logical_not')._logical_not_tensor(tensor(True, ms.bool_)) is False
    assert module('logical_not')._logical_not_tensor(tensor(0, ms.int32)) is True
    assert calls == [
        ('and', True, False),
        ('or', True, False),
        ('not', False),
        ('strnot', ''),
        ('not', True),
        ('not', False),
    ]
    assert module('logical_not').default_logical_not(0)
    for value in [True, False, 3, -2]:
        assert module('invert')._invert_scalar(value) == ~value
    with pytest.raises(TypeError, match='bad operand'):
        module('invert')._invert_scalar(1.5)
    equal(module('invert')._invert_tensor(tensor([True, False], ms.bool_)), [False, True])
    assert module('negative')._neg_scalar(3) == -3
    equal(module('negative')._negative_tensor(tensor([2, -3])), [-2, 3])
    assert module('negative').default_negative(3) == -3
    assert module('uadd')._uadd_scala(3) == 3
    assert module('add')._none_add_none(None, None) is None
    for mod, expected in [('equal', True), ('not_equal', False)]:
        name = '_none_equal_none' if mod == 'equal' else '_none_not_equal_none'
        assert getattr(module(mod), name)(None, None) is expected


@pytest.mark.parametrize(
    'index',
    [
        True,
        False,
        1,
        slice(0, 2),
        Ellipsis,
        None,
        tensor(1, ms.int32),
        tensor(True, ms.bool_),
        tensor(False, ms.bool_),
        tensor([0, 2], ms.int64),
    ],
)
def test_process_dim_real_views_and_remaining_index_contract(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Process dim real views and remaining index contract.
    Expectation: Prepared views match NumPy while consumed dimensions, remaining indices and scalar-index
        metadata retain their expected values.
    """
    array = np.arange(12).reshape(3, 4)
    source = tensor(array)
    viewed, dim, remaining, original_dim, need = cu._process_dim_in_multi_dim_index(
        source, source, index, 0, cu._count_indexed_dims((index,)), 0, (), 0, False
    )
    if isinstance(index, bool) or index is None or (isinstance(index, ms.Tensor) and index.dtype == ms.bool_):
        equal(viewed, array[None])
        assert dim == 1 and original_dim == 0
        assert need is (index is not None)
        if index is not None:
            equal(remaining[0], [0] if bool(index) else [], ms.int64)
    elif isinstance(index, int) or (isinstance(index, ms.Tensor) and index.ndim == 0):
        equal(viewed, array[int(index)])
        assert dim == 0 and original_dim == 1 and need is False
        if isinstance(index, ms.Tensor):
            assert len(remaining) == 1 and remaining[0].shape == (0,) * 9
    elif isinstance(index, slice):
        equal(viewed, array[index])
        assert dim == 1 and original_dim == 1 and need is False
    elif index is Ellipsis:
        equal(viewed, array)
        assert dim == 2 and original_dim == 2 and need is False
    else:
        equal(viewed, array)
        assert dim == 1 and original_dim == 1 and need is True
        assert remaining[0] is index


@pytest.mark.parametrize(
    'index',
    [
        True,
        False,
        1,
        slice(0, 2),
        Ellipsis,
        None,
        tensor(1, ms.int32),
        tensor(True, ms.bool_),
        tensor(False, ms.bool_),
        tensor([0, 2], ms.int64),
    ],
)
def test_inplace_index_native_boundary_receives_views_indices_value(monkeypatch, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: The CPU lacks IndexPut; verify the Python preparation boundary separately.
    Expectation: The boundary receives the NumPy-equivalent view, remaining indices and scalar replacement once
        with accumulation disabled; source storage stays unchanged.
    """
    calls = []

    def record(view, remaining, value, accumulate):
        # Consume tensors synchronously so failures cannot leak into another case.
        calls.append((view.asnumpy().copy(), remaining, value.asnumpy().copy(), accumulate))
        return view

    monkeypatch.setattr(cu, 'inplace_index_put_op', record)
    source = tensor(np.arange(12).reshape(3, 4))
    value = tensor(9)
    result = cu._process_with_inplace_index_input(source, source, index, 0, 0, (), 0, value)
    assert result is source
    assert len(calls) == 1
    view, remaining, actual_value, accumulate = calls[0]
    if isinstance(index, bool) or index is None or (isinstance(index, ms.Tensor) and index.dtype == ms.bool_):
        np.testing.assert_array_equal(view, np.arange(12).reshape(1, 3, 4))
    elif isinstance(index, int) or (isinstance(index, ms.Tensor) and index.ndim == 0):
        np.testing.assert_array_equal(view, np.arange(12).reshape(3, 4)[1])
    elif isinstance(index, slice):
        np.testing.assert_array_equal(view, np.arange(12).reshape(3, 4)[:2])
    else:
        np.testing.assert_array_equal(view, np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(actual_value, 9)
    assert accumulate is False
    if isinstance(index, bool) or (isinstance(index, ms.Tensor) and index.dtype == ms.bool_):
        equal(remaining[0], [0] if bool(index) else [], ms.int64)
    elif isinstance(index, ms.Tensor):
        assert remaining[0].shape == ((0,) * 9 if index.ndim == 0 else index.shape)
        if index.ndim > 0:
            assert remaining[0] is index
    else:
        assert remaining == ()
    equal(source, np.arange(12).reshape(3, 4))


def test_index_preparation_errors_before_native_boundary():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Index preparation errors before native boundary.
    Expectation: Invalid basic and multidimensional indices raise the documented IndexError before native
        execution.
    """
    source = tensor(np.arange(6).reshape(2, 3))
    for fn in [cu._process_with_inplace_index_input, cu._process_with_do_copy]:
        for index, message in [(tensor(2, ms.int32), 'bounds'), ('bad', 'Invalid')]:
            with pytest.raises(IndexError, match=message):
                fn(source, source, index, 0, 0, (), 0, tensor(1))
    with pytest.raises(IndexError, match='Invalid'):
        cu._process_dim_in_multi_dim_index(source, source, 'bad', 0, 1, 0, (), 0, False)


@pytest.mark.parametrize(
    'index', [([0, 2],), (np.array([0, 2], np.int8),), ((0, 2),), (slice(None), tensor([0, 2], ms.int32))]
)
def test_multi_dim_preparation_converts_sequence_dtype(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Multi dim preparation converts sequence dtype.
    Expectation: Prepared sequence indices retain their expected values in int32 or int64 and request the native
        indexing primitive.
    """
    array = np.arange(12).reshape(3, 4)
    viewed, remaining, need = cu._process_multi_dim_index(tensor(array), index, (), 1 if len(index) == 1 else 2)
    equal(viewed, array)
    assert need is True
    expected = np.asarray(index[-1]) if not isinstance(index[-1], ms.Tensor) else index[-1].asnumpy()
    equal(remaining[-1], expected)
    assert remaining[-1].dtype in (ms.int32, ms.int64)


@pytest.mark.parametrize(
    'mod,name,index,boost',
    [
        ('getitem', '_tensor_getitem_by_number', 1, False),
        ('getitem', '_tensor_getitem_by_number', 1, True),
        ('getitem', '_tensor_getitem_by_none', None, False),
        ('getitem', '_tensor_getitem_by_ellipsis', Ellipsis, False),
        ('getitem', '_tensor_getitem_by_slice', slice(0, 2), True),
        ('getitem', '_tensor_getitem_by_tensor', tensor(1, ms.int32), True),
        ('getitem', '_tensor_getitem_by_tensor', tensor([0, 2], ms.int32), False),
        ('getitem', '_tensor_getitem_by_list', [0, 2], False),
        ('getitem', '_tensor_getitem_by_tuple', (slice(None), 1), True),
    ],
)
def test_getitem_tensor_dispatch_real_values(monkeypatch, mod, name, index, boost):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Getitem tensor dispatch real values.
    Expectation: Every dispatched tensor index selects the same values and shape as NumPy.
    """
    monkeypatch.setenv('MS_DEV_TENSOR_INDEX_BOOST', str(int(boost)))
    array = np.arange(12).reshape(3, 4)
    equal(getattr(module(mod), name)(tensor(array), index), array[np_index(index)])


@pytest.mark.parametrize(
    'name,index,value',
    [
        ('_tensor_setitem_by_tensor_with_tensor', tensor([0, 2], ms.int64), tensor(9)),
        ('_tensor_setitem_by_tensor_with_number', tensor([0, 2], ms.int64), 9),
        ('_tensor_setitem_by_tensor_with_tuple', tensor([0, 2], ms.int64), (7, 8, 9, 10)),
        ('_tensor_setitem_by_tensor_with_list', tensor([0, 2], ms.int64), [7, 8, 9, 10]),
        ('_tensor_setitem_by_number_with_number', 0, 9),
        ('_tensor_setitem_by_number_with_number', True, 9),
        ('_tensor_setitem_by_number_with_tensor', 0, tensor(9)),
        ('_tensor_setitem_by_number_with_tuple', 0, (7, 8, 9, 10)),
        ('_tensor_setitem_by_number_with_list', 0, [7, 8, 9, 10]),
        ('_tensor_setitem_by_ellipsis_with_number', Ellipsis, 9),
        ('_tensor_setitem_by_ellipsis_with_tensor', Ellipsis, tensor(9)),
        ('_tensor_setitem_by_ellipsis_with_tuple', Ellipsis, (7, 8, 9, 10)),
        ('_tensor_setitem_by_ellipsis_with_list', Ellipsis, [7, 8, 9, 10]),
        ('_tensor_setitem_by_none_with_number', None, 9),
        ('_tensor_setitem_by_none_with_tensor', None, tensor(9)),
        ('_tensor_setitem_by_none_with_tuple', None, (7, 8, 9, 10)),
        ('_tensor_setitem_by_none_with_list', None, [7, 8, 9, 10]),
        ('_tensor_setitem_by_list_with_number', [0, 2], 9),
        ('_tensor_setitem_by_list_with_tensor', [0, 2], tensor(9)),
        ('_tensor_setitem_by_list_with_tuple', [0, 2], (7, 8, 9, 10)),
        ('_tensor_setitem_by_list_with_list', [0, 2], [7, 8, 9, 10]),
    ],
)
def test_setitem_dispatch_legacy_values(monkeypatch, name, index, value):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Setitem dispatch legacy values.
    Expectation: Legacy replacement dispatch produces the same selected updates as NumPy assignment.
    """
    monkeypatch.delenv('MS_DEV_TENSOR_INDEX_BOOST', raising=False)
    array = np.arange(12).reshape(3, 4)
    expected = array.copy()
    expected[np_index(index)] = np_index(value)
    equal(getattr(module('setitem'), name)(tensor(array), index, value), expected)


@pytest.mark.parametrize('kind', ['number', 'tensor', 'tuple', 'list'])
@pytest.mark.parametrize(
    'index',
    [(1, slice(1, 3)), (-1, slice(1, 3)), (0, slice(2, 2)), (0, slice(tensor(0, ms.int32), tensor(2, ms.int32)))],
)
def test_tuple_dispatch_copy_slice_real_values(monkeypatch, kind, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Tuple dispatch copy slice real values.
    Expectation: Tuple slice replacement and tuple itemset produce identical intended NumPy updates for all
        replacement kinds.
    """
    monkeypatch.delenv('MS_DEV_TENSOR_INDEX_BOOST', raising=False)
    array = np.arange(12).reshape(3, 4)
    expected = array.copy()
    value = 8 if kind == 'number' else [7, 8]
    np_idx = tuple(
        (
            slice(
                int(x.start) if isinstance(x.start, ms.Tensor) else x.start,
                int(x.stop) if isinstance(x.stop, ms.Tensor) else x.stop,
                x.step,
            )
            if isinstance(x, slice)
            else x
        )
        for x in index
    )
    if expected[np_idx].size:
        expected[np_idx] = value
    value_arg = tensor(value) if kind == 'tensor' else (tuple(value) if kind == 'tuple' else value)
    equal(
        getattr(module('setitem'), '_tensor_setitem_by_tuple_with_' + kind)(tensor(array), index, value_arg), expected
    )
    equal(cu.tensor_itemset_by_tuple_with_tensor(tensor(array), index, tensor(value)), expected)


@pytest.mark.parametrize('step', [1, 2])
def test_slice_setitem_native_metadata_boundary_and_real_updates(monkeypatch, step):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Supply the graph-only TensorIndexSetitem protocol; execute actual CPU update ops.
    Expectation: The metadata boundary retains original arguments and the real CPU update produces the same
        replacement values as NumPy.
    """
    array = np.arange(12).reshape(3, 4)
    index = slice(0, 3, step)
    rows = np.arange(3)[index]
    coordinates = np.stack(np.meshgrid(rows, np.arange(4), indexing='ij'), -1)
    source, value = tensor(array), tensor(9)
    calls = []

    def metadata(data_arg, index_arg, value_arg):
        assert data_arg is source and index_arg is index and value_arg is value
        calls.append(True)
        return tensor(coordinates, ms.int64), (len(rows), 4), 0, 3, step, value_arg

    monkeypatch.setattr(cu, '_tensor_index_setitem', metadata)
    expected = array.copy()
    expected[index] = 9
    equal(cu.tensor_setitem_by_slice(source, index, value), expected)
    assert calls == [True]


def test_slice_setitem_empty_metadata_preserves_input(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Slice setitem empty metadata preserves input.
    Expectation: Empty slice metadata returns the source unchanged and preserves its exact arguments at the
        metadata boundary.
    """
    source, value = tensor([1, 2, 3]), tensor(9)
    calls = []
    index = slice(1, 1)

    def metadata(data_arg, index_arg, value_arg):
        assert data_arg is source and index_arg is index and value_arg is value
        calls.append(True)
        return tensor([], ms.int64), (0,), 1, 1, 1, value

    monkeypatch.setattr(cu, '_tensor_index_setitem', metadata)
    assert cu.tensor_setitem_by_slice(source, index, value) is source
    equal(source, [1, 2, 3])
    assert calls == [True]


@pytest.mark.parametrize(
    'name,container,index,expected',
    [
        ('_tuple_getitem_by_bool', (4, 5), False, 4),
        ('_tuple_getitem_by_bool', (4, 5), True, 5),
        ('_tuple_getitem_by_number', (4, 5), -1, 5),
        ('_list_getitem_by_bool', [4, 5], False, 4),
        ('_list_getitem_by_bool', [4, 5], True, 5),
        ('_list_getitem_by_number', [4, 5], -1, 5),
        ('_dict_getitem_by_key', {'a': 4}, 'a', 4),
        ('_string_getitem_by_number', 'abc', -1, 'c'),
    ],
)
def test_getitem_python_container_dispatch(name, container, index, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Getitem python container dispatch.
    Expectation: Python list, tuple and dictionary dispatch returns the expected selected element.
    """
    assert getattr(module('getitem'), name)(container, index) == expected


@pytest.mark.parametrize(
    'suffix,value',
    [
        ('string', 'x'),
        ('number', 7),
        pytest.param(
            'tensor',
            tensor(7, ms.int64),
            marks=pytest.mark.xfail(
                strict=True, reason='CPU list_setitem with Tensor replacement has no output address'
            ),
        ),
        ('list', [7]),
        ('tuple', (7,)),
    ],
)
def test_list_setitem_actual_primitive(suffix, value):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List setitem actual primitive.
    Expectation: The returned list contains the replacement at index one while the original list retains [1, 2,
        3].
    """
    source = [1, 2, 3]
    result = getattr(module('setitem'), '_list_setitem_with_' + suffix)(source, 1, value)
    assert result[0] == 1 and result[2] == 3
    if isinstance(value, ms.Tensor):
        assert int(result[1]) == int(value)
    else:
        assert result[1] == value
    assert source == [1, 2, 3]


@pytest.mark.parametrize(
    'step,message', [(None, 'can only assign'), (1, 'can only assign'), (2, 'extended slice'), (-1, 'extended slice')]
)
def test_list_slice_requires_iterable(step, message):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List slice requires iterable.
    Expectation: A noniterable slice replacement raises TypeError with the expected iterable-validation message.
    """
    with pytest.raises(TypeError, match=message):
        module('setitem')._list_slice_setitem_with_number([1, 2, 3], slice(None, None, step), 9)


@pytest.mark.parametrize(
    'index',
    [
        True,
        False,
        1,
        slice(0, 2),
        Ellipsis,
        None,
        tensor(1, ms.int32),
        tensor(True, ms.bool_),
        tensor(False, ms.bool_),
        tensor([0, 2], ms.int64),
    ],
)
def test_copy_native_boundary_receives_correct_view(monkeypatch, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Copy native boundary receives correct view.
    Expectation: The copy boundary receives the selected NumPy-equivalent view and scalar replacement while
        preserving source storage.
    """
    calls = []

    def native_copy(dst, src):
        calls.append((dst.asnumpy().copy(), src.asnumpy().copy()))
        return dst

    monkeypatch.setattr(cu, 'inplace_copy_op', native_copy)
    array = np.arange(12).reshape(3, 4)
    source = tensor(array)
    assert cu._process_with_do_copy(source, source, index, 0, 0, (), 0, tensor(9)) is source
    assert len(calls) == 1
    expected = array
    if isinstance(index, bool) or index is None or (isinstance(index, ms.Tensor) and index.dtype == ms.bool_):
        expected = array[None]
    elif isinstance(index, int) or (isinstance(index, ms.Tensor) and index.ndim == 0):
        expected = array[1]
    elif isinstance(index, slice):
        expected = array[:2]
    np.testing.assert_array_equal(calls[0][0], expected)
    np.testing.assert_array_equal(calls[0][1], 9)
    equal(source, array)


@pytest.mark.parametrize('src_shape', [(1, 1, 3), (1, 3), (3,), (), (2, 3)])
def test_do_copy_strips_only_leading_unit_dimensions(monkeypatch, src_shape):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Do copy strips only leading unit dimensions.
    Expectation: Copy retains destination identity and removes only leading unit dimensions from the source
        shape.
    """
    calls = []

    def native_copy(dst, src):
        calls.append((dst, src))
        return dst

    monkeypatch.setattr(cu, 'inplace_copy_op', native_copy)
    source = tensor(np.arange(np.prod(src_shape, dtype=int)).reshape(src_shape))
    destination = tensor(np.zeros((2, 3)))
    assert cu.do_copy(destination, source) is destination
    assert len(calls) == 1 and calls[0][0] is destination
    shape = src_shape
    if shape not in ((), (2, 3)):
        shape = (3,)
    equal(calls[0][1], source.asnumpy().reshape(shape))


@pytest.mark.parametrize('index', [([0, 2],), (np.array([0, 2], np.int8),), (slice(None), [0, 2]), ()])
def test_setitem_sequence_preparation_native_boundary(monkeypatch, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Setitem sequence preparation native boundary.
    Expectation: The native boundary receives the original replacement, correctly converted sequence indices and
        accumulation=False.
    """
    calls = []

    def native_put(data, indices, value, accumulate):
        calls.append((data, indices, value, accumulate))
        return data

    def native_copy(data, value):
        calls.append((data, (), value, False))
        return data

    monkeypatch.setattr(cu, 'inplace_index_put_op', native_put)
    monkeypatch.setattr(cu, 'inplace_copy_op', native_copy)
    source, value = tensor(np.arange(12).reshape(3, 4)), tensor(9)
    assert cu._process_multi_dim_index_for_setitem(source, index, (), len(index), value) is source
    assert len(calls) == 1 and calls[0][2] is value and calls[0][3] is False
    equal(calls[0][0], np.arange(12).reshape(3, 4))
    if index:
        equal(calls[0][1][-1], [0, 2])
        assert calls[0][1][-1].dtype in (ms.int32, ms.int64)
        if len(index) > 1:
            assert calls[0][1][0].shape == (0,) * 9
    else:
        assert calls[0][1] == ()


def test_list_update_and_gather_transfer_use_actual_broadcast_indices():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List update and gather transfer use actual broadcast indices.
    Expectation: Gather and scatter use actual broadcast coordinates; selected views, copies and formatted
        negative indices retain expected values.
    """
    source = tensor(np.arange(6).reshape(2, 3))
    new_index = [np.array([0, 1]), np.array([1, 2])]
    arg = ((2,),)
    equal(cu.data_update_by_ops(cu.ValueTransferType.kGatherND, arg, source, new_index, source), [1, 5])
    equal(
        cu.data_update_by_ops(
            cu.ValueTransferType.kTensorScatterUpdate, arg, source, new_index, source, tensor([8, 9])
        ),
        [[0, 8, 2], [3, 4, 9]],
    )
    equal(cu.data_update_by_ops(cu.ValueTransferType.kSelectView, (0, 0), source, None, source), [0, 1, 2])
    copy_target = tensor(np.arange(6).reshape(2, 3))
    assert cu.data_update_by_ops(cu.ValueTransferType.kCopyView, None, copy_target, None, source, tensor(9)) is source
    equal(copy_target, np.full((2, 3), 9))
    equal(
        cu.data_update(
            (cu.ValueTransferType.kFormatIndexTensor, cu.ValueTransferType.kGatherND),
            ((None, (2, 3)), None),
            source,
            tensor([[-1, -1]], ms.int64),
        ),
        [5],
    )


@pytest.mark.parametrize(
    'name,container', [('_tuple_getitem_by_slice', (1, 2, 3, 4)), ('_list_getitem_by_slice', [1, 2, 3, 4])]
)
@pytest.mark.parametrize('index', [slice(None), slice(None, None, -1), slice(1, 4, 2)])
def test_dynamic_sequence_slice_compiler_predicate_boundary(monkeypatch, name, container, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Dynamic sequence slice compiler predicate boundary.
    Expectation: The dynamic slice matches Python values and container type while consuming the original
        sequence at the compiler predicate.
    """
    calls = []

    def compiler_shape_unknown(sequence):
        assert sequence is container
        calls.append(sequence)
        return True

    monkeypatch.setattr(module('getitem').F, 'is_sequence_shape_unknown', compiler_shape_unknown)
    result = getattr(module('getitem'), name)(container, index)
    assert result == container[index]
    assert type(result) is type(container)
    assert len(calls) == 1


@pytest.mark.parametrize(
    'mod,name,left,right,expected',
    [
        ('equal', '_equal_mstype', ms.int32, ms.int32, True),
        ('not_equal', '_not_equal_mstype', ms.int32, ms.float32, True),
        *[
            (mod, name, left, right, expected)
            for mod, expected, names in [
                (
                    'equal',
                    False,
                    [
                        '_string_equal_none',
                        '_scalar_equal_none',
                        '_tuple_euqal_none',
                        '_tensor_equal_none',
                        '_list_equal_none',
                    ],
                ),
                (
                    'not_equal',
                    True,
                    [
                        '_string_not_equal_none',
                        '_scalar_not_equal_none',
                        '_tuple_not_euqal_none',
                        '_tensor_not_equal_none',
                        '_list_not_equal_none',
                    ],
                ),
            ]
            for name, left, right in zip(names, ['x', 3, (1,), tensor([1]), [1]], [None] * 5)
        ],
        *[
            (mod, name, left, right, expected)
            for mod, expected, names in [
                (
                    'equal',
                    False,
                    [
                        '_none_equal_string',
                        '_none_equal_scalar',
                        '_none_equal_tuple',
                        '_none_equal_tensor',
                        '_none_equal_list',
                    ],
                ),
                (
                    'not_equal',
                    True,
                    [
                        '_none_not_equal_string',
                        '_none_not_equal_scalar',
                        '_none_not_equal_tuple',
                        '_none_not_equal_tensor',
                        '_none_not_equal_list',
                    ],
                ),
            ]
            for name, left, right in zip(names, [None] * 5, ['x', 3, (1,), tensor([1]), [1]])
        ],
        ('equal', '_number_equal_string', 3, '3', False),
        ('equal', '_string_equal_number', '3', 3, False),
    ],
)
def test_none_and_dtype_comparison_dispatch(mod, name, left, right, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: None and dtype comparison dispatch.
    Expectation: None and dtype comparisons return the exact expected Python boolean.
    """
    assert getattr(module(mod), name)(left, right) is expected


@pytest.mark.parametrize(
    'mod,name,op',
    [
        ('add', 'default_add', operator.add),
        ('sub', 'default_sub', operator.sub),
        ('mul', 'default_mul', operator.mul),
        ('div', 'default_div', operator.truediv),
        ('mod', 'default_mod', operator.mod),
        ('pow', 'default_pow', operator.pow),
        ('floordiv', 'default_floordiv', operator.floordiv),
        ('bitwise_and', 'default_bitwsie_add', operator.and_),
        ('bitwise_or', 'default_bitwsie_or', operator.or_),
        ('bitwise_xor', 'default_bitwsie_xor', operator.xor),
        ('left_shift', 'default_left_shift', operator.lshift),
        ('right_shift', 'default_right_shift', operator.rshift),
    ],
)
def test_default_arithmetic_dispatch_and_zero_division(mod, name, op):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Default arithmetic dispatch and zero division.
    Expectation: Fallback arithmetic matches Python operations and zero division raises the documented
        ValueError.
    """
    assert getattr(module(mod), name)(7, 2) == op(7, 2)
    if mod in ('div', 'floordiv'):
        with pytest.raises(ValueError, match='division by zero'):
            getattr(module(mod), name)(7, 0)


@pytest.mark.parametrize('value', [3, 1.5])
def test_ones_and_zeros_like_scalars(value):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Ones and zeros like scalars.
    Expectation: Scalar identities retain numeric types; metadata, None and slice cases preserve their
        documented identity behavior.
    """
    zero = module('zeros_like')._zeros_like_scalar(value)
    assert zero == type(value)(0) and type(zero) is type(value)
    assert module('zeros_like')._zeros_like_bool(True) is False
    assert module('zeros_like')._zeros_like_string('a') == 0
    assert module('ones_like')._ones_like_type_type(ms.int32) is ms.int32
    assert module('zeros_like')._zeros_like_type_type(ms.int32) is ms.int32
    assert module('ones_like')._ones_like_none(None) is None
    token = object()
    assert module('ones_like').default_ones_like_leaf_for_grad(token) is token
    slice_arg = slice(1, 3)
    assert module('zeros_like')._zeros_like_slice(slice_arg) is slice_arg
    equal(module('ones_like')._ones_like_tensor(tensor([[2, 3]])), [[1, 1]])
    equal(module('zeros_like')._zeros_like_tensor(tensor([[2, 3]])), [[0, 0]])
    assert module('zeros_like').zeros_like(3) == 0


@pytest.mark.xfail(strict=True, reason='_zeros_like_dict passes dict_keys to sequence-only MakeDict primitive')
def test_zeros_like_dictionary_preserves_keys_and_typed_values():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Zeros like dictionary preserves keys and typed values.
    Expectation: Dictionary keys remain intact with exact integer, float and boolean zero value types.
    """
    result = module('zeros_like')._zeros_like_dict({'count': 3, 'ratio': 2.5, 'enabled': True})
    assert result == {'count': 0, 'ratio': 0.0, 'enabled': False}
    # Exact Python result types are part of this conversion contract.
    # pylint: disable-next=unidiomatic-typecheck
    assert type(result['count']) is int and type(result['ratio']) is float and type(result['enabled']) is bool


def test_coo_arithmetic_helpers_preserve_sparse_structure():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: COO arithmetic helpers preserve sparse structure.
    Expectation: COO arithmetic retains its indices and shape while computing the independently expected value
        array.
    """
    indices = tensor([[0, 1], [1, 0]], ms.int32)
    values = tensor([2, 3])
    sparse = ms.COOTensor(indices, values, (2, 2))
    dense = tensor([[1, 4], [5, 2]])
    dense_np = dense.asnumpy()
    sparse_np = np.array([[0, 2], [3, 0]], np.float32)
    for fn, left, right, expected in [
        (cu._tensor_add, dense, sparse, dense_np + sparse_np),
        (cu._tensor_sub, dense, sparse, dense_np - sparse_np),
        (module('add')._add_cootensor_tensor, sparse, dense, dense_np + sparse_np),
        (module('add')._add_tensor_cootensor, dense, sparse, dense_np + sparse_np),
        (module('sub')._tensor_sub_cootensor, dense, sparse, dense_np - sparse_np),
        (module('sub')._cootensor_sub_tensor, sparse, dense, sparse_np - dense_np),
    ]:
        equal(fn(left, right), expected)
    for fn, args, expected in [
        (cu._tensor_mul, (dense, sparse), [8, 15]),
        (module('mul')._cootensor_mul_tensor, (sparse, dense), [8, 15]),
        (module('mul')._tensor_mul_cootensor, (dense, sparse), [8, 15]),
        (module('div')._cootensor_div_tensor, (sparse, dense), [0.5, 0.6]),
        (module('negative')._negative_cootensor, (sparse,), [-2, -3]),
    ]:
        result = fn(*args)
        assert isinstance(result, ms.COOTensor) and result.shape == (2, 2)
        equal(result.indices, indices.asnumpy(), ms.int32)
        equal(result.values, expected, ms.float32)


@pytest.mark.parametrize('value', [3, 1.5])
def test_ones_like_scalar_retains_python_numeric_type(value, request):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Ones like scalar retains python numeric type.
    Expectation: The result equals numeric one with the same exact Python scalar type as its input.
    """
    scalar_dtype = cu.F.typeof(value)
    # ScalarCast checks this whitelist before accessing NumPy, so the Float32
    # failure also applies when NumPy 1.x still provides np.cast.
    if scalar_dtype not in (ms.int64, ms.float64, ms.bool_):
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason=f'ScalarCast rejects inferred {scalar_dtype} before NumPy conversion')
        )
    elif not hasattr(np, 'cast'):
        request.node.add_marker(pytest.mark.xfail(strict=True, reason='NumPy 2 removed np.cast used by ScalarCast'))
    actual = module('ones_like')._ones_like_scalar(value)
    assert actual == type(value)(1) and type(actual) is type(value)


@pytest.mark.parametrize('kind', ['ones_like', 'zeros_like'])
def test_coo_like_graph_constructor_boundary(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: COO like graph constructor boundary.
    Expectation: The constructor retains COO indices and dense shape and replaces values with typed ones or
        zeros.
    """
    source = ms.COOTensor(tensor([[0, 1], [1, 0]], ms.int32), tensor([2, 3]), (2, 2))
    calls = []

    def make_coo(indices, values, shape):
        equal(indices, source.indices.asnumpy(), ms.int32)
        assert tuple(shape) == source.shape
        calls.append((indices, values, shape))
        return ms.COOTensor(indices, values, shape)

    monkeypatch.setattr(module(kind).F, 'make_coo_tensor', make_coo)
    result = getattr(module(kind), '_' + kind + '_coo_tensor')(source)
    assert isinstance(result, ms.COOTensor) and result.shape == (2, 2)
    equal(result.indices, [[0, 1], [1, 0]], ms.int32)
    equal(result.values, [1, 1] if kind == 'ones_like' else [0, 0], ms.float32)
    assert len(calls) == 1


@pytest.mark.parametrize(
    'scenario',
    [
        'normal',
        'empty_mask',
        'special_zero',
        'special_one',
        'empty_tuple',
        'tensor_empty_shape',
        'tuple_empty_shape',
        'parameter',
    ],
)
def test_tuple_setitem_compiler_metadata_contract_and_real_scatter(monkeypatch, scenario):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Native metagraphs supply metadata; casts, reshapes and scatters remain real.
    Expectation: Tuple metadata retains original source and index arguments, empty selections preserve identity,
        and real scatters match NumPy updates.
    """
    array = np.arange(12).reshape(3, 4)
    source = tensor(array)
    if scenario == 'parameter':
        source = ms.Parameter(source, name='tuple_metadata')
    index = () if scenario == 'empty_tuple' else (tensor([0, 2], ms.int64),)
    value = tensor(9, ms.int32)
    calls = []

    def bool_metadata(actual_index):
        assert actual_index is index
        calls.append('bool')
        return index, (), ((0,),) if scenario == 'empty_mask' else ((2,),)

    def pre_metadata(data_arg, index_arg, value_arg):
        assert data_arg is source and index_arg is index
        equal(value_arg, 9, ms.float32)
        calls.append('pre')
        special = 0 if scenario == 'special_zero' else (1 if scenario == 'special_one' else 2)
        shape = (
            tensor([0], ms.int64)
            if scenario == 'tensor_empty_shape'
            else (() if scenario == 'tuple_empty_shape' else (2, 4))
        )
        return special, index, (), 0, shape

    def index_metadata(data_arg, index_arg, value_arg, fancy_position, empty_shape):
        assert data_arg is source and index_arg is index
        equal(value_arg, 9, ms.float32)
        assert fancy_position == 0
        assert empty_shape is (scenario in ('tensor_empty_shape', 'tuple_empty_shape'))
        calls.append('index')
        return tensor([[0], [2]], ms.int64)

    monkeypatch.setattr(cu, '_handle_bool_tensor', bool_metadata)
    monkeypatch.setattr(cu, '_pre_setitem_by_tuple', pre_metadata)
    monkeypatch.setattr(cu, '_tensor_index_setitem', index_metadata)
    result = cu.tensor_setitem_by_tuple_with_tensor(source, index, value)
    expected = array.copy()
    if scenario in ('empty_mask', 'special_zero'):
        assert result is source
    elif scenario in ('special_one', 'empty_tuple'):
        expected[:] = 9
    else:
        expected[[0, 2]] = 9
    equal(result, expected)
    if scenario == 'parameter':
        assert result is source
        equal(source, expected)
    if scenario == 'empty_mask':
        assert calls == ['bool']
    elif scenario in ('special_zero', 'special_one', 'empty_tuple'):
        assert calls == ['bool', 'pre']
    else:
        assert calls == ['bool', 'pre', 'index']


@pytest.mark.parametrize(
    'name,index,value',
    [
        ('_tensor_setitem_by_tensor_with_tensor', tensor(0, ms.int32), tensor(9)),
        ('_tensor_setitem_by_tensor_with_number', tensor(0, ms.int32), 9),
        ('_tensor_setitem_by_tuple_with_number', (0, slice(None)), 9),
        ('_tensor_setitem_by_tuple_with_tensor', (0, slice(None)), tensor(9)),
        ('_tensor_setitem_by_slice_with_tensor', slice(0, 2), tensor(9)),
        ('_tensor_setitem_by_slice_with_number', slice(0, 2), 9),
        ('_tensor_setitem_by_number_with_number', 0, 9),
        ('_tensor_setitem_by_number_with_tensor', 0, tensor(9)),
        ('_tensor_setitem_by_ellipsis_with_number', Ellipsis, 9),
        ('_tensor_setitem_by_ellipsis_with_tensor', Ellipsis, tensor(9)),
        ('_tensor_setitem_by_none_with_number', None, 9),
        ('_tensor_setitem_by_none_with_tensor', None, tensor(9)),
    ],
)
def test_setitem_boost_contiguous_views_update_real_storage(monkeypatch, name, index, value):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Setitem boost contiguous views update real storage.
    Expectation: Contiguous assignment through boosted dispatch produces the same updated tensor values as
        NumPy.
    """
    monkeypatch.setenv('MS_DEV_TENSOR_INDEX_BOOST', '1')
    array = np.arange(12).reshape(3, 4)
    expected = array.copy()
    expected[np_index(index)] = np_index(value)
    equal(getattr(module('setitem'), name)(tensor(array), index, value), expected)


@pytest.mark.parametrize('kind', ['tensor', 'number', 'list', 'tuple'])
def test_slice_dispatch_legacy_native_metadata_boundary(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Slice dispatch legacy native metadata boundary.
    Expectation: Legacy slice dispatch retains original data and index arguments and produces the expected real
        tensor update.
    """
    monkeypatch.delenv('MS_DEV_TENSOR_INDEX_BOOST', raising=False)
    array = np.arange(12).reshape(3, 4)
    source, index = tensor(array), slice(0, 2)
    value = (
        tensor(9)
        if kind == 'tensor'
        else (9 if kind == 'number' else ([7, 8, 9, 10] if kind == 'list' else (7, 8, 9, 10)))
    )
    calls = []

    def metadata(data_arg, index_arg, value_arg):
        assert data_arg is source and index_arg is index
        if isinstance(value, ms.Tensor):
            assert value_arg is value
        else:
            assert value_arg == value
        calls.append(True)
        converted = value_arg if isinstance(value_arg, ms.Tensor) else tensor(value_arg)
        return tensor([[0], [1]], ms.int64), (2, 4), 0, 2, 1, converted

    monkeypatch.setattr(cu, '_tensor_index_setitem', metadata)
    expected = array.copy()
    expected[index] = np_index(value)
    equal(getattr(module('setitem'), '_tensor_setitem_by_slice_with_' + kind)(source, index, value), expected)
    assert calls == [True]


def test_dynamic_scalar_compiler_predicate_then_actual_tensor_add(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Dynamic scalar compiler predicate then actual tensor add.
    Expectation: The compiler predicate receives integer three once and real CPU conversion returns the same
        value in int64.
    """
    calls = []

    def isconstant(value):
        assert value == 3
        calls.append(value)
        return False

    monkeypatch.setattr(cu.ops, 'isconstant', isconstant)
    equal(cu._scalar_to_tensor(3), 3, ms.int64)
    assert calls == [3]


def test_unknown_shape_copy_compiler_predicate_boundary(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Unknown shape copy compiler predicate boundary.
    Expectation: The shape predicate receives (2,) once and real copying updates both destination and result to
        [1, 2].
    """
    source, destination = tensor([1, 2]), tensor([0, 0])
    calls = []

    def unknown_shape(shape):
        assert shape == (2,)
        calls.append(shape)
        return True

    monkeypatch.setattr(cu.F, 'is_sequence_value_unknown', unknown_shape)
    result = cu.do_copy(destination, source)
    equal(result, [1, 2])
    equal(destination, [1, 2])
    assert calls == [(2,)]


@pytest.mark.parametrize('for_setitem', [False, True])
def test_mutable_sequence_compiler_predicate_rejects_index(monkeypatch, for_setitem):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Mutable sequence compiler predicate rejects index.
    Expectation: A mutable sequence index raises the documented IndexError after exactly one predicate call with
        the original object.
    """
    index = [0, 1]
    calls = []

    def isconstant(value):
        assert value is index
        calls.append(value)
        return False

    monkeypatch.setattr(cu.F, 'isconstant', isconstant)
    source = tensor(np.zeros((2, 3)))
    with pytest.raises(IndexError, match='mutable list/tuple'):
        if for_setitem:
            cu._process_multi_dim_index_for_setitem(source, (index,), (), 1, tensor(9))
        else:
            cu._process_multi_dim_index(source, (index,), (), 1)
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['_list_mul_scalar', '_scalar_mul_list', '_tuple_mul_scalar', '_scalar_mul_tuple'])
def test_dynamic_sequence_repetition_native_kernel(monkeypatch, name):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Dynamic sequence repetition native kernel.
    Expectation: The real native repetition result equals Python repetition and retains the exact list or tuple
        type.
    """
    sequence = [1, 2] if 'list' in name else (1, 2)
    calls = []

    def unknown_shape(value):
        assert value is sequence
        calls.append(value)
        return True

    monkeypatch.setattr(module('mul').F, 'is_sequence_shape_unknown', unknown_shape)
    args = (2, sequence) if name.startswith('_scalar') else (sequence, 2)
    result = getattr(module('mul'), name)(*args)
    assert result == sequence * 2 and type(result) is type(sequence)
    assert len(calls) == 1


@pytest.mark.parametrize('mod,kind', [('in', 'tuple'), ('in', 'list'), ('not_in', 'tuple'), ('not_in', 'list')])
def test_dynamic_membership_native_kernel(monkeypatch, mod, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Dynamic membership native kernel.
    Expectation: The real native membership result preserves inclusion or negation semantics and the predicate
        receives the original sequence.
    """
    sequence = [1, 3] if kind == 'list' else (1, 3)
    calls = []

    def unknown_shape(value):
        assert value is sequence
        calls.append(value)
        return True

    monkeypatch.setattr(module(mod).F, 'is_sequence_value_unknown', unknown_shape)
    name = '_number_in_' + kind if mod == 'in' else '_number_not_in_' + kind
    assert bool(getattr(module(mod), name)(1, sequence)) is (mod == 'in')
    assert len(calls) == 1


def test_list_tensor_index_graph_primitive_receives_python_integer(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List tensor index graph primitive receives python integer.
    Expectation: The graph primitive receives Python index one and the original data and value, returning the
        expected list replacement.
    """
    source, value, index = [1, 2, 3], tensor(9), tensor(1, ms.int64)
    calls = []

    def list_setitem(data_arg, index_arg, value_arg):
        # Exact Python result types are part of this conversion contract.
        # pylint: disable-next=unidiomatic-typecheck
        assert data_arg is source and type(index_arg) is int and index_arg == 1 and value_arg is value
        calls.append(index_arg)
        return [1, value, 3]

    monkeypatch.setattr(module('setitem').F, 'list_setitem', list_setitem)
    result = module('setitem')._list_setitem_with_tensor_index(source, index, value)
    assert result[0] == 1 and result[1] is value and result[2] == 3 and calls == [1]


@pytest.mark.parametrize(
    'index',
    [
        (tensor([0, 2], ms.int64), tensor([1, 3], ms.int64)),
        (slice(None), tensor([0, 2], ms.int64)),
        pytest.param(
            (slice(1, 1), tensor([0, 2], ms.int64)),
            marks=pytest.mark.xfail(
                strict=True, reason='legacy mixed indexing rejects broadcastable empty slice and two tensor indices'
            ),
        ),
        (False,),
        (0,),
        (None, Ellipsis),
    ],
)
def test_itemset_tuple_mixed_indices_execute_real_scatter(index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Itemset tuple mixed indices execute real scatter.
    Expectation: Mixed tuple itemset produces the same scattered replacement values as NumPy.
    """
    array = np.arange(12).reshape(3, 4)
    expected = array.copy()
    expected[np_index(index)] = 9
    equal(cu.tensor_itemset_by_tuple_with_tensor(tensor(array), index, tensor(9)), expected)


@pytest.mark.parametrize('empty', [False, True])
def test_tuple_getitem_graph_metadata_boundary(monkeypatch, empty):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Tuple getitem graph metadata boundary.
    Expectation: The tuple path requests metadata in order and returns the intended selection, including the
        empty-index normalization case.
    """
    source = tensor(np.arange(12).reshape(3, 4))
    index = (tensor([True, False, True], ms.bool_), slice(None))
    zero_index = (tensor([], ms.int64), slice(None))
    calls = []

    def bool_metadata(index_arg):
        assert index_arg is index
        calls.append('bool')
        return index, zero_index, ((0,),) if empty else ((2,),)

    def getitem_metadata(data_arg, index_arg):
        assert data_arg is source and index_arg is (zero_index if empty else index)
        calls.append('getitem')
        return ops.gather(data_arg, tensor([], ms.int64) if empty else tensor([0, 2], ms.int64), 0)

    monkeypatch.setattr(cu, '_handle_bool_tensor', bool_metadata)
    monkeypatch.setattr(cu, '_tensor_index_getitem', getitem_metadata)
    expected = np.empty((0, 4)) if empty else np.arange(12).reshape(3, 4)[[0, 2]]
    equal(cu.tensor_index_by_tuple(source, index), expected)
    assert calls == ['bool', 'getitem']


@pytest.mark.parametrize(
    'function,index', [('tensor_index_by_list', [slice(None), 1]), ('tensor_setitem_by_list', [slice(None), 1])]
)
def test_list_mixed_index_graph_tuple_boundary(monkeypatch, function, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List mixed index graph tuple boundary.
    Expectation: Mixed list indices become the expected tuple at each boundary, and selection or empty
        replacement preserves the intended values.
    """
    source, value = tensor(np.arange(12).reshape(3, 4)), tensor(9)
    calls = []

    def bool_metadata(index_arg):
        assert index_arg == tuple(index)
        calls.append('bool')
        return index_arg, (), ()

    monkeypatch.setattr(cu, '_handle_bool_tensor', bool_metadata)
    if function == 'tensor_index_by_list':

        def getitem_metadata(data_arg, index_arg):
            assert data_arg is source and index_arg == tuple(index)
            calls.append('getitem')
            return ops.gather(data_arg, tensor(1, ms.int32), 1)

        monkeypatch.setattr(cu, '_tensor_index_getitem', getitem_metadata)
        equal(cu.tensor_index_by_list(source, index), [1, 5, 9])
        assert calls == ['bool', 'getitem']
    else:

        def pre_metadata(data_arg, index_arg, value_arg):
            assert data_arg is source and index_arg == tuple(index)
            equal(value_arg, 9)
            calls.append('pre')
            return 0, index_arg, (), 0, (3,)

        monkeypatch.setattr(cu, '_pre_setitem_by_tuple', pre_metadata)
        assert cu.tensor_setitem_by_list(source, index, value) is source
        equal(source, np.arange(12).reshape(3, 4))
        assert calls == ['bool', 'pre']


def test_empty_bool_tensor_and_separated_advanced_indices():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Empty bool tensor and separated advanced indices.
    Expectation: Empty boolean conversion preserves empty metadata, and separated advanced indices keep the
        expected broadcast assignment shape.
    """
    index = tensor([False, False, False], ms.bool_)
    result, shapes, dim = cu.remove_expanded_dims_parse_bool_tensor_index(index, (), [], 0)
    assert result is None and shapes == [] and dim == 0
    indices = (tensor([0, 2], ms.int64), slice(None), tensor([1, 3], ms.int64))
    converted, value, advanced = cu.remove_expanded_dims(indices, (3, 2, 4), tensor(np.full((2, 2), 9)))
    assert advanced == 0 and len(converted) == 3
    equal(value, np.full((2, 2), 9))


@pytest.mark.parametrize(
    'kind,order',
    [
        ('coo', 'sparse_sparse'),
        ('coo', 'sparse_tuple'),
        ('csr', 'sparse_sparse'),
        ('csr', 'sparse_tuple'),
        ('csr', 'tuple_sparse'),
    ],
)
def test_sparse_backward_add_graph_constructor_boundary(monkeypatch, kind, order):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Sparse backward add graph constructor boundary.
    Expectation: Sparse gradient addition retains COO or CSR storage metadata and sums values to [6, 8] through
        one constructor request.
    """
    add = module('add')
    values = tensor([2, 3])
    shape = (2, 2)
    if kind == 'coo':
        indices = tensor([[0, 1], [1, 0]], ms.int32)
        sparse = ms.COOTensor(indices, values, shape)
        components = (indices, tensor([4, 5]), shape)
        other = ms.COOTensor(*components)
    else:
        indptr, indices = tensor([0, 1, 2], ms.int32), tensor([1, 0], ms.int32)
        sparse = ms.CSRTensor(indptr, indices, values, shape)
        components = (indptr, indices, tensor([4, 5]), shape)
        other = ms.CSRTensor(*components)
    calls = []

    def construct(*args):
        calls.append(args)
        if kind == 'coo':
            equal(args[0], indices.asnumpy())
            assert tuple(args[2]) == shape
            return ms.COOTensor(*args)
        equal(args[0], indptr.asnumpy())
        equal(args[1], indices.asnumpy())
        assert tuple(args[3]) == shape
        return ms.CSRTensor(*args)

    monkeypatch.setattr(add.F, 'make_coo_tensor' if kind == 'coo' else 'make_csr_tensor', construct)
    if order == 'sparse_sparse':
        function = '_add_cootensor_cootensor' if kind == 'coo' else '_add_csrtensor_csrtensor'
        args = sparse, other
    elif order == 'sparse_tuple':
        function = '_add_cootensor_tuple' if kind == 'coo' else '_add_csrtensor_tuple'
        args = sparse, components
    else:
        function = '_add_tuple_cootensor' if kind == 'coo' else '_add_tuple_csrtensor'
        args = components, sparse
    result = getattr(add, function)(*args)
    assert type(result) is type(sparse) and result.shape == shape and len(calls) == 1
    equal(result.values, [6, 8])
    equal(result.indices, indices.asnumpy())
    if kind == 'csr':
        equal(result.indptr, indptr.asnumpy())


@pytest.mark.xfail(strict=True, reason='_add_tuple_cootensor misspells check_coo_tensor_input_length')
def test_tuple_cootensor_backward_add_accepts_valid_components():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Tuple cootensor backward add accepts valid components.
    Expectation: Tuple components reconstruct a COO tensor of shape (2, 2) with summed value five.
    """
    indices = tensor([[0, 1]], ms.int32)
    sparse = ms.COOTensor(indices, tensor([2]), (2, 2))
    result = module('add')._add_tuple_cootensor((indices, tensor([3]), (2, 2)), sparse)
    assert isinstance(result, ms.COOTensor) and result.shape == (2, 2)
    equal(result.values, [5])


@pytest.mark.xfail(strict=True, reason='_add_nonetensor_tensor attempts unsupported None + None')
def test_none_backward_add_preserves_absent_gradient():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: None backward add preserves absent gradient.
    Expectation: Adding two absent gradients returns None.
    """
    assert module('add')._add_nonetensor_tensor(None, None) is None


@pytest.mark.parametrize('kind', ['list', 'tuple'])
def test_dynamic_sequence_add_real_native_kernel(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Dynamic sequence add real native kernel.
    Expectation: The real sequence-add kernel returns Python concatenation with the exact input container type
        and one metadata query.
    """
    left, right = ([1, 2], [3, 4]) if kind == 'list' else ((1, 2), (3, 4))
    calls = []

    def unknown_shape(value):
        assert value is left
        calls.append(value)
        return True

    monkeypatch.setattr(module('add').F, 'is_sequence_shape_unknown', unknown_shape)
    fn = module('add')._list_add_list if kind == 'list' else module('add')._add_tuple
    result = fn(left, right)
    assert result == left + right and type(result) is type(left) and len(calls) == 1


def test_addn_backward_real_tensors():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Addn backward real tensors.
    Expectation: The real AddN helper sums [1, 3] and [2, 4] to [3, 7].
    """
    equal(module('add')._add_addn(tensor([1, 3]), tensor([2, 4])), [3, 7])


@pytest.mark.parametrize('index', [True, False, tensor([0, 2], ms.int64), (slice(None), tensor([1, 3], ms.int32))])
def test_index_native_boundary_preserves_prepared_data_and_arguments(monkeypatch, index):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Index native boundary preserves prepared data and arguments.
    Expectation: The native boundary receives the expected view and remaining indices, returns its sentinel and
        leaves source storage unchanged.
    """
    array = np.arange(12).reshape(3, 4)
    source, sentinel = tensor(array), tensor([123, 456])
    calls = []

    def native_index(viewed, remaining):
        equal(viewed, array[None] if isinstance(index, bool) else array)
        if isinstance(index, bool):
            assert len(remaining) == 1
            equal(remaining[0], [0] if index else [], ms.int64)
        elif isinstance(index, ms.Tensor):
            assert len(remaining) == 1 and remaining[0] is index
        else:
            assert len(remaining) == 2 and remaining[0].shape == (0,) * 9
            assert remaining[1] is index[1]
        calls.append(True)
        return sentinel

    monkeypatch.setattr(cu, 'index_op', native_index)
    assert cu._tensor_getitem(source, index) is sentinel
    equal(sentinel, [123, 456])
    equal(source, array)
    assert calls == [True]


@pytest.mark.parametrize('kind', ['tuple', 'list'])
def test_static_sequence_slice_native_metagraph_boundary(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Static sequence slice native metagraph boundary.
    Expectation: The static slice boundary receives the original container and slice and returns its sentinel
        with the matching container type.
    """
    source = (1, 2, 3) if kind == 'tuple' else [1, 2, 3]
    index, sentinel = slice(1, 3), (7, 8) if kind == 'tuple' else [7, 8]
    calls = []

    def native_slice(data_arg, index_arg):
        assert data_arg is source and index_arg is index
        calls.append(True)
        return sentinel

    monkeypatch.setattr(module('getitem'), '_tuple_slice' if kind == 'tuple' else '_list_slice', native_slice)
    result = getattr(module('getitem'), '_' + kind + '_getitem_by_slice')(source, index)
    assert result is sentinel and result == ((7, 8) if kind == 'tuple' else [7, 8])
    assert calls == [True]


@pytest.mark.parametrize('kind', ['tuple', 'list', 'tensor'])
@pytest.mark.parametrize('dynamic', [False, True])
def test_list_slice_assignment_native_metagraph_boundary(monkeypatch, kind, dynamic):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: List slice assignment native metagraph boundary.
    Expectation: The metagraph receives the normalized slice and int64 replacement tensors, yielding [8, 9, 3]
        without mutating the source.
    """
    source, index = [1, 2, 3], slice(0, 2, 1)
    value = (8, 9) if kind == 'tuple' else ([8, 9] if kind == 'list' else tensor([8, 9], ms.int64))
    calls = []

    def validate(data_arg, value_arg):
        assert data_arg is source
        assert [int(x) for x in value_arg] == [8, 9]
        if kind == 'tensor':
            assert all(isinstance(x, ms.Tensor) and x.dtype == ms.int64 for x in value_arg)

    if dynamic:

        def unknown_shape(sequence):
            assert sequence is source
            return True

        def native_slice(data_arg, value_arg, start, stop, step):
            validate(data_arg, value_arg)
            assert (start, stop, step) == (0, 2, 1)
            calls.append(True)
            return [8, 9, 3]

        monkeypatch.setattr(module('setitem').F, 'is_sequence_shape_unknown', unknown_shape)
        monkeypatch.setattr(module('setitem'), 'sequence_slice_setitem', native_slice)
    else:

        def native_slice(data_arg, index_arg, value_arg):
            validate(data_arg, value_arg)
            assert index_arg is index and isinstance(value_arg, list)
            calls.append(True)
            return [8, 9, 3]

        monkeypatch.setattr(module('setitem'), '_list_slice_set_item', native_slice)
    result = getattr(module('setitem'), '_list_slice_setitem_with_' + kind)(source, index, value)
    assert result == [8, 9, 3] and source == [1, 2, 3] and calls == [True]


@pytest.mark.parametrize(
    'mod,name,args,expected',
    [
        ('mul', '_csrtensor_mul_tensor', 'sparse_dense', [8, 15]),
        ('mul', '_tensor_mul_csrtensor', 'dense_sparse', [8, 15]),
        ('div', '_csrtensor_div_tensor', 'sparse_dense', [0.5, 0.6]),
        ('negative', '_negative_csrtensor', 'sparse', [-2, -3]),
        ('compile', '_tensor_mul', 'dense_sparse', [8, 15]),
    ],
)
def test_csr_arithmetic_real_values(mod, name, args, expected):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: CSR arithmetic real values.
    Expectation: Real CSR arithmetic preserves int32 row pointers and column indices and computes the expected
        float32 stored values.
    """
    sparse = ms.CSRTensor(tensor([0, 1, 2], ms.int32), tensor([1, 0], ms.int32), tensor([2, 3]), (2, 2))
    dense = tensor([[1, 4], [5, 2]])
    arguments = (sparse,) if args == 'sparse' else ((sparse, dense) if args == 'sparse_dense' else (dense, sparse))
    try:
        result = getattr(cu if mod == 'compile' else module(mod), name)(*arguments)
        result.values.asnumpy()
    except RuntimeError as error:
        if 'found LLVM 12.0.1 not installed' in str(error):
            pytest.skip('CSR AKG CPU kernel requires unavailable LLVM 12.0.1')
        raise
    assert isinstance(result, ms.CSRTensor) and result.shape == (2, 2)
    equal(result.indptr, [0, 1, 2], ms.int32)
    equal(result.indices, [1, 0], ms.int32)
    equal(result.values, expected, ms.float32)


def test_nested_object_tensor_conversion_rejects_unsupported_elements():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Nested object tensor conversion rejects unsupported elements.
    Expectation: A nested sequence with unsupported unequal element shapes raises the documented ValueError.
    """
    with pytest.raises(ValueError, match='same size across all dimensions'):
        co.make_tensor([[object()]])


def test_itemset_full_coordinate_and_leading_unit_broadcast():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Itemset full coordinate and leading unit broadcast.
    Expectation: Full-coordinate itemset updates only the selected element and leading-unit replacements
        broadcast to the expected full array.
    """
    array = np.arange(6).reshape(2, 3)
    equal(cu.tensor_itemset(tensor(array), (1, 2), 9), [[0, 1, 2], [3, 4, 9]])
    value = tensor(np.arange(6).reshape(1, 2, 3))
    equal(cu.tensor_setitem_by_ellipsis_with_tensor(tensor(np.zeros((2, 3))), value), array)
    equal(cu.tensor_setitem_by_bool(tensor(np.zeros((2, 3))), True, value), array)


@pytest.mark.parametrize('kind', ['coo', 'csr'])
@pytest.mark.parametrize('operation', ['add', 'sub'])
def test_sparse_binary_helpers_validate_shapes_before_native_operation(kind, operation):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Sparse binary helpers validate shapes before native operation.
    Expectation: Sparse operands with different dense shapes raise ValueError before the native arithmetic
        boundary.
    """
    if kind == 'coo':
        left = ms.COOTensor(tensor([[0, 1]], ms.int32), tensor([2]), (2, 2))
        right = ms.COOTensor(tensor([[0, 1]], ms.int32), tensor([3]), (2, 3))
    else:
        left = ms.CSRTensor(tensor([0, 1, 1], ms.int32), tensor([1], ms.int32), tensor([2]), (2, 2))
        right = ms.CSRTensor(tensor([0, 1, 1], ms.int32), tensor([1], ms.int32), tensor([3]), (2, 3))
    with pytest.raises(ValueError, match='same shape'):
        getattr(module(operation), '_' + operation + '_' + kind + 'tensor')(left, right)


def test_empty_bool_tensor_removes_all_updates():
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Empty bool tensor removes all updates.
    Expectation: An empty boolean tensor produces the False update sentinel and preserves the original
        replacement tensor.
    """
    value = tensor(9)
    indices, actual_value, advanced = cu.remove_expanded_dims((tensor([False] * 3, ms.bool_),), (3, 4), value)
    assert indices is False and actual_value is value and advanced == 0
    equal(actual_value, 9)


@pytest.mark.parametrize('kind', ['tensor', 'tuple', 'list'])
def test_legacy_boolean_number_dispatch_broadcasts_values(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Legacy boolean number dispatch broadcasts values.
    Expectation: A true boolean numeric index broadcasts each replacement kind across the expected tensor shape.
    """
    monkeypatch.delenv('MS_DEV_TENSOR_INDEX_BOOST', raising=False)
    value = tensor(9) if kind == 'tensor' else ((7, 8, 9, 10) if kind == 'tuple' else [7, 8, 9, 10])
    expected = np.full((3, 4), 9) if kind == 'tensor' else np.tile([7, 8, 9, 10], (3, 1))
    equal(
        getattr(module('setitem'), '_tensor_setitem_by_number_with_' + kind)(tensor(np.zeros((3, 4))), True, value),
        expected,
    )


@pytest.mark.parametrize('kind', ['ones_like', 'zeros_like'])
def test_csr_like_graph_constructor_boundary(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: CSR like graph constructor boundary.
    Expectation: CSR ones or zeros preserve shape, int32 row pointers and column indices through a single
        constructor call.
    """
    source = ms.CSRTensor(tensor([0, 1, 2], ms.int32), tensor([1, 0], ms.int32), tensor([2, 3]), (2, 2))
    calls = []

    def make_csr(indptr, indices, values, shape):
        equal(indptr, [0, 1, 2], ms.int32)
        equal(indices, [1, 0], ms.int32)
        assert shape == source.shape
        calls.append(True)
        return ms.CSRTensor(indptr, indices, values, shape)

    monkeypatch.setattr(module(kind).F, 'make_csr_tensor', make_csr)
    result = getattr(module(kind), '_' + kind + '_csr_tensor')(source)
    assert isinstance(result, ms.CSRTensor) and result.shape == source.shape and calls == [True]
    equal(result.values, [1, 1] if kind == 'ones_like' else [0, 0], ms.float32)


def test_dynamic_zero_sequence_real_native_kernel(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Dynamic zero sequence real native kernel.
    Expectation: The real dynamic zero kernel returns the exact list [0, 0] after one predicate call with the
        original sequence.
    """
    sequence, calls = [3, 4], []

    def unknown_shape(value):
        assert value is sequence
        calls.append(value)
        return True

    monkeypatch.setattr(module('zeros_like').F, 'is_sequence_shape_unknown', unknown_shape)
    result = module('zeros_like').zeros_like(sequence)
    # Exact Python result types are part of this conversion contract.
    # pylint: disable-next=unidiomatic-typecheck
    assert result == [0, 0] and type(result) is list and len(calls) == 1


@pytest.mark.parametrize('kind', ['slice', 'tuple'])
def test_legacy_getitem_dispatch_native_metagraph_boundary(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Legacy getitem dispatch native metagraph boundary.
    Expectation: Legacy dispatch forwards the original source and index in the expected metadata-call order and
        returns the supplied sentinel.
    """
    monkeypatch.delenv('MS_DEV_TENSOR_INDEX_BOOST', raising=False)
    source, sentinel = tensor(np.arange(12).reshape(3, 4)), tensor([123])
    index = slice(0, 2) if kind == 'slice' else (slice(None), 1)
    calls = []

    def bool_metadata(index_arg):
        assert index_arg is index
        calls.append('bool')
        return index, (), ()

    def getitem_metadata(data_arg, index_arg):
        assert data_arg is source and index_arg is index
        calls.append('getitem')
        return sentinel

    monkeypatch.setattr(cu, '_handle_bool_tensor', bool_metadata)
    monkeypatch.setattr(cu, '_tensor_index_getitem', getitem_metadata)
    assert getattr(module('getitem'), '_tensor_getitem_by_' + kind)(source, index) is sentinel
    equal(sentinel, [123])
    assert calls == (['getitem'] if kind == 'slice' else ['bool', 'getitem'])


def test_empty_list_index_compiler_dynamic_dimension_error(monkeypatch):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Empty list index compiler dynamic dimension error.
    Expectation: An empty index under dynamic dimensions raises the documented IndexError after checking both
        dimension values.
    """
    calls = []

    def constant(value):
        assert value in (2, 3)
        calls.append(value)
        return value == 2

    monkeypatch.setattr(cu.F, 'isconstant', constant)
    with pytest.raises(IndexError, match="list can't be empty"):
        cu.tensor_index_by_list(tensor(np.zeros((3, 4))), [])
    assert calls == [2, 3]


@pytest.mark.parametrize('kind', ['coo', 'csr'])
@pytest.mark.parametrize('operation', ['add', 'sub'])
def test_sparse_binary_helpers_real_cpu_values(kind, operation):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Sparse binary helpers real CPU values.
    Expectation: Real COO or CSR addition and subtraction retain sparse type and indices while producing [6, 8]
        or [-2, -2].
    """
    if kind == 'coo':
        indices = tensor([[0, 1], [1, 0]], ms.int64)
        left = ms.COOTensor(indices, tensor([2, 3]), (2, 2))
        right = ms.COOTensor(indices, tensor([4, 5]), (2, 2))
    else:
        indptr, indices = tensor([0, 1, 2], ms.int32), tensor([1, 0], ms.int32)
        left = ms.CSRTensor(indptr, indices, tensor([2, 3]), (2, 2))
        right = ms.CSRTensor(indptr, indices, tensor([4, 5]), (2, 2))
    result = getattr(module(operation), '_' + operation + '_' + kind + 'tensor')(left, right)
    assert type(result) is type(left) and result.shape == (2, 2)
    equal(result.values, [6, 8] if operation == 'add' else [-2, -2])
    equal(result.indices, indices.asnumpy())
    if kind == 'csr':
        equal(result.indptr, [0, 1, 2])


@pytest.mark.parametrize(
    'mod,name,primitive,scalar',
    [
        ('add', '_tensor_add_scalar_augassign', 'InplaceAddsExt', True),
        ('add', '_tensor_add_tensor_augassign', 'InplaceAddExt', False),
        ('mul', '_mul_tensor_augassign', 'InplaceMul', False),
        ('mul', '_tensor_mul_scalar_augassign', 'InplaceMuls', True),
        ('sub', '_sub_tensor_augassign', 'InplaceSubExt', False),
        ('sub', '_tensor_sub_scalar_augassign', 'InplaceSubScalar', True),
        ('div', '_div_tensor_augassign', 'InplaceDiv', False),
        ('div', '_tensor_div_scalar_augassign', 'InplaceDivs', True),
        ('floordiv', '_floordiv_tensor_augassign', 'InplaceFloorDivide', False),
        ('floordiv', '_tensor_floordiv_scalar_augassign', 'InplaceFloorDivides', True),
        ('mod', '_mul_tensor_augassign', 'InplaceRemainderTensorTensor', False),
        ('mod', '_tensor_mod_scalar_augassign', 'InplaceRemainderTensorScalar', True),
    ],
)
def test_augassign_native_primitive_constructor_and_argument_boundary(monkeypatch, mod, name, primitive, scalar):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Validate dispatch only; these native inplace kernels are absent on CPU.
    Expectation: Augmented assignment constructs and invokes the expected primitive with original operands and
        retains its supplied result.
    """
    left, right, sentinel = tensor([4, 6]), (2 if scalar else tensor([2, 3])), tensor([123, 456])
    calls = []

    class NativeBoundary:
        def __init__(self):
            calls.append('construct')

        def __call__(self, x, y):
            assert x is left and y is right
            calls.append('invoke')
            return sentinel

    monkeypatch.setattr(module(mod), primitive, NativeBoundary)
    assert getattr(module(mod), name)(left, right) is sentinel
    equal(sentinel, [123, 456])
    equal(left, [4, 6])
    assert calls == ['construct', 'invoke']


@pytest.mark.parametrize(
    'name,primitive,scalar_side',
    [
        ('_mod_tensor', 'remainder_tensor_tensor_op', None),
        ('_tensor_mod_scalar', 'remainder_tensor_scalar_op', 'right'),
        ('_scalar_mod_tensor', 'remainder_scalar_tensor_op', 'left'),
    ],
)
def test_remainder_native_primitive_dispatch_boundary(monkeypatch, name, primitive, scalar_side):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Remainder native primitive dispatch boundary.
    Expectation: Remainder dispatch invokes the native boundary once with original operands and returns the
        supplied result.
    """
    left = 3 if scalar_side == 'left' else tensor([3, 5])
    right = 2 if scalar_side == 'right' else tensor([2, 3])
    sentinel, calls = tensor([123, 456]), []

    def native_remainder(x, y):
        assert x is left and y is right
        calls.append(True)
        return sentinel

    monkeypatch.setattr(module('mod'), primitive, native_remainder)
    assert getattr(module('mod'), name)(left, right) is sentinel
    equal(sentinel, [123, 456])
    assert calls == [True]


@pytest.mark.parametrize('kind', ['list', 'tuple', 'dict'])
def test_static_collection_add_native_metagraph_boundary(monkeypatch, kind):
    """Feature: Tensor indexing and multitype composite dispatch.
    Description: Static collection add native metagraph boundary.
    Expectation: Static collection addition forwards the original operands once and preserves the sentinel's
        exact list or tuple type.
    """
    if kind == 'list':
        left, right = [1, 2], [3, 4]
        attribute, function, sentinel = '_list_add', '_list_add_list', [7, 8]
    elif kind == 'tuple':
        left, right = (1, 2), (3, 4)
        attribute, function, sentinel = '_tuple_add', '_add_tuple', (7, 8)
    else:
        left, right = {'a': 1}, {'b': 2}
        attribute, function, sentinel = '_dict_update', '_dict_add_dict', {'c': 3}
    calls = []

    def native_add(x, y):
        assert x is left and y is right
        calls.append(True)
        return sentinel

    monkeypatch.setattr(module('add'), attribute, native_add)
    result = getattr(module('add'), function)(left, right)
    assert result is sentinel and type(result) is type(left) and calls == [True]
