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
"""Eager rule-level checks complement graph-compiled vmap system tests."""
# These unit tests intentionally cover private vmap shape and axis helpers.
# pylint: disable=protected-access
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops._vmap import vmap_base as base
from mindspore.ops._vmap import vmap_array_ops as array_rules
from mindspore.ops._vmap import vmap_nn_ops as nn_rules
from mindspore.ops.primitive import Primitive
from mindspore.ops import operations as P


@pytest.fixture(scope='module', autouse=True)
def cpu_pynative_context():
    """Keep real primitive execution on CPU and restore the caller's context."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)
    yield
    ms.set_context(**previous)


@pytest.mark.parametrize('shape,target,expected_shape,expected_axes', [
    ((2, 3, 4), (2, 4), (2, 3, 4), ()),
    ((), (2, 3, 4), (1, 1, 1), (0, 1, 2)),
    ((2,), (2, 3, 4), (2, 1, 1), (1, 2)),
    ((2, 4), (2, 3, 4), (2, 1, 4), (1,)),
    ((2, 3), (2, 3, 4), (2, 3, 1), (2,)),
    ((2, 1), (2, 3, 4), (2, 1, 1), (1,)),
])
def test_front_batch_broadcast_shape_and_inserted_axes(shape, target, expected_shape, expected_axes):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Broadcast helpers must keep batch dimensions separate from sample axes.
    Expectation: Broadcast helpers must keep batch dimensions separate from sample axes.
    """
    assert base._get_broadcast_shape_with_front_axis(shape, target) == expected_shape
    assert base._get_broadcasting_with_front_axis_additional_axis(shape, target) == expected_axes


@pytest.mark.parametrize('destination,expected', [(0, (4, 2, 3)), (1, (2, 4, 3)),
                                                (-1, (2, 3, 4)), (-3, (4, 2, 3))])
def test_batch_broadcast_shape_inserts_axis_at_normalized_position(destination, expected):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Negative insertion axes must refer to the rank after adding the batch.
    Expectation: Negative insertion axes must refer to the rank after adding the batch.
    """
    assert base._get_broadcast_shape((2, 3), destination, 4) == expected


@pytest.mark.parametrize('destination', [-4, 3])
def test_batch_broadcast_shape_rejects_out_of_range_axis(destination):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Invalid destinations must fail before reshaping the operand.
    Expectation: Invalid destinations must fail before reshaping the operand.
    """
    with pytest.raises(ValueError, match='out of bounds'):
        base._get_broadcast_shape((2, 3), destination, 4)


@pytest.mark.parametrize('destination', [0, -1])
def test_broadcast_scalar_to_batch(destination):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: An unbatched scalar must be repeated for each mapped sample.
    Expectation: An unbatched scalar must be repeated for each mapped sample.
    """
    result = base._broadcast_by_axis(3.5, destination, 2)
    np.testing.assert_array_equal(result.asnumpy(), [3.5, 3.5])


def test_broadcast_scalar_rejects_nonexistent_axis():
    """
    Feature: Vmap axis and broadcast boundaries
    Description: A scalar has only one possible insertion axis.
    Expectation: A scalar has only one possible insertion axis.
    """
    with pytest.raises(ValueError, match='out of bounds'):
        base._broadcast_by_axis(3.5, 1, 2)


def test_broadcast_tensor_and_move_existing_batch_axis():
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Adding or moving a batch axis must preserve every tensor element.
    Expectation: Adding or moving a batch axis must preserve every tensor element.
    """
    tensor = ms.Tensor([2, 4, 6], ms.float32)
    result = base._bdim_at_front(tensor, None, 2)
    np.testing.assert_array_equal(result.asnumpy(), [[2, 4, 6], [2, 4, 6]])
    moved = base._bdim_at_back(result, 0, 2)
    np.testing.assert_array_equal(moved.asnumpy(), [[2, 2], [4, 4], [6, 6]])
    with pytest.raises(ValueError, match='can not be None'):
        base._bdim_at_any(result, 0, None, 2)


@pytest.mark.parametrize('axis,batch_axis,rank,expected', [
    (-1, 1, 4, 3), (0, 1, 4, 0), (1, -2, 4, 1),
    ((0, -1), 1, 4, (0, 3)), ((), 1, 4, (0, 2, 3)),
])
def test_reduce_batch_axis_excludes_mapped_dimension(axis, batch_axis, rank, expected):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Logical reduction axes must skip the physical batch dimension.
    Expectation: Logical reduction axes must skip the physical batch dimension.
    """
    assert base._get_reduce_batch_axis(axis, batch_axis, rank) == expected


@pytest.mark.parametrize('batch_axis,reduce_axes,keep_dims,expected', [
    (2, (0, 1), False, 0), (2, (0, 3), False, 1), (2, 0, True, 2),
])
def test_reduce_output_batch_dimension_tracks_removed_axes(batch_axis, reduce_axes, keep_dims, expected):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Output metadata must account only for removed axes before the batch.
    Expectation: Output metadata must account only for removed axes before the batch.
    """
    assert base._get_reduce_out_dim(batch_axis, reduce_axes, keep_dims) == expected


@pytest.mark.parametrize('value,expected', [(3, (3, None)), ((3, 4), ((3, None), (4, None)))])
def test_bind_unmapped_outputs_preserves_tuple_structure(value, expected):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Each output operand must retain independent unmapped-axis metadata.
    Expectation: Each output operand must retain independent unmapped-axis metadata.
    """
    assert base.vmap_bind_all_none(value) == expected


@pytest.mark.parametrize('x_axis,dy_axis', [(0, 0), (1, 0), (0, 1), (None, 0), (1, None)])
def test_unary_gradient_rule_aligns_and_broadcasts_operands(x_axis, dy_axis):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: TanhGrad must keep each sample's activation and sensitivity together.
    Expectation: TanhGrad must keep each sample's activation and sensitivity together.
    """
    samples = np.array([[-0.5, 0, 0.5], [0.25, -0.25, 0.75]], np.float32)
    sensitivities = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    x = samples[0] if x_axis is None else np.moveaxis(samples, 0, x_axis)
    dy = sensitivities[0] if dy_axis is None else np.moveaxis(sensitivities, 0, dy_axis)
    rule = base.get_unary_grad_vmap_rule('TanhGrad', 2)
    result, result_axis = rule((ms.Tensor(x), x_axis), (ms.Tensor(dy), dy_axis))
    expected_x = np.broadcast_to(samples[0], (2, 3)) if x_axis is None else samples
    expected_dy = np.broadcast_to(sensitivities[0], (2, 3)) if dy_axis is None else sensitivities
    expected = (1 - expected_x ** 2) * expected_dy
    assert result_axis == (dy_axis if x_axis is None else x_axis)
    np.testing.assert_allclose(np.moveaxis(result.asnumpy(), result_axis, 0), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('kind', ['soft', 'hard'])
@pytest.mark.parametrize('dy_axis', [0, 1])
def test_shrink_gradient_rule_keeps_sensitivity_batch_alignment(kind, dy_axis):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Direct rule execution must preserve nonuniform sensitivity weights.
    Expectation: Direct rule execution must preserve nonuniform sensitivity weights.
    """
    data = np.array([[-2, -0.75], [0.75, 2], [0.25, -0.25]], np.float32)
    sensitivity = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    dy = np.moveaxis(sensitivity, 0, dy_axis)
    name = 'SoftShrinkGrad' if kind == 'soft' else 'HShrinkGrad'
    factory = nn_rules.get_softshrink_grad_vmap_rule if kind == 'soft' else nn_rules.get_hshrink_grad_vmap_rule
    rule = factory(Primitive(name), 2)
    result, result_axis = rule((ms.Tensor(dy), dy_axis), (ms.Tensor(data), 1), (0.75, None))
    assert result_axis == (1 if dy_axis == 1 else 0)
    expected = (np.abs(data.T) > 0.75).astype(np.float32) * sensitivity
    np.testing.assert_array_equal(np.moveaxis(result.asnumpy(), result_axis, 0), expected)


@pytest.mark.parametrize('name,factory', [('SoftShrinkGrad', nn_rules.get_softshrink_grad_vmap_rule),
                                        ('HShrinkGrad', nn_rules.get_hshrink_grad_vmap_rule)])
def test_shrink_gradient_rule_rejects_incompatible_sample_shapes(name, factory):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: A mismatched sensitivity cannot silently broadcast to a different sample.
    Expectation: A mismatched sensitivity cannot silently broadcast to a different sample.
    """
    rule = factory(Primitive(name), 2)
    with pytest.raises(RuntimeError, match='shape is supposed to be the same'):
        rule((ms.Tensor(np.ones((2, 4), np.float32)), 0),
             (ms.Tensor(np.ones((3, 2), np.float32)), 1), (0.75, None))


@pytest.mark.parametrize('name,factory', [('SoftShrink', nn_rules.get_softshrink_vmap_rule),
                                        ('HShrink', nn_rules.get_hshrink_vmap_rule)])
def test_shrink_rule_retains_batch_axis_and_threshold_boundary(name, factory):
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Values exactly at the shrink threshold must produce zero.
    Expectation: Values exactly at the shrink threshold must produce zero.
    """
    data = np.array([[-2, -0.75], [0.75, 2]], np.float32)
    result, axis = factory(Primitive(name), 2)((ms.Tensor(data), 1), (0.75, None))
    expected = [[-1.25, 0], [0, 1.25]] if name == 'SoftShrink' else [[-2, 0], [0, 2]]
    assert axis == 1
    np.testing.assert_array_equal(result.asnumpy(), expected)


def test_cast_rule_preserves_axis_and_rejects_batched_dtype():
    """
    Feature: Vmap axis and broadcast boundaries
    Description: Casting changes elements' dtype while preserving batch metadata.
    Expectation: Casting changes elements' dtype while preserving batch metadata.
    """
    tensor = ms.Tensor([[1.75, -2.75], [3, 4]], ms.float32)
    rule = array_rules.get_cast_vmap_rule(P.Cast(), 2)
    result, axis = rule((tensor, 1), (ms.int32, None))
    assert axis == 1
    assert result.dtype == ms.int32
    np.testing.assert_array_equal(result.asnumpy(), [[1, -2], [3, 4]])
    with pytest.raises(ValueError, match='must be None'):
        rule((tensor, 1), (ms.int32, 0))
