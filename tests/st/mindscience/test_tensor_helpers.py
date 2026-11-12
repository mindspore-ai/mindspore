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
"""CPU coverage for the Tensor helpers used by retained MindScience models."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops
from tests.mark_utils import arg_mark
from tests.st.mindscience.mindsponge.mindsponge.common import geometry, utils
from tests.st.mindscience.mindsponge.mindsponge.common.tensor_utils import concat_features
from tests.st.mindscience.mindflow.src.pde_node import MINDSPORE_SYMPY_TRANSLATIONS


@pytest.fixture(autouse=True)
def restore_context():
    """Do not change the execution mode of subsequent model tests."""
    original = ms.get_context('mode')
    yield
    ms.set_context(mode=original)


class GeometryNet(nn.Cell):
    """Cover quaternion normalization, axis reduction and index split paths."""
    def construct(self, quat, translation, update):
        affine = geometry.quat_affine(quat, translation)
        updated = geometry.pre_compose(*affine, update)
        return geometry.quaternion_to_tensor(updated[0], updated[2])


class ResidueLookupNet(nn.Cell):
    def construct(self, table, indices):
        return utils._take_residue_constants(table, indices)


class DistogramNet(nn.Cell):
    def construct(self, positions):
        return utils.dgram_from_positions(positions, 4, 1.0, 4.0, ms.float32)


class FeatureConcatNet(nn.Cell):
    def construct(self, left, right):
        return concat_features((left, right), axis=-1)


class SymbolicMathNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sqrt = MINDSPORE_SYMPY_TRANSLATIONS['sqrt']
        self.sin = MINDSPORE_SYMPY_TRANSLATIONS['sin']

    def construct(self, x):
        return self.sqrt(x) + self.sin(x)


class SymbolicPowerDivisionNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.power = MINDSPORE_SYMPY_TRANSLATIONS['Pow']
        self.div = MINDSPORE_SYMPY_TRANSLATIONS['div']

    def construct(self, x):
        return self.div(self.power(x, 3), 2)


class RelativePositionNet(nn.Cell):
    def __init__(self, max_relative_feature):
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.one_hot = nn.OneHot(depth=2 * max_relative_feature + 1, axis=-1)

    def construct(self, residue_index):
        offset = ops.expand_dims(residue_index, 1) - ops.expand_dims(residue_index, 0)
        return self.one_hot(ops.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_residue_table_clipped_indices(mode):
    """
    Feature: Retained residue table lookup.
    Description: Preserve clipped rather than wrapping out-of-range indices.
    Expectation: Values and dtype match NumPy take with mode=clip.
    """
    ms.set_context(mode=mode, device_target='CPU')
    table = np.arange(12, dtype=np.float32).reshape(4, 3)
    indices = np.array([[-2, 0], [3, 7]], dtype=np.int32)
    result = ResidueLookupNet()(Tensor(table), Tensor(indices))
    np.testing.assert_array_equal(result.asnumpy(), np.take(table, indices, axis=0, mode='clip'))
    assert result.dtype == ms.float32


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_geometry_quaternion_update(mode):
    """
    Feature: Retained MindSponge geometry.
    Description: Normalize a quaternion and split its six-component update.
    Expectation: Identity quaternion and translation are preserved by zero update.
    """
    ms.set_context(mode=mode, device_target='CPU')
    quat = Tensor(np.array([[2., 0., 0., 0.], [3., 0., 0., 0.]], np.float32))
    translation = tuple(Tensor(np.array([v, v + 1], np.float32)) for v in [1., 3., 5.])
    result = GeometryNet()(quat, translation, Tensor(np.zeros((2, 6), np.float32)))
    expected = np.array([[1., 0., 0., 0., 1., 3., 5.], [1., 0., 0., 0., 2., 4., 6.]], np.float32)
    np.testing.assert_allclose(result.asnumpy(), expected, atol=1e-6)
    assert result.dtype == ms.float32


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_distogram_tensor_reductions(mode):
    """
    Feature: Retained distogram feature construction.
    Description: Exercise constant creation, concatenation and kept dimensions.
    Expectation: Pairwise distance bins agree with a host reference.
    """
    ms.set_context(mode=mode, device_target='CPU')
    positions = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]], np.float32)
    lower = np.linspace(1., 4., 4, dtype=np.float32) ** 2
    upper = np.concatenate((lower[1:], np.array([1e8], np.float32)))
    dist = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1, keepdims=True)
    expected = ((dist > lower) & (dist < upper)).astype(np.float32)
    np.testing.assert_array_equal(DistogramNet()(Tensor(positions)).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('left_dtype,right_dtype,out_dtype', [
    (ms.float16, ms.float16, ms.float16),
    (ms.float16, ms.float32, ms.float32),
    (ms.float32, ms.float16, ms.float32),
    (ms.float32, ms.float64, ms.float64),
])
def test_mixed_precision_feature_concat(mode, left_dtype, right_dtype, out_dtype):
    """
    Feature: Retained folding model mixed precision features.
    Description: Cast feature inputs before invoking the shared Concat kernel.
    Expectation: Promoted output dtype and values are preserved in both modes.
    """
    ms.set_context(mode=mode, device_target='CPU')
    left, right = Tensor([[1., 2.]], left_dtype), Tensor([[3.]], right_dtype)
    net = FeatureConcatNet()
    result = net(left, right)
    assert result.dtype == out_dtype
    np.testing.assert_array_equal(result.asnumpy(), [[1., 2., 3.]])
    grad_left, grad_right = ops.grad(net, (0, 1))(left, right)
    assert grad_left.dtype == left_dtype and grad_right.dtype == right_dtype
    np.testing.assert_array_equal(grad_left.asnumpy(), [[1., 1.]])
    np.testing.assert_array_equal(grad_right.asnumpy(), [[1.]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_symbolic_math_tensor_gradients(mode):
    """
    Feature: MindFlow symbolic expression mapping.
    Description: Keep expressions and derivatives in Tensor computation.
    Expectation: Both numerical expression values and derivatives are correct.
    """
    ms.set_context(mode=mode, device_target='CPU')
    values = np.array([[0.5], [1.], [2.]], np.float32)
    net = SymbolicMathNet()
    np.testing.assert_allclose(net(Tensor(values)).asnumpy(), np.sqrt(values) + np.sin(values), rtol=1e-5)
    expected_grad = 0.5 / np.sqrt(values) + np.cos(values)
    np.testing.assert_allclose(ops.grad(net)(Tensor(values)).asnumpy(), expected_grad, rtol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_symbolic_power_division_gradients(mode):
    """
    Feature: MindFlow power and true division mappings.
    Description: Evaluate a fractional result and its gradient through both mappings.
    Expectation: Tensor values and gradients match the analytic expression x**3 / 2.
    """
    ms.set_context(mode=mode, device_target='CPU')
    values = np.array([[-1.5], [0.5], [2.]], np.float32)
    net = SymbolicPowerDivisionNet()
    result = net(Tensor(values))
    assert result.dtype == ms.float32
    np.testing.assert_allclose(result.asnumpy(), np.power(values, 3) / 2, rtol=1e-5)
    np.testing.assert_allclose(ops.grad(net)(Tensor(values)).asnumpy(), 1.5 * values**2, rtol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_relative_position_clamp(mode, index_dtype):
    """
    Feature: Retained MindSponge relative position features.
    Description: Clamp integer residue offsets before one-hot encoding.
    Expectation: Boundary and out-of-range offsets match a NumPy reference.
    """
    ms.set_context(mode=mode, device_target='CPU')
    values = np.array([0, 1, 2, 8], dtype=index_dtype)
    limit = 2
    indices = np.clip(values[:, None] - values[None, :] + limit, 0, 2 * limit)
    expected = np.eye(2 * limit + 1, dtype=np.float32)[indices]
    result = RelativePositionNet(limit)(Tensor(values))
    assert result.dtype == ms.float32
    np.testing.assert_array_equal(result.asnumpy(), expected)
