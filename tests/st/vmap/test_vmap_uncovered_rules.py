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
"""Numerical coverage of vmap rules with non-leading and unbatched operands."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    """Use CPU kernels and leave the surrounding test process unchanged."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(device_target='CPU')
    yield
    ms.set_context(**previous)


@pytest.fixture(name='execution_mode', params=[ms.PYNATIVE_MODE, ms.GRAPH_MODE], ids=['pynative', 'graph'])
def execution_mode_fixture(request):
    """Exercise eager vmap and an explicitly compiled vmap graph."""
    previous = ms.get_context('mode')
    ms.set_context(mode=request.param)
    yield request.param
    ms.set_context(mode=previous)


def run_vmap(function, in_axes, mode, *inputs, out_axes=0):
    """Compile the complete mapped function when graph execution is requested."""

    def forward(*args):
        return function(*args)

    mapped = ops.vmap(forward, in_axes, out_axes)
    if mode == ms.GRAPH_MODE:

        @ms.jit
        def compiled(*args):
            return mapped(*args)

        return compiled(*inputs)
    return mapped(*inputs)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_vmap_ger_broadcasts_unbatched_operand(execution_mode):
    """Feature: Ger vmap rule.
    Description: Map a non-leading axis while broadcasting the second vector.
    Expectation: Each mapped output equals the NumPy outer product.
    """
    x = np.arange(6, dtype=np.float32).reshape((3, 2)) - 2
    y = np.array([2, -1, 3, 0], np.float32)
    result = run_vmap(P.Ger(), (1, None), execution_mode, ms.Tensor(x), ms.Tensor(y))
    np.testing.assert_allclose(result.asnumpy(), np.einsum('ib,j->bij', x, y), rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_nested_vmap_ger_increments_batch_rank(execution_mode):
    """Feature: Nested Ger vmap rule.
    Description: Add two batch dimensions to the same primitive.
    Expectation: Every vector pair produces its independent outer product.
    """
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4)) / 4
    y = np.array([2, -1, 0, 3, 1], np.float32)
    primitive = P.Ger()

    def forward(left, right):
        return primitive(left, right)

    inner = ops.vmap(forward, (0, None), 0)
    result = run_vmap(inner, (0, None), execution_mode, ms.Tensor(x), ms.Tensor(y))
    np.testing.assert_allclose(result.asnumpy(), np.einsum('abi,j->abij', x, y), rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('lower,upper', [(0, 0), (1, -1), (-1, 1)])
def test_vmap_matrix_band_part_preserves_matrix_axes(execution_mode, lower, upper):
    """Feature: MatrixBandPart vmap rule.
    Description: The batch axis lies between the two matrix axes.
    Expectation: Band limits apply to each matrix independently.
    """
    data = np.arange(24, dtype=np.float32).reshape((3, 2, 4))
    primitive = P.MatrixBandPart()

    def forward(x):
        return primitive(x, lower, upper)

    result = run_vmap(forward, 1, execution_mode, ms.Tensor(data))
    expected = np.moveaxis(data, 1, 0).copy()
    rows, columns = np.indices((3, 4))
    mask = (rows - columns <= lower) if lower >= 0 else np.ones((3, 4), bool)
    mask &= (columns - rows <= upper) if upper >= 0 else np.ones((3, 4), bool)
    expected *= mask
    np.testing.assert_array_equal(result.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['soft', 'hard'])
@pytest.mark.parametrize('sensitivity_axis', [0, 1])
def test_vmap_shrink_forward_and_gradient(execution_mode, kind, sensitivity_axis):
    """Feature: Shrink and shrink-gradient vmap rules.
    Description: Map activations and their gradients over a trailing batch axis.
    Expectation: Values and gradients agree with the piecewise scalar formula.
    """
    data = np.array([[-2, -0.25], [0.25, 2], [1.25, -1.25], [-0.75, 0.75]], np.float32)
    primitive = P.SoftShrink(0.75) if kind == 'soft' else P.HShrink(0.75)

    def forward(x):
        return primitive(x)

    result = run_vmap(forward, 1, execution_mode, ms.Tensor(data))
    mask = np.abs(data) > 0.75
    expected = np.where(mask, data - np.sign(data) * 0.75, 0) if kind == 'soft' else np.where(mask, data, 0)
    np.testing.assert_allclose(result.asnumpy(), expected.T, rtol=1e-6, atol=1e-6)
    sensitivity = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32)
    sensitivity_input = np.moveaxis(sensitivity, 0, sensitivity_axis)
    gradient_fn = ops.GradOperation(sens_param=True)(forward)
    gradient = run_vmap(
        gradient_fn, (1, sensitivity_axis), execution_mode, ms.Tensor(data), ms.Tensor(sensitivity_input)
    )
    np.testing.assert_array_equal(gradient.asnumpy(), mask.T.astype(np.float32) * sensitivity)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum', 'batchmean'])
def test_vmap_kl_div_loss_reduces_each_sample(execution_mode, reduction):
    """Feature: KLDivLoss vmap reduction rules.
    Description: Broadcast an unbatched target across a non-leading batch axis.
    Expectation: Reduction excludes the mapped dimension, including batchmean.
    """
    data = np.linspace(-2, -0.25, 24, dtype=np.float32).reshape((3, 2, 4))
    target = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape((3, 4))
    result = run_vmap(P.KLDivLoss(reduction), (1, None), execution_mode, ms.Tensor(data), ms.Tensor(target))
    expected = target * (np.log(target) - np.moveaxis(data, 1, 0))
    if reduction == 'mean':
        expected = expected.mean(axis=(1, 2))
    elif reduction == 'sum':
        expected = expected.sum(axis=(1, 2))
    elif reduction == 'batchmean':
        expected = expected.sum(axis=(1, 2)) / 3
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-5, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_vmap_smooth_l1_loss_reduces_each_sample(execution_mode, reduction):
    """Feature: SmoothL1Loss vmap reduction rules.
    Description: Broadcast the target and reduce only within each sample.
    Expectation: Outputs match an independent NumPy Huber-loss formula.
    """
    data = np.linspace(-2, 2, 24, dtype=np.float32).reshape((3, 2, 4))
    target = np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape((3, 4))
    result = run_vmap(
        P.SmoothL1Loss(beta=0.75, reduction=reduction), (1, None), execution_mode, ms.Tensor(data), ms.Tensor(target)
    )
    difference = np.moveaxis(data, 1, 0) - target
    absolute = np.abs(difference)
    expected = np.where(absolute < 0.75, difference**2 / 1.5, absolute - 0.375)
    if reduction == 'mean':
        expected = expected.mean(axis=(1, 2))
    elif reduction == 'sum':
        expected = expected.sum(axis=(1, 2))
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-5, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['maximum', 'minimum'])
@pytest.mark.parametrize('batch_axis', [0, 1])
def test_vmap_extremum_gradient_reduces_broadcast_axes(execution_mode, kind, batch_axis):
    """Feature: Broadcast binary-gradient vmap rules.
    Description: Differentiate a vector/scalar maximum or minimum per sample.
    Expectation: Scalar gradients sum only the elements in their own sample.
    """
    samples = np.array([[-2, 1, 3], [2, -1, 4]], np.float32)
    data = np.moveaxis(samples, 0, batch_axis)
    function = ops.maximum if kind == 'maximum' else ops.minimum
    gradients = run_vmap(
        ms.grad(function, (0, 1)),
        (batch_axis, None),
        execution_mode,
        ms.Tensor(data),
        ms.Tensor(0.5, ms.float32),
        out_axes=(0, 0),
    )
    expected_x = (samples > 0.5) if kind == 'maximum' else (samples < 0.5)
    expected_y = (~expected_x).sum(axis=1).astype(np.float32)
    np.testing.assert_array_equal(gradients[0].asnumpy(), expected_x.astype(np.float32))
    np.testing.assert_array_equal(gradients[1].asnumpy(), expected_y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('keep_dims', [False, True])
def test_vmap_reduce_std_adjusts_nonleading_batch_axis(execution_mode, keep_dims):
    """Feature: ReduceStd vmap rule.
    Description: Reduce a negative sample axis while preserving batch position.
    Expectation: Both standard deviation and mean match NumPy.
    """
    data = np.arange(24, dtype=np.float32).reshape((3, 2, 4)) / 4
    result = run_vmap(
        P.ReduceStd(axis=-1, unbiased=False, keep_dims=keep_dims), 1, execution_mode, ms.Tensor(data), out_axes=(0, 0)
    )
    samples = np.moveaxis(data, 1, 0)
    np.testing.assert_allclose(result[0].asnumpy(), samples.std(axis=-1, keepdims=keep_dims), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result[1].asnumpy(), samples.mean(axis=-1, keepdims=keep_dims), rtol=1e-5, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('batch_axis', [0, 2])
def test_vmap_log_matrix_determinant_preserves_tuple_output(execution_mode, batch_axis):
    """Feature: LogMatrixDeterminant vmap rule.
    Description: Place batch dimensions before or inside the matrix dimensions.
    Expectation: Both sign and log determinant retain the correct sample order.
    """
    samples = np.array([[[2, 0], [0, 3]], [[-2, 0], [0, 4]]], np.float32)
    data = np.moveaxis(samples, 0, batch_axis)
    result = run_vmap(P.LogMatrixDeterminant(), batch_axis, execution_mode, ms.Tensor(data), out_axes=(0, 0))
    np.testing.assert_array_equal(result[0].asnumpy(), [1, -1])
    np.testing.assert_allclose(result[1].asnumpy(), np.log([6, 8]), rtol=1e-6, atol=1e-6)
