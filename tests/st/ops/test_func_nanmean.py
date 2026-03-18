# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ST tests for nanmean functional API."""
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.graph.api import _pynative_executor


class Net(nn.Cell):
    def construct(self, x):
        return ops.nanmean(x, axis=0, keepdims=True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean(mode):
    """
    Feature: ops.nanmean
    Description: Verify the result of nanmean
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[float("nan"), 128.1, -256.9], [float("nan"), float("nan"), 128]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[float("nan"), 128.1, -64.45]]
    assert np.allclose(output.asnumpy(), expect_output, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_int_error(mode):
    """
    Feature: ops.nanmean
    Description: Whether typeerror can be caught if the input is int
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([1, 2, 3], ms.int32)
    net = Net()
    with pytest.raises(TypeError):
        net(x)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_grad_all_nan(mode):
    """
    Feature: ops.nanmean backward
    Description: Verify grad behavior when all reduced elements are nan
    Expectation: gradients are nan
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(mode=mode, jit_level='O0')
    else:
        ms.set_context(mode=mode)
    x = ms.Tensor(np.full((5, 7, 4), np.nan).astype(np.float32))
    sens = ms.Tensor(np.ones((7, 4)).astype(np.float64))

    def forward_func(input_x):
        return ops.nanmean(input_x, axis=0, keepdims=False, dtype=ms.float64)

    grad_func = ops.grad(forward_func, grad_position=(0,), sens_param=True)
    dx = grad_func(x, sens)
    assert np.isnan(dx.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=0.02)
def test_ops_nanmean_grad_all_nan_graph_o2():
    """
    Feature: ops.nanmean backward
    Description: Verify grad behavior with all nan input in graph O2 mode
    Expectation: gradients are nan
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")
    x = ms.Tensor(np.full((5, 7, 4), np.nan).astype(np.float32))
    sens = ms.Tensor(np.ones((7, 4)).astype(np.float64))

    def forward_func(input_x):
        return ops.nanmean(input_x, axis=0, keepdims=False, dtype=ms.float64)

    grad_func = ops.grad(forward_func, grad_position=(0,), sens_param=True)
    dx = grad_func(x, sens)
    assert np.isnan(dx.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_grad_axis_none_scalar_dout(mode):
    """
    Feature: ops.nanmean backward
    Description: axis=None with scalar dout should recover all dims correctly
    Expectation: gradients match expected broadcasted value
    """
    ms.set_context(mode=mode)
    x_np = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
    x = ms.Tensor(x_np)
    sens = ms.Tensor(np.array(2.0, dtype=np.float32))

    def forward_func(input_x):
        return ops.nanmean(input_x, axis=None, keepdims=False)

    grad_func = ops.grad(forward_func, grad_position=(0,), sens_param=True)
    dx = grad_func(x, sens).asnumpy()
    expect = np.full_like(x_np, 2.0 / 24.0)
    np.testing.assert_allclose(dx, expect, rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_grad_axis_non_sorted(mode):
    """
    Feature: ops.nanmean backward
    Description: non-sorted multi-axis input should be canonicalized and broadcasted correctly
    Expectation: gradients match manual result for axis=(2, 0)
    """
    ms.set_context(mode=mode)
    x_np = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
    x = ms.Tensor(x_np)
    sens_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sens = ms.Tensor(sens_np)

    def forward_func(input_x):
        return ops.nanmean(input_x, axis=(2, 0), keepdims=False)

    grad_func = ops.grad(forward_func, grad_position=(0,), sens_param=True)
    dx = grad_func(x, sens).asnumpy()
    expect = np.zeros_like(x_np)
    for j in range(3):
        expect[:, j, :] = sens_np[j] / 8.0
    np.testing.assert_allclose(dx, expect, rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_grad_keepdims_true(mode):
    """
    Feature: ops.nanmean backward
    Description: keepdims=True should not need expand_dims path and keep gradient shape correct
    Expectation: gradients match manual keepdims broadcast
    """
    ms.set_context(mode=mode)
    x_np = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
    x = ms.Tensor(x_np)
    sens_np = np.array([[[1.0]], [[3.0]]], dtype=np.float32)
    sens = ms.Tensor(sens_np)

    def forward_func(input_x):
        return ops.nanmean(input_x, axis=(1, 2), keepdims=True)

    grad_func = ops.grad(forward_func, grad_position=(0,), sens_param=True)
    dx = grad_func(x, sens).asnumpy()
    expect = np.zeros_like(x_np)
    expect[0, :, :] = sens_np[0, 0, 0] / 12.0
    expect[1, :, :] = sens_np[1, 0, 0] / 12.0
    np.testing.assert_allclose(dx, expect, rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_grad_axis_non_sorted_with_nan(mode):
    """
    Feature: ops.nanmean backward
    Description: non-sorted multi-axis with uneven nan distribution
    Expectation: grad uses per-output valid count and nan positions keep zero grad
    """
    ms.set_context(mode=mode)
    x_np = np.array([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, np.nan, 7.0, 8.0],
         [np.nan, np.nan, 11.0, 12.0]],
        [[13.0, 14.0, 15.0, 16.0],
         [17.0, 18.0, np.nan, 20.0],
         [np.nan, 22.0, np.nan, np.nan]]
    ], dtype=np.float32)
    x = ms.Tensor(x_np)
    sens_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sens = ms.Tensor(sens_np)

    def forward_func(input_x):
        return ops.nanmean(input_x, axis=(2, 0), keepdims=False)

    grad_func = ops.grad(forward_func, grad_position=(0,), sens_param=True)
    dx = grad_func(x, sens).asnumpy()
    valid_count = (~np.isnan(x_np)).sum(axis=(0, 2))
    expect = np.zeros_like(x_np)
    for j in range(x_np.shape[1]):
        grad_value = sens_np[j] / valid_count[j]
        mask = ~np.isnan(x_np[:, j, :])
        expect[:, j, :][mask] = grad_value
    np.testing.assert_allclose(dx, expect, rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_forward_backward_dynamic_shape(mode):
    """
    Feature: ops.nanmean forward and backward
    Description: dynamic shape should work in pynative and graph O0
    Expectation: forward and gradients are correct for multiple runtime input shapes
    """

    class NanMeanForwardNet(nn.Cell):
        def construct(self, input_x):
            return ops.nanmean(input_x, axis=1, keepdims=False, dtype=ms.float32)

    class NanMeanGradNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.forward = NanMeanForwardNet()
            self.grad_fn = ops.grad(self.forward, grad_position=(0,), sens_param=True)

        def construct(self, input_x, sens):
            return self.grad_fn(input_x, sens)

    if mode == ms.GRAPH_MODE:
        ms.set_context(mode=mode, jit_level='O0')
    else:
        ms.set_context(mode=mode)

    def get_forward_expect(x_np):
        return np.nanmean(x_np, axis=1, dtype=np.float32)

    def get_grad_expect(x_np, sens_np):
        valid_count = (~np.isnan(x_np)).sum(axis=1, keepdims=True).astype(np.float32)
        expect = np.zeros_like(x_np, dtype=np.float32)
        grad_val = sens_np.reshape((-1, 1)) / valid_count
        grad_val = np.broadcast_to(grad_val, x_np.shape)
        mask = ~np.isnan(x_np)
        expect[mask] = grad_val[mask]
        return expect

    forward_net = NanMeanForwardNet()
    grad_net = NanMeanGradNet()
    if mode == ms.GRAPH_MODE:
        x_dyn = ms.Tensor(shape=[2, None], dtype=ms.float32)
        sens_dyn = ms.Tensor(shape=[2], dtype=ms.float32)
        forward_net.set_inputs(x_dyn)
        grad_net.set_inputs(x_dyn, sens_dyn)

    def run_case(x_np, sens_np):
        x = ms.Tensor(x_np)
        sens = ms.Tensor(sens_np)
        forward_out = forward_net(x).asnumpy()
        grad_out = grad_net(x, sens).asnumpy()
        np.testing.assert_allclose(forward_out, get_forward_expect(x_np), rtol=1e-6, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(grad_out, get_grad_expect(x_np, sens_np), rtol=1e-6, atol=1e-6)

    x1_np = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    sens1_np = np.array([2.0, 6.0], dtype=np.float32)
    run_case(x1_np, sens1_np)

    x2_np = np.array([[1.0, 2.0, np.nan, 4.0, np.nan], [np.nan, 3.0, 5.0, 7.0, 9.0]], dtype=np.float32)
    sens2_np = np.array([3.0, 8.0], dtype=np.float32)
    run_case(x2_np, sens2_np)
