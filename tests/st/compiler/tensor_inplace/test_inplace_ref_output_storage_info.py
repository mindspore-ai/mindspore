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
# ==============================================================================
"""Regression tests for storage metadata on inplace ref outputs."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops.auto_generate import ExpandDimsView, SelectExtView
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def pynative_context():
    """Restore the execution mode after checking PyNative and JIT together."""
    previous_mode = ms.get_context("mode")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    yield
    ms.set_context(mode=previous_mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=4.0)
@pytest.mark.parametrize('dynamic_shape', [False, True])
@pytest.mark.parametrize('base_value', [1.0, 11.0])
@pytest.mark.parametrize('step', [1, 2])
def test_indexed_inplace_ref_output_storage_info(monkeypatch, dynamic_shape, base_value, step):
    """
    Feature: Storage metadata propagation for inplace ref outputs.
    Description: Write an inplace result back through an offset, optionally strided view.
    Expectation: The selected elements are incremented and gradients agree with the two outputs.
    """
    monkeypatch.setenv('MS_DEV_TENSOR_INDEX_BOOST', '1')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.step = step

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            view_obj1 = x[1, 0:2, ::self.step]
            view_obj1[...] += Tensor(5)
            view_obj2 = y[Tensor(1)]
            view_obj2[None] -= 3
            return view_obj1, x

    net = Net()
    if dynamic_shape:
        net.set_inputs(Tensor(shape=[2, 4, None], dtype=ms.float32),
                       Tensor(shape=[2, 4, None], dtype=ms.float32))
    for use_jit in (False, True):
        if use_jit:
            net.construct = ms.jit(net.construct, backend="ms_backend")
        for width in (8, 12 if dynamic_shape else 8):
            x_np = np.ones((2, 4, width), dtype=np.float32)
            x_np[0] = base_value
            inputs = (Tensor(x_np), Tensor(np.full_like(x_np, 2)))
            expected_x = x_np.copy()
            expected_x[1, :2, ::step] = 6
            expected_dx = np.ones_like(x_np)
            expected_dx[1, :2, ::step] = 2

            view, base = net(*inputs)
            np.testing.assert_allclose(view.asnumpy(), np.full((2, width // step), 6), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(base.asnumpy(), expected_x, rtol=1e-5, atol=1e-5)
            # Backward through dynamic Tensor indexing is not supported.
            if not dynamic_shape:
                dx, dy = ms.grad(net, grad_position=(0, 1))(*inputs)
                np.testing.assert_allclose(dx.asnumpy(), expected_dx, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(dy.asnumpy(), np.zeros_like(x_np), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=4.0)
@pytest.mark.parametrize('dynamic_shape', [False, True])
@pytest.mark.parametrize('base_value', [2.0, 11.0])
@pytest.mark.parametrize('x_value, updated_y, expected_dy', [(1.0, 5.0, 2.0), (0.125, 4.0, 1.125)])
def test_conditional_inplace_ref_output_gradient(dynamic_shape, base_value, x_value, updated_y, expected_dy):
    """
    Feature: Storage metadata propagation for inplace ref outputs in backward graphs.
    Description: Differentiate a product using a view updated by either conditional branch.
    Expectation: dx uses the updated selected row, while dy includes both returned outputs.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.selectview = SelectExtView()
            self.expanddimsview = ExpandDimsView()

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            x = self.selectview(x, 0, 1)
            y = self.selectview(y, 0, 1)
            y.add_(2) if ops.ReduceSum()(x) < 3 else y.add_(3)  # pylint: disable=expression-not-assigned
            view_obj1 = None
            for _ in range(2):
                view_obj1 = self.expanddimsview(x, 0)
            view_obj1.mul_(y)
            return view_obj1, y

    net = Net()
    if dynamic_shape:
        net.set_inputs(Tensor(shape=[4, None], dtype=ms.float32), Tensor(shape=[4, None], dtype=ms.float32))
    for use_jit in (False, True):
        if use_jit:
            net.construct = ms.jit(net.construct, backend="ms_backend")
        for width in (8, 12 if dynamic_shape else 8):
            x_np = np.full((4, width), x_value, dtype=np.float32)
            y_np = np.full_like(x_np, 2)
            y_np[0] = base_value
            inputs = (Tensor(x_np), Tensor(y_np))
            expected_dx = np.zeros_like(x_np)
            expected_dx[1] = updated_y
            expected_y_grad = np.zeros_like(y_np)
            expected_y_grad[1] = expected_dy

            view, y = net(*inputs)
            np.testing.assert_allclose(view.asnumpy(), np.full((1, width), x_value * updated_y))
            np.testing.assert_allclose(y.asnumpy(), np.full((width,), updated_y))
            dx, dy = ms.grad(net, grad_position=(0, 1))(*inputs)
            np.testing.assert_allclose(dx.asnumpy(), expected_dx)
            np.testing.assert_allclose(dy.asnumpy(), expected_y_grad)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=4.0)
@pytest.mark.parametrize('use_view', [False, True])
def test_control_flow_ref_output_with_parameter(use_view):
    """
    Feature: Ref output storage information for control-flow runners.
    Description: Update a Parameter from a tensor or view in either branch with a Monad control dependency.
    Expectation: Both branches execute without invalid ref indices and preserve updates.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = ms.Parameter(Tensor([1.0, 2.0], ms.float32), name="weight")
            self.use_view = use_view
            self.select_view = SelectExtView()

        def construct(self, cond, value):
            if self.use_view:
                # Even a zero-offset view carries storage info and triggers the control-flow runner's Resize.
                value = self.select_view(value, 0, 0)
            if cond:
                self.weight += value
            else:
                self.weight -= value
            return self.weight * 2

    net = Net()
    net.construct = ms.jit(net.construct, backend="ms_backend")
    add_value = [[3.0, 4.0], [11.0, 11.0]] if use_view else [3.0, 4.0]
    sub_value = [[1.0, 2.0], [11.0, 11.0]] if use_view else [1.0, 2.0]
    np.testing.assert_allclose(net(Tensor(True), Tensor(add_value, ms.float32)).asnumpy(), [8.0, 12.0])
    np.testing.assert_allclose(net(Tensor(False), Tensor(sub_value, ms.float32)).asnumpy(), [6.0, 8.0])
    np.testing.assert_allclose(net.weight.asnumpy(), [3.0, 4.0])
