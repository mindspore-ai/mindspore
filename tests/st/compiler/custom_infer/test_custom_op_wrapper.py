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
"""Test custom op infer"""

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter, ops, mint, jit
from mindspore.graph import register_op_wrapper, clear_op_infer, clear_op_wrapper
from tests.mark_utils import arg_mark


def teardown_module():
    clear_op_infer()
    clear_op_wrapper()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_custom_op_wrapper_gradient_1():
    """
    Feature: Custom op wrapper gradient computation.
    Description: Test gradient computation when Add is replaced with x + 2*y.
    Expectation: Gradients reflect the modified addition operation.
    """
    class WrapperNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            return ops.Add()(x, y)

    @register_op_wrapper("Add")
    def double_add_wrapper(x, y):
        def func(x, y):
            return x + 2 * y
        return func

    net = WrapperNet()
    x = Tensor([[1., 2.], [3., 4.]])
    y = Tensor([[0.5, 1.0], [1.5, 2.0]])

    def loss_fn(x, y):
        output = net(x, y)
        return ops.ReduceSum()(output)

    grad_fn = ms.grad(loss_fn, grad_position=(0, 1))
    grad_x, grad_y = grad_fn(x, y)
    expected_grad_x = np.ones_like(x.asnumpy())
    expected_grad_y = 2 * np.ones_like(y.asnumpy())

    np.allclose(grad_x.asnumpy(), expected_grad_x, rtol=1e-5)
    np.allclose(grad_y.asnumpy(), expected_grad_y, rtol=1e-5)

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_custom_op_wrapper_1():
    """
    Feature: Custom op wrapper.
    Description: Test register_op_wrapper decorator function.
    Expectation: Custom wrapper set to the operator replace the operator successfully.
    """
    class WrapperNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            x = ops.Add()(x, y)
            return x

    @register_op_wrapper("Add")
    def double_add_wrapper(x, b):
        def func(x, b):
            return x + b + b
        return func

    net = WrapperNet()
    input_x = Tensor([[1., 2., 3.], [4., 5., 6.]])
    input_y = Tensor([[1., 2., 3.], [4., 5., 6.]])
    res = net(input_x, input_y)
    expected = input_x + input_y + input_y
    assert np.allclose(res.asnumpy(), expected.asnumpy())
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_custom_op_wrapper_2():
    """
    Feature: Custom op wrapper.
    Description: Test register_op_wrapper decorator function.
    Expectation: Custom wrapper set to the operator in wrapper will not replace the operator.
    """
    class WrapperNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            x = ops.Add()(x, y)
            return x

    @register_op_wrapper("Add")
    def add_wrapper(x, b):
        def func(x, b):
            return x - b
        return func

    @register_op_wrapper("Sub")
    def sub_wrapper(x, b):
        def func(x, b):
            return x + b
        return func

    net = WrapperNet()
    input_x = Tensor([[1., 2., 3.], [4., 5., 6.]])
    input_y = Tensor([[1., 2., 3.], [4., 5., 6.]])
    res = net(input_x, input_y)
    expected = input_x - input_y
    assert np.allclose(res.asnumpy(), expected.asnumpy())
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_custom_op_wrapper_gradient_2():
    """
    Feature: Custom op wrapper gradient computation with multiple operations.
    Description: Test gradient computation when Add and Sub are both wrapped.
    Expectation: Gradients reflect both modified operations.
    """
    class WrapperNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            z = ops.Add()(x, y)
            w = ops.Sub()(z, x)
            return w

    @register_op_wrapper("Add")
    def triple_add_wrapper(a, b):
        def func(a, b):
            return a + 3 * b
        return func

    @register_op_wrapper("Sub")
    def add_sub_wrapper(a, b):
        def func(a, b):
            return a + b
        return func

    net = WrapperNet()
    x = Tensor([[1., 2.], [3., 4.]])
    y = Tensor([[0.5, 1.0], [1.5, 2.0]])

    def loss_fn(x, y):
        output = net(x, y)
        return ops.ReduceSum()(output)

    grad_fn = ms.grad(loss_fn, grad_position=(0, 1))
    grad_x, grad_y = grad_fn(x, y)
    expected_grad_x = 2 * np.ones_like(x.asnumpy())
    expected_grad_y = 3 * np.ones_like(y.asnumpy())
    np.allclose(grad_x.asnumpy(), expected_grad_x, rtol=1e-5)
    np.allclose(grad_y.asnumpy(), expected_grad_y, rtol=1e-5)

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_register_op_wrapper_control_flow_1():
    """
    Feature: Custom op wrapper in control flow.
    Description: Test register_op_wrapper in network with conditional branches.
    Expectation: Custom operations work correctly in different branches.
    """
    class ConditionalNetwork(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(4, 4)

        @jit(backend='ms_backend')
        def construct(self, x, condition):
            x = self.dense(x)
            if condition > 0:
                x = ops.Add()(x, x)
                x = ops.Mul()(x, Tensor(2.0))
            else:
                x = ops.Sub()(x, x)
                x = ops.Div()(x, Tensor(2.0))

            return x

    @register_op_wrapper("Add")
    def enhanced_add_wrapper(x, y):
        def enhanced_add(x, y):
            return x + y + ops.Sin()(x)
        return enhanced_add

    @register_op_wrapper("Mul")
    def scaled_mul_wrapper(x, y):
        def scaled_mul(x, y):
            scale = ops.ReduceMean()(ops.Abs()(x)) + 1.0
            return x * y * scale
        return scaled_mul

    @register_op_wrapper("Sub")
    def smooth_sub_wrapper(x, y):
        def smooth_sub(x, y):
            return ops.Tanh()(x - y)
        return smooth_sub

    @register_op_wrapper("Div")
    def safe_div_wrapper(x, y):
        def safe_div(x, y):
            eps = 1e-8
            return x / (y + eps)
        return safe_div

    net = ConditionalNetwork()
    x1 = Tensor(np.random.randn(2, 4).astype(np.float32))
    result1 = net(x1, Tensor(1.0))
    x2 = Tensor(np.random.randn(2, 4).astype(np.float32))
    result2 = net(x2, Tensor(-1.0))
    assert not np.allclose(result1.asnumpy(), result2.asnumpy())

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_register_op_wrapper_resnet_block():
    """
    Feature: Custom op wrapper in ResNet.
    Description: Test register_op_wrapper in a residual block.
    Expectation: Custom operations work correctly in residual architecture.
    """
    class ResidualBlock(ms.nn.Cell):
        def __init__(self, channels=16):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, pad_mode='same')
            self.conv2 = nn.Conv2d(channels, channels, 3, pad_mode='same')
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)

        @jit(backend='ms_backend')
        def construct(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = ops.ReLU()(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = ops.Add()(out, identity)
            out = ops.ReLU()(out)

            return out

    @register_op_wrapper("ReLU")
    def adaptive_activation_wrapper(x):
        def adaptive_activation(x):
            mean_val = ops.ReduceMean()(ops.Abs()(x))
            return mint.where(mean_val > 1.0,
                              ops.ReLU()(x),
                              ops.Tanh()(x))
        return adaptive_activation

    @register_op_wrapper("Add")
    def residual_fusion_wrapper(x, y):
        def residual_fusion(x, y):
            scale_factor = 0.8
            return x + y * scale_factor
        return residual_fusion

    net = ResidualBlock(channels=16)
    x = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float32))
    result = net(x)
    result_np = result.asnumpy()
    assert not np.allclose(result_np, np.zeros_like(result_np))

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_register_op_wrapper_error_invalid_op_name():
    """
    Feature: Custom op wrapper error handling.
    Description: Test error handling for invalid operator names.
    Expectation: ValueError is raised for invalid operator names.
    """
    with pytest.raises(ValueError, match="op_name must be a non-empty string"):
        @register_op_wrapper("")
        def dummy_wrapper(x, y):
            return lambda x, y: x + y

    with pytest.raises(ValueError, match="op_name must be a non-empty string"):
        @register_op_wrapper(None)
        def dummy_wrapper(x, y):
            return lambda x, y: x + y

    with pytest.raises(ValueError, match="op_name must be a non-empty string"):
        @register_op_wrapper(123)
        def dummy_wrapper(x, y):
            return lambda x, y: x + y

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_register_op_wrapper_error_duplicate_registration():
    """
    Feature: Custom op wrapper error handling for duplicate registration.
    Description: Test error handling when registering wrapper for same operator twice.
    Expectation: RuntimeError is raised for duplicate operator registration.
    """
    @register_op_wrapper("DuplicateAdd")
    def first_wrapper(x, y):
        return x + y

    with pytest.raises(RuntimeError, match="multiple times"):
        @register_op_wrapper("DuplicateAdd")
        def second_wrapper(x, y):
            return x + y

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_register_op_wrapper_error_invalid_return():
    """
    Feature: Custom op wrapper return string.
    Description: Test behavior when wrapper function doesn't return a callable.
    Expectation: Wrapper not working, no exception raised.
    """
    class InvalidReturnNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            return ops.Add()(x, y)

    @register_op_wrapper("Add")
    def invalid_wrapper(x, y):
        return "not a function"

    net = InvalidReturnNet()
    x = Tensor([[1., 2.], [3., 4.]])
    y = Tensor([[1., 1.], [1., 1.]])
    result = net(x, y)
    expected = np.array([[2., 3.], [4., 5.]])
    assert np.allclose(result.asnumpy(), expected)
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_register_op_wrapper_control_flow_2():
    """
    Feature: Custom op wrapper in control flow.
    Description: Test register_op_wrapper in a network with for loop.
    Expectation: Custom op wrappers work correctly.
    """
    class DeepRecursiveNet(ms.nn.Cell):
        def __init__(self, depth=3, layers=ms.nn.CellList()):
            super().__init__()
            self.depth = depth
            self.layers = layers

        @jit(backend='ms_backend')
        def construct(self, x):
            current = x
            for i in range(self.depth):
                current = self.layers[i](current)
                current = ops.ReLU()(current)
                if i > 0:
                    current = ops.Add()(current, x)

            return current

    @register_op_wrapper("ReLU")
    def progressive_activation_wrapper(x):
        def progressive_activation(x):
            mean_val = ops.ReduceMean()(ops.Abs()(x))
            linear_part = x
            non_linear_part = ops.Tanh()(x) * ops.Sigmoid()(mean_val)
            residual_part = ops.Sin()(x) * 0.1
            return linear_part + non_linear_part + residual_part
        return progressive_activation

    @register_op_wrapper("Add")
    def residual_enhancement_wrapper(x, y):
        def residual_enhancement(x, y):
            gate = ops.Sigmoid()(ops.ReduceMean(keep_dims=True)(x, axis=-1))
            scaled_residual = y * gate
            return x + scaled_residual
        return residual_enhancement

    class DeepRecursiveReplaceNet(ms.nn.Cell):
        def __init__(self, depth=3, layers=ms.nn.CellList()):
            super().__init__()
            self.depth = depth
            self.layers = layers

        @jit(backend='ms_backend')
        def construct(self, x):
            current = x
            for i in range(self.depth):
                current = self.layers[i](current)
                mean_val = ops.ReduceMean()(ops.Abs()(current))
                linear_part = current
                non_linear_part = ops.Tanh()(current) * ops.Sigmoid()(mean_val)
                residual_part = ops.Sin()(current) * 0.1
                current = linear_part + non_linear_part + residual_part
                if i > 0:
                    gate = ops.Sigmoid()(ops.ReduceMean(keep_dims=True)(current, axis=-1))
                    scaled_residual = x * gate
                    current = current + scaled_residual

            return current

    layers = ms.nn.CellList()
    for _ in range(3):
        layers.append(nn.Dense(16, 16))
    net = DeepRecursiveNet(3, layers)
    x = Tensor(np.random.randn(4, 16).astype(np.float32))
    result = net(x)
    teardown_module()
    expected_net = DeepRecursiveReplaceNet(3, layers)
    expected_net_result = expected_net(x)
    result_np = result.asnumpy()
    expected_net_result_np = expected_net_result.asnumpy()
    assert np.allclose(result_np, expected_net_result_np)

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_custom_op_wrapper_with_inplace():
    """
    Feature: Custom op wrapper.
    Description: Test register_op_wrapper decorator function.
    Expectation: Custom wrapper set to the operator in wrapper will not replace the operator.
    """
    class WrapperNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.y = Parameter(Tensor([[1., 2., 3.], [4., 5., 6.]]), name='y')

        @jit(backend='ms_backend')
        def construct(self, x):
            x = ops.Add()(x, self.y)
            x = x - self.y
            return x

    @register_op_wrapper("Add")
    def inplace_add(x, b):
        def func(x, b):
            b += x
            return x + b

        return func

    net = WrapperNet()
    input_x = Tensor([[1., 2., 3.], [4., 5., 6.]])
    res = net(input_x)
    expected = input_x
    assert np.allclose(res.asnumpy(), expected.asnumpy())
    teardown_module()
