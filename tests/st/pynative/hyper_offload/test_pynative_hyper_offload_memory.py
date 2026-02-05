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
"""Test memory peak detection for HyperOffload in PyNative mode.

This module separates gradient accuracy tests from memory tests to avoid
tensor resource interference during memory measurement.
"""

import os
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, Tensor, context, ops, mint
from mindspore import hal
from mindspore._c_expression import OffloadManager
from mindspore.graph._parse import compile_config

os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"

RTOL = 1e-4
ATOL = 1e-4


class SimpleNet(nn.Cell):
    """Simple network for testing."""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ConvNet(nn.Cell):
    """Convolutional network for testing."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, weight_init='ones', has_bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, weight_init='ones', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class DeepResidualNet(nn.Cell):
    """Deep network with residual connections."""

    def __init__(self, num_layers=4):
        super(DeepResidualNet, self).__init__()
        self.fc_input = nn.Dense(128, 128, weight_init='ones', bias_init='zeros')
        self.layers = nn.CellList()
        for _ in range(num_layers):
            self.layers.append(nn.Dense(128, 128, weight_init='ones', bias_init='zeros'))
        self.fc_output = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()
        self.add = ops.add

    def construct(self, x):
        x = self.fc_input(x)
        x = self.relu(x)
        identity = x
        for i, layer in enumerate(self.layers):
            out = layer(x)
            out = self.relu(out)
            if i == 0:
                identity = out
            out = self.add(out, identity)
            out = self.relu(out)
            identity = out
        x = self.fc_output(x)
        return x


class LargeTensorNet(nn.Cell):
    """Network with large intermediate tensors."""

    def __init__(self):
        super(LargeTensorNet, self).__init__()
        self.matmul1 = ops.matmul
        self.matmul2 = ops.matmul
        self.relu = ops.relu

    def construct(self, x):
        batch_size = x.shape[0]
        # x: (batch_size, 128), large_tensor1: (batch_size, 128, 512)
        # matmul(x, large_tensor1) -> (batch_size, 512)
        large_tensor1 = ops.randn(batch_size, 128, 512)
        large_tensor2 = ops.randn(batch_size, 512, 512)
        intermediate1 = self.matmul1(x, large_tensor1)
        intermediate1 = self.relu(intermediate1)
        intermediate2 = self.matmul2(intermediate1, large_tensor2)
        intermediate2 = self.relu(intermediate2)
        output = ops.sum(intermediate2, dim=-1)
        return output


def _setup_offload_config():
    """Setup HyperOffload configuration parameters."""
    origin_select = compile_config.HYPER_OFFLOAD_SELECT_DISTANCE
    origin_prefetch = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = 1
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = 0
    return origin_select, origin_prefetch


def _restore_offload_config(origin_select, origin_prefetch):
    """Restore HyperOffload configuration."""
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = origin_select
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = origin_prefetch


def _verify_gradients(grads_with_offload, grads_no_offload):
    """Verify that gradients are numerically consistent."""
    if isinstance(grads_no_offload, tuple):
        for i, (gw, gn) in enumerate(zip(grads_with_offload, grads_no_offload)):
            if isinstance(gn, (tuple, list)):
                _verify_gradients(gw, gn)
            elif hasattr(gw, 'asnumpy') and hasattr(gn, 'asnumpy'):
                np.testing.assert_allclose(
                    gw.asnumpy(), gn.asnumpy(), rtol=RTOL, atol=ATOL,
                    err_msg=f"Gradient mismatch at index {i}"
                )
    elif hasattr(grads_with_offload, 'asnumpy') and hasattr(grads_no_offload, 'asnumpy'):
        np.testing.assert_allclose(
            grads_with_offload.asnumpy(), grads_no_offload.asnumpy(),
            rtol=RTOL, atol=ATOL, err_msg="Gradient mismatch"
        )


def _cleanup_resources(*args):
    """Explicitly delete variables to free memory."""
    for var in args:
        if var is not None:
            del var


# ============================================================================
# Part 1: Gradient Accuracy Tests (without memory monitoring)
# ============================================================================

def test_auto_offload_accuracy_simple():
    """
    Feature: Test gradient accuracy with auto_offload for simple network.
    Description: Verify that gradients are numerically consistent with and without offload.
    Expectation: Gradients should be numerically consistent.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = SimpleNet()
    net_no_offload.set_grad()
    grads_no_offload = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = SimpleNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    for p1, p2 in zip(net_with_offload.trainable_params(), net_no_offload.trainable_params()):
        p2.set_data(p1.data.copy())

    _ = net_with_offload(x)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    manager.analyse()

    grads_with_offload = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    _verify_gradients(grads_with_offload, grads_no_offload)

    _cleanup_resources(grads_no_offload, grads_with_offload, net_no_offload, net_with_offload, x)
    _restore_offload_config(origin_select, origin_prefetch)


def test_auto_offload_accuracy_conv_net():
    """
    Feature: Test gradient accuracy with auto_offload for convolutional network.
    Description: Verify gradient accuracy for CNN with and without auto_offload.
    Expectation: Gradients should be numerically consistent.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 16
    x = Tensor(np.random.randn(batch_size, 3, 32, 32).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = ConvNet()
    net_no_offload.set_grad()
    grads_no_offload = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = ConvNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    for p1, p2 in zip(net_with_offload.trainable_params(), net_no_offload.trainable_params()):
        p2.set_data(p1.data.copy())

    _ = net_with_offload(x)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    manager.analyse()

    grads_with_offload = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    _verify_gradients(grads_with_offload, grads_no_offload)

    _cleanup_resources(grads_no_offload, grads_with_offload, net_no_offload, net_with_offload, x)
    _restore_offload_config(origin_select, origin_prefetch)


def test_auto_offload_accuracy_deep_residual():
    """
    Feature: Test gradient accuracy with auto_offload for deep residual network.
    Description: Verify gradient accuracy for deep network with residual connections.
    Expectation: Gradients should be numerically consistent.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = DeepResidualNet(num_layers=6)
    net_no_offload.set_grad()
    grads_no_offload = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = DeepResidualNet(num_layers=6)
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    for p1, p2 in zip(net_with_offload.trainable_params(), net_no_offload.trainable_params()):
        p2.set_data(p1.data.copy())

    _ = net_with_offload(x)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    manager.analyse()

    grads_with_offload = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    _verify_gradients(grads_with_offload, grads_no_offload)

    _cleanup_resources(grads_no_offload, grads_with_offload, net_no_offload, net_with_offload, x)
    _restore_offload_config(origin_select, origin_prefetch)


def test_auto_offload_accuracy_large_tensors():
    """
    Feature: Test gradient accuracy with auto_offload for network with large tensors.
    Description: Verify gradient accuracy with large intermediate tensors.
    Expectation: Gradients should be numerically consistent.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 8
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True)

    net_no_offload = LargeTensorNet()
    net_no_offload.set_grad()
    grads_no_offload = grad_net(net_no_offload)(x)

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = LargeTensorNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    _ = net_with_offload(x)
    _ = grad_net(net_with_offload)(x)

    manager.analyse()

    grads_with_offload = grad_net(net_with_offload)(x)

    _verify_gradients(grads_with_offload, grads_no_offload)

    _cleanup_resources(grads_no_offload, grads_with_offload, net_no_offload, net_with_offload, x)
    _restore_offload_config(origin_select, origin_prefetch)


def test_auto_offload_accuracy_multiple_iterations():
    """
    Feature: Test gradient accuracy stability across multiple iterations.
    Description: Verify gradient accuracy is consistent across multiple forward-backward passes.
    Expectation: Gradients should be numerically consistent across all iterations.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = SimpleNet()
    net_no_offload.set_grad()

    for _ in range(5):
        _ = net_no_offload(x)
        grads_no_offload = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = SimpleNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    for p1, p2 in zip(net_with_offload.trainable_params(), net_no_offload.trainable_params()):
        p2.set_data(p1.data.copy())

    _ = net_with_offload(x)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    manager.analyse()

    for _ in range(5):
        _ = net_with_offload(x)
        grads_with_offload = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x)

    _verify_gradients(grads_with_offload, grads_no_offload)

    _cleanup_resources(grads_no_offload, grads_with_offload, net_no_offload, net_with_offload, x)
    _restore_offload_config(origin_select, origin_prefetch)


# ============================================================================
# Part 2: Memory Peak Tests (separate from accuracy tests)
# ============================================================================

def test_auto_offload_memory_peak_simple():
    """
    Feature: Test memory peak with auto_offload for simple network.
    Description: Measure peak memory usage with and without offload.
    Expectation: Memory peak should be lower with auto_offload.
    Note: Accuracy is verified in test_auto_offload_accuracy_simple.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = SimpleNet()
    net_no_offload.set_grad()
    hal.reset_peak_memory_stats()
    _ = net_no_offload(x)
    _ = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)
    mem_no_offload = hal.max_memory_allocated()

    _cleanup_resources(net_no_offload, x)
    del x

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = SimpleNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    x_new = Tensor(np.random.randn(batch_size, 128).astype(np.float32))
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)

    manager.analyse()

    hal.reset_peak_memory_stats()
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)
    mem_with_offload = hal.max_memory_allocated()

    _cleanup_resources(net_with_offload, x_new)
    _restore_offload_config(origin_select, origin_prefetch)

    print(f"Memory without offload: {mem_no_offload}")
    print(f"Memory with offload: {mem_with_offload}")
    print(f"Memory reduction: {((mem_no_offload - mem_with_offload) / mem_no_offload) * 100:.2f}%")

    assert mem_with_offload < mem_no_offload, \
        f"Expected memory with offload ({mem_with_offload}) < without offload ({mem_no_offload})"


def test_auto_offload_memory_peak_conv_net():
    """
    Feature: Test memory peak with auto_offload for convolutional network.
    Description: Measure peak memory usage for CNN with and without offload.
    Expectation: Memory peak should be significantly lower with auto_offload.
    Note: Accuracy is verified in test_auto_offload_accuracy_conv_net.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 16
    x = Tensor(np.random.randn(batch_size, 3, 32, 32).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = ConvNet()
    net_no_offload.set_grad()
    hal.reset_peak_memory_stats()
    _ = net_no_offload(x)
    _ = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)
    mem_no_offload = hal.max_memory_allocated()

    _cleanup_resources(net_no_offload, x)
    del x

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = ConvNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    x_new = Tensor(np.random.randn(batch_size, 3, 32, 32).astype(np.float32))
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)

    manager.analyse()

    hal.reset_peak_memory_stats()
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)
    mem_with_offload = hal.max_memory_allocated()

    _cleanup_resources(net_with_offload, x_new)
    _restore_offload_config(origin_select, origin_prefetch)

    print(f"Memory without offload (ConvNet): {mem_no_offload}")
    print(f"Memory with offload (ConvNet): {mem_with_offload}")
    print(f"Memory reduction: {((mem_no_offload - mem_with_offload) / mem_no_offload) * 100:.2f}%")

    assert mem_with_offload < mem_no_offload, \
        f"Expected memory with offload ({mem_with_offload}) < without offload ({mem_no_offload})"


def test_auto_offload_memory_peak_deep_residual():
    """
    Feature: Test memory peak with auto_offload for deep residual network.
    Description: Measure peak memory usage for deep network with residual connections.
    Expectation: Memory peak should be lower with auto_offload.
    Note: Accuracy is verified in test_auto_offload_accuracy_deep_residual.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = DeepResidualNet(num_layers=6)
    net_no_offload.set_grad()
    hal.reset_peak_memory_stats()
    _ = net_no_offload(x)
    _ = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)
    mem_no_offload = hal.max_memory_allocated()

    _cleanup_resources(net_no_offload, x)
    del x

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = DeepResidualNet(num_layers=6)
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    x_new = Tensor(np.random.randn(batch_size, 128).astype(np.float32))
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)

    manager.analyse()

    hal.reset_peak_memory_stats()
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)
    mem_with_offload = hal.max_memory_allocated()

    _cleanup_resources(net_with_offload, x_new)
    _restore_offload_config(origin_select, origin_prefetch)

    print(f"Memory without offload (DeepResidual): {mem_no_offload}")
    print(f"Memory with offload (DeepResidual): {mem_with_offload}")
    print(f"Memory reduction: {((mem_no_offload - mem_with_offload) / mem_no_offload) * 100:.2f}%")

    assert mem_with_offload < mem_no_offload, \
        f"Expected memory with offload ({mem_with_offload}) < without offload ({mem_no_offload})"


def test_auto_offload_memory_peak_large_tensors():
    """
    Feature: Test memory peak with auto_offload for network with large tensors.
    Description: Measure peak memory usage with large intermediate tensors.
    Expectation: Memory peak reduction should be more significant with large tensors.
    Note: Accuracy is verified in test_auto_offload_accuracy_large_tensors.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 8
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True)

    net_no_offload = LargeTensorNet()
    net_no_offload.set_grad()
    hal.reset_peak_memory_stats()
    _ = net_no_offload(x)
    _ = grad_net(net_no_offload)(x)
    mem_no_offload = hal.max_memory_allocated()

    _cleanup_resources(net_no_offload, x)
    del x

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = LargeTensorNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    x_new = Tensor(np.random.randn(batch_size, 128).astype(np.float32))
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload)(x_new)

    manager.analyse()

    hal.reset_peak_memory_stats()
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload)(x_new)
    mem_with_offload = hal.max_memory_allocated()

    _cleanup_resources(net_with_offload, x_new)
    _restore_offload_config(origin_select, origin_prefetch)

    print(f"Memory without offload (LargeTensor): {mem_no_offload}")
    print(f"Memory with offload (LargeTensor): {mem_with_offload}")
    print(f"Memory reduction: {((mem_no_offload - mem_with_offload) / mem_no_offload) * 100:.2f}%")

    assert mem_with_offload < mem_no_offload, \
        f"Expected memory with offload ({mem_with_offload}) < without offload ({mem_no_offload})"


def test_auto_offload_memory_peak_multiple_iterations():
    """
    Feature: Test memory peak stability across multiple training iterations.
    Description: Measure memory usage consistency across multiple iterations.
    Expectation: Memory peak should remain lower across all iterations.
    Note: Accuracy is verified in test_auto_offload_accuracy_multiple_iterations.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = SimpleNet()
    net_no_offload.set_grad()
    hal.reset_peak_memory_stats()

    for _ in range(5):
        _ = net_no_offload(x)
        _ = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)

    mem_no_offload = hal.max_memory_allocated()

    _cleanup_resources(net_no_offload, x)
    del x

    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = SimpleNet()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()

    x_new = Tensor(np.random.randn(batch_size, 128).astype(np.float32))
    _ = net_with_offload(x_new)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)

    manager.analyse()

    hal.reset_peak_memory_stats()

    for _ in range(5):
        _ = net_with_offload(x_new)
        _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)

    mem_with_offload = hal.max_memory_allocated()

    _cleanup_resources(net_with_offload, x_new)
    _restore_offload_config(origin_select, origin_prefetch)

    print(f"Memory without offload (5 iterations): {mem_no_offload}")
    print(f"Memory with offload (5 iterations): {mem_with_offload}")
    print(f"Memory reduction: {((mem_no_offload - mem_with_offload) / mem_no_offload) * 100:.2f}%")

    assert mem_with_offload < mem_no_offload, \
        f"Expected memory with offload ({mem_with_offload}) < without offload ({mem_no_offload})"


def test_auto_offload_memory_peak_different_batch_sizes():
    """
    Feature: Test memory peak with different batch sizes.
    Description: Verify memory savings across various batch sizes.
    Expectation: Memory reduction should be consistent across different batch sizes.
    Note: Accuracy tests should be run separately for each batch size.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    origin_select, origin_prefetch = _setup_offload_config()

    batch_sizes = [8, 16, 32, 64]
    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    for batch_size in batch_sizes:
        x = Tensor(np.random.randn(batch_size, 128).astype(np.float32))

        net_no_offload = SimpleNet()
        net_no_offload.set_grad()
        hal.reset_peak_memory_stats()
        _ = net_no_offload(x)
        _ = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(x)
        mem_no_offload = hal.max_memory_allocated()

        _cleanup_resources(net_no_offload, x)
        del x

        manager = OffloadManager.get_instance()
        manager.reset()

        net_with_offload = SimpleNet()
        net_with_offload.auto_offload()
        net_with_offload.set_grad()

        x_new = Tensor(np.random.randn(batch_size, 128).astype(np.float32))
        _ = net_with_offload(x_new)
        _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)

        manager.analyse()

        hal.reset_peak_memory_stats()
        _ = net_with_offload(x_new)
        _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(x_new)
        mem_with_offload = hal.max_memory_allocated()

        _cleanup_resources(net_with_offload, x_new)

        reduction = ((mem_no_offload - mem_with_offload) / mem_no_offload) * 100
        print(f"Batch size {batch_size}: No offload={mem_no_offload}, With offload={mem_with_offload}, Reduction={reduction:.2f}%")

        assert mem_with_offload < mem_no_offload, \
            f"Batch size {batch_size}: Expected memory with offload ({mem_with_offload}) < without ({mem_no_offload})"

    _restore_offload_config(origin_select, origin_prefetch)
