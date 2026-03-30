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
"""Test auto_offload functionality in PyNative mode."""

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, context, ops, mint, _Function
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter
from mindspore._c_expression import OffloadManager


class ComplexNet(nn.Cell):
    """A complex network for testing auto_offload with multiple layers and operations."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(256)

    def construct(self, x):
        """Forward pass with multiple operations."""
        # First layer with batch norm
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)

        # Second layer with dropout
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.tanh(x)

        # Third layer
        x = self.fc3(x)
        x = self.relu(x)

        return x


def test_auto_offload_with_gradient():
    """
    Feature: Test auto_offload with gradient computation.
    Description: Test that auto_offload works correctly with forward and backward passes.
    Expectation: The network should execute correctly with offload enabled, and gradients should be computed correctly.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create network and enable auto_offload
    net = ComplexNet()
    net.auto_offload()
    net.set_grad()
    # Create input tensor
    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Forward pass (warm-up phase)
    output1 = net(x)
    assert output1.shape == (32, 64), f"Expected output shape (32, 64), got {output1.shape}"

    # Test gradient computation with auto_offload
    grad_net = C.GradOperation(get_all=True, get_by_list=True)

    # Compute gradients for inputs and parameters
    grads1 = grad_net(net, list(net.trainable_params()))(x)

    # Analyze after first step to generate offload plan
    manager.analyse()

    grads2 = grad_net(net, list(net.trainable_params()))(x)

    for grad1, grad2 in zip(grads1[1], grads2[1]):
        assert grad1.shape == grad2.shape, f"Gradient shape mismatch: {grad1.shape} vs {grad2.shape}"
        np.testing.assert_allclose(grad1.asnumpy(), grad2.asnumpy(), rtol=1e-4, atol=1e-4)


class ViewOpNet(nn.Cell):
    """Network with view operations (reshape, transpose) for testing tracer."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with view operations."""
        # x: [batch, 128]
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        # View operation: reshape
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 16, 16)  # [batch, 16, 16]

        # Use the reshaped tensor
        x = x + 1.0

        # View operation: reshape back
        x = x.reshape(batch_size, 256)  # [batch, 256]

        x = self.fc2(x)  # [batch, 128]
        x = self.relu(x)

        return x


def test_auto_offload_with_view_ops():
    """
    Feature: Test auto_offload with view operations.
    Description: View ops (reshape, transpose) share device_address but have different tensor objects.
                 The tracer should correctly handle this using composite key (addr, tensor_id).
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create network with view ops and enable auto_offload
    net = ViewOpNet()
    net.auto_offload()

    # Create input tensor
    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = ViewOpNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_view_ops passed successfully")


class InplaceOpNet(nn.Cell):
    """Network with inplace operations for testing tracer."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')

    def construct(self, x):
        """Forward pass with inplace-like operations."""
        x = self.fc1(x)  # [batch, 256]

        # Inplace-like operation: add scalar (simulated)
        # Note: MindSpore ops are typically not inplace, but we simulate
        # the scenario where output overwrites input tensor
        x +=1.0
        x = ops.relu(x)

        # Another computation using the modified tensor
        x = x * 2.0

        x = self.fc2(x)  # [batch, 128]

        return x


def test_auto_offload_with_inplace_ops():
    """
    Feature: Test auto_offload with inplace-like operations.
    Description: Inplace ops modify tensor in place (same tensor object).
                 The tracer should correctly overwrite the node mapping.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create network and enable auto_offload
    net = InplaceOpNet()
    net.auto_offload()

    # Create input tensor
    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = InplaceOpNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_inplace_ops passed successfully")


class ChainedViewOpNet(nn.Cell):
    """Network with chained view operations for testing tracer."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with chained view operations."""
        # x: [batch, 64]
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        batch_size = x.shape[0]

        # Chained view: A -> reshape -> B -> transpose -> C
        # All share the same device_address but different tensor_ids
        a = x  # Original tensor
        b = a.reshape(batch_size, 8, 16)  # View 1: [batch, 8, 16]
        c = b.transpose(0, 2, 1)  # View 2: [batch, 16, 8]

        # Use both the original and views
        # This tests that the tracer correctly tracks different tensor_ids
        a_sum = ops.reduce_mean(a)  # Use original
        c_sum = ops.reduce_mean(c)  # Use view 2

        # Combine results
        x = a + a_sum + c_sum

        # Continue processing
        x = self.fc2(x)  # [batch, 64]

        return x


def test_auto_offload_with_chained_view_ops():
    """
    Feature: Test auto_offload with chained view operations.
    Description: Multiple views from the same tensor (A -> B -> C) all share device_address
                 but have different tensor_ids. The tracer should track each separately.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create network and enable auto_offload
    net = ChainedViewOpNet()
    net.auto_offload()

    # Create input tensor
    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = ChainedViewOpNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_chained_view_ops passed successfully")


class ViewThenInplaceNet(nn.Cell):
    """Network with view followed by inplace-like operation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')

    def construct(self, x):
        """Forward pass with view then inplace-like operations."""
        # x: [batch, 64]
        x = self.fc1(x)  # [batch, 128]

        batch_size = x.shape[0]

        # View operation
        y = x.reshape(batch_size, 8, 16)  # View: [batch, 8, 16]

        # Inplace-like operation on the view
        y += 1.0  # Modify view
        y = ops.relu(y)

        # Reshape back
        y = y.reshape(batch_size, 128)  # [batch, 128]

        # Continue with the result
        y = self.fc2(y)  # [batch, 64]

        return y


def test_auto_offload_view_then_inplace():
    """
    Feature: Test auto_offload with view followed by inplace-like operation.
    Description: After view, perform inplace-like operation. The tracer should
                 correctly track the modified view tensor.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create network and enable auto_offload
    net = ViewThenInplaceNet()
    net.auto_offload()

    # Create input tensor
    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = ViewThenInplaceNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_view_then_inplace passed successfully")


class MultiStreamNet(nn.Cell):
    """Network for testing auto_offload with multiple streams."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x, stream1):
        """Forward pass with operations on different streams."""
        # x: [batch, 64]
        # Operations on default stream
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        # Record event on default stream before switching
        ev = ms.runtime.current_stream().record_event()

        # Operations on stream1 (wait for default stream first)
        with ms.runtime.StreamCtx(stream1):
            stream1.wait_event(ev)
            x = x + 1.0
            x = x * 2.0

        # Synchronize stream1 before continuing on default stream
        stream1.synchronize()

        # Continue on default stream
        x = self.fc2(x)  # [batch, 64]
        x = self.relu(x)

        return x


def test_auto_offload_with_multi_stream():
    """
    Feature: Test auto_offload with multiple streams.
    Description: Part of the computation is executed on a different stream.
                 The tracer should correctly track operations across streams.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create a separate stream for some operations
    stream1 = ms.runtime.Stream()

    # Create network and enable auto_offload
    net = MultiStreamNet()
    net.auto_offload()

    # Create input tensor
    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = MultiStreamNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x, stream1)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x, stream1)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x, stream1)

    # Synchronize all streams before verification
    ms.runtime.synchronize()

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_multi_stream passed successfully")


class MultiStreamChainNet(nn.Cell):
    """Network with chained multi-stream operations for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 128, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x, stream1, stream2):
        """Forward pass with chained operations on multiple streams."""
        # x: [batch, 64]
        # Stage 1: Default stream
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        # Stage 2: Stream1 (depends on default stream)
        ev1 = ms.runtime.current_stream().record_event()
        with ms.runtime.StreamCtx(stream1):
            stream1.wait_event(ev1)
            x = x + 1.0
            x = ops.tanh(x)

        # Record completion of stream1
        ev2 = stream1.record_event()

        # Stage 3: Stream2 (depends on stream1)
        with ms.runtime.StreamCtx(stream2):
            stream2.wait_event(ev2)
            x = x * 2.0
            x = self.fc2(x)  # [batch, 128]

        # Record completion of stream2
        ev3 = stream2.record_event()

        # Stage 4: Back to default stream (depends on stream2)
        ms.runtime.current_stream().wait_event(ev3)
        x = self.fc3(x)  # [batch, 64]
        x = self.relu(x)

        return x


def test_auto_offload_with_multi_stream_chain():
    """
    Feature: Test auto_offload with chained multi-stream operations.
    Description: Computations are distributed across default stream -> stream1 -> stream2 -> default.
                 This tests that the tracer correctly handles stream switching and synchronization.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create two separate streams
    stream1 = ms.runtime.Stream()
    stream2 = ms.runtime.Stream()

    # Create network and enable auto_offload
    net = MultiStreamChainNet()
    net.auto_offload()

    # Create input tensor
    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = MultiStreamChainNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x, stream1, stream2)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x, stream1, stream2)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x, stream1, stream2)

    # Synchronize all streams before verification
    ms.runtime.synchronize()

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_multi_stream_chain passed successfully")


class SimpleMultiStreamNet(nn.Cell):
    """Simple network for testing auto_offload with one additional stream."""

    def __init__(self):
        super().__init__()
        # Use Parameter instead of Dense for simple weight
        self.weight = Parameter(Tensor(np.ones((32,)).astype(np.float32)), name='weight')

    def construct(self, x, stream1, stream2):
        """Simple forward pass with stream switching."""
        # Default stream: element-wise multiplication with weight
        x = x * self.weight

        # Stage 2: Stream1 (depends on default stream)
        ev1 = ms.runtime.current_stream().record_event()
        with ms.runtime.StreamCtx(stream1):
            stream1.wait_event(ev1)
            x = x + 1.0
            x = ops.tanh(x)

        # Record completion of stream1
        ev2 = stream1.record_event()

        # Stage 3: Stream2 (depends on stream1)
        with ms.runtime.StreamCtx(stream2):
            stream2.wait_event(ev2)
            x = x * 2.0

        # Record completion of stream2
        ev3 = stream2.record_event()

        # Stage 4: Back to default stream (depends on stream2)
        ms.runtime.current_stream().wait_event(ev3)

        return x


def test_auto_offload_with_simple_multi_stream():
    """
    Feature: Test auto_offload with simple multi-stream operations.
    Description: Simple test case with minimal operators for debugging.
                 Uses element-wise multiplication and ReLU (which needs activation value for backward).
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    # Reset offload manager before test
    manager = OffloadManager.get_instance()
    manager.reset()

    # Create one additional stream
    stream1 = ms.runtime.Stream()
    stream2 = ms.runtime.Stream()

    # Create network and enable auto_offload
    net = SimpleMultiStreamNet()
    net.auto_offload()

    # Create small input tensor for faster execution
    x = Tensor(np.random.randn(8, 32).astype(np.float32))

    # Reference: compute expected gradients without auto_offload
    net_ref = SimpleMultiStreamNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x, stream1, stream2)

    # Warm-up phase: complete forward + backward pass
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x, stream1, stream2)

    # Analyze after warm-up to generate offload plan
    manager.analyse()

    # Execution phase: complete forward + backward pass again
    exec_grads = grad_fn(net, list(net.trainable_params()))(x, stream1, stream2)

    # Synchronize all streams before verification
    ms.runtime.synchronize()

    # Verify input gradients
    assert warmup_grads[0][0].shape == ref_grads[0][0].shape, "Warmup input gradient shape mismatch"
    assert exec_grads[0][0].shape == ref_grads[0][0].shape, "Exec input gradient shape mismatch"
    np.testing.assert_allclose(warmup_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4)

    # Verify parameter gradients
    for i, (wg, eg, rg) in enumerate(zip(warmup_grads[1], exec_grads[1], ref_grads[1])):
        assert wg.shape == rg.shape, f"Warmup param[{i}] gradient shape mismatch"
        assert eg.shape == rg.shape, f"Exec param[{i}] gradient shape mismatch"
        np.testing.assert_allclose(wg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_simple_multi_stream passed successfully")


# ============================================================================
# P0 Priority Test Cases - Must Cover
# ============================================================================

class BranchTopologyNet(nn.Cell):
    """Network with branch topology: single tensor used by multiple operators."""

    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32)), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32)), name='weight2')

    def construct(self, x):
        """Forward pass with branch topology: A used by both B and C."""
        # x: [batch, 64]
        # A = x * weight1 (common tensor)
        a = mint.matmul(x, self.weight1)
        a = mint.nn.functional.relu(a)

        # Branch 1: B = A + 1
        b = mint.add(a, 1.0)
        b = mint.tanh(b)

        # Branch 2: C = A * 2
        c = mint.mul(a, 2.0)
        c = mint.sigmoid(c)

        # Merge: D = B + C (both depend on A)
        d = mint.add(b, c)

        # Final output
        out = mint.matmul(d, self.weight2)
        return out


def test_auto_offload_branch_topology():
    """
    Feature: Test auto_offload with branch topology.
    Description: Single tensor (A) is used by multiple operators (B and C).
                 This is a common source of precision issues if tensor tracking is incorrect.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = BranchTopologyNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = BranchTopologyNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    # Execution phase - multiple iterations
    for iter_idx in range(3):
        exec_grads = grad_fn(net, list(net.trainable_params()))(x)

        # Verify input gradients
        np.testing.assert_allclose(
            exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(),
            rtol=1e-4, atol=1e-4,
            err_msg=f"Input gradient mismatch at iteration {iter_idx}"
        )

        # Verify parameter gradients
        for i, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
            np.testing.assert_allclose(
                eg.asnumpy(), rg.asnumpy(),
                rtol=1e-4, atol=1e-4,
                err_msg=f"Param[{i}] gradient mismatch at iteration {iter_idx}"
            )

    print("test_auto_offload_branch_topology passed successfully")


class ContiguousAfterViewNet(nn.Cell):
    """Network testing tensor copy after view operation."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((128, 64)).astype(np.float32)), name='weight')

    def construct(self, x):
        """Forward pass with view followed by contiguous (copy)."""
        # x: [batch, 64]
        y = mint.matmul(x, self.weight.T)  # [batch, 128]
        y = mint.nn.functional.relu(y)

        batch_size = y.shape[0]

        # View operation
        y_view = y.reshape(batch_size, 8, 16)

        # Transpose creates a non-contiguous view
        y_transposed = mint.transpose(y_view, 1, 2)  # [batch, 16, 8]

        # contiguous() creates a copy with new tensor_id
        y_contig = y_transposed.contiguous()

        # Use the contiguous tensor
        y_contig = mint.nn.functional.relu(y_contig)

        # Reshape back
        out = y_contig.reshape(batch_size, 128)

        return out


def test_auto_offload_contiguous_after_view():
    """
    Feature: Test auto_offload with contiguous after view.
    Description: contiguous() creates a tensor copy with new tensor_id but potentially
                 same device_address pattern. Tests tensor tracking correctness.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ContiguousAfterViewNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = ContiguousAfterViewNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_contiguous_after_view passed successfully")


class ResidualNet(nn.Cell):
    """Network with residual connection (x + F(x))."""

    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight2')

    def construct(self, x):
        """Forward pass with residual connection."""
        # x: [batch, 64]
        identity = x

        # F(x): transform branch
        out = mint.matmul(x, self.weight1)
        out = mint.nn.functional.relu(out)
        out = mint.matmul(out, self.weight2)

        # Residual: x + F(x)
        out = mint.add(out, identity)
        out = mint.nn.functional.relu(out)

        return out


def test_auto_offload_residual_connection():
    """
    Feature: Test auto_offload with residual connection.
    Description: Residual connection (x + F(x)) is the most common pattern in modern networks.
                 Tests that the original tensor is correctly tracked across skip connection.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ResidualNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = ResidualNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    # Test multiple iterations for stability
    for iter_idx in range(5):
        exec_grads = grad_fn(net, list(net.trainable_params()))(x)

        np.testing.assert_allclose(
            exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(),
            rtol=1e-4, atol=1e-4,
            err_msg=f"Iteration {iter_idx} failed"
        )
        for i, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
            np.testing.assert_allclose(
                eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4,
                err_msg=f"Param[{i}] at iteration {iter_idx} failed"
            )

    print("test_auto_offload_residual_connection passed successfully")


class CustomBpropNet(nn.Cell):
    """Network with custom bprop (user-defined backward)."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((64, 64)).astype(np.float32)), name='weight')

    def construct(self, x):
        """Forward: y = x * weight, then custom scale."""
        y = mint.matmul(x, self.weight)
        y = mint.nn.functional.relu(y)
        # Custom operation: scale by 2
        y = y * 2.0
        return y

    def bprop(self, x, out, dout):
        """Custom backward: scale gradient by 3 instead of 2."""
        # This tests that custom bprop is correctly traced
        grad_x = mint.matmul(dout * 3.0, self.weight.T)
        grad_w = mint.matmul(x.T, dout * 3.0)
        return (grad_x,), {self.weight: grad_w}


class CustomBpropWrapper(nn.Cell):
    """Wrapper to use CustomBpropNet in gradient computation."""

    def __init__(self):
        super().__init__()
        self.custom_net = CustomBpropNet()

    def construct(self, x):
        return self.custom_net(x)


def test_auto_offload_custom_bprop():
    """
    Feature: Test auto_offload with custom bprop.
    Description: User-defined backward (bprop) should be correctly traced and executed.
    Expectation: Forward and backward passes should produce correct results with custom gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = CustomBpropWrapper()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = CustomBpropWrapper()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_custom_bprop passed successfully")


def test_auto_offload_multiple_iterations():
    """
    Feature: Test auto_offload stability across multiple iterations.
    Description: Run multiple forward-backward passes to verify offload stability.
    Expectation: Results should be consistent across all iterations.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ResidualNet()
    net.auto_offload()

    net_ref = ResidualNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)

    # Warm-up with first input
    x1 = Tensor(np.random.randn(16, 64).astype(np.float32))
    grad_fn(net_ref, list(net_ref.trainable_params()))(x1)
    grad_fn(net, list(net.trainable_params()))(x1)

    manager.analyse()

    # Test with multiple different inputs
    for iter_idx in range(10):
        x = Tensor(np.random.randn(16, 64).astype(np.float32))
        ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)
        exec_grads = grad_fn(net, list(net.trainable_params()))(x)

        np.testing.assert_allclose(
            exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(),
            rtol=1e-4, atol=1e-4,
            err_msg=f"Input gradient mismatch at iteration {iter_idx}"
        )
        for i, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
            np.testing.assert_allclose(
                eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4,
                err_msg=f"Param[{i}] gradient mismatch at iteration {iter_idx}"
            )

    print("test_auto_offload_multiple_iterations passed successfully")


# ============================================================================
# P1 Priority Test Cases - High Priority
# ============================================================================

class ConvNet(nn.Cell):
    """Network with convolution operations."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                               weight_init='ones', bias_init='zeros')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                               weight_init='ones', bias_init='zeros')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        """Forward pass with conv operations."""
        # x: [batch, 3, H, W]
        x = self.conv1(x)
        x = mint.nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = mint.nn.functional.relu(x)

        # Global average pooling
        x = mint.mean(x, dim=(2, 3))

        return x


def test_auto_offload_conv_net():
    """
    Feature: Test auto_offload with convolution network.
    Description: Conv2d has multiple inputs (kernel) and backward needs activation values.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ConvNet()
    net.auto_offload()

    # [batch, channels, height, width]
    x = Tensor(np.random.randn(4, 3, 32, 32).astype(np.float32))

    net_ref = ConvNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-3, atol=1e-3
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-3, atol=1e-3)

    print("test_auto_offload_conv_net passed successfully")


class TransposeViewNet(nn.Cell):
    """Network with transpose view operation."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((64, 64)).astype(np.float32)), name='weight')

    def construct(self, x):
        """Forward pass with transpose view."""
        # x: [batch, 64]
        y = mint.matmul(x, self.weight)
        y = mint.nn.functional.relu(y)

        batch_size = y.shape[0]

        # Reshape to 3D
        y = y.reshape(batch_size, 8, 8)

        # Transpose (view operation, not copy)
        y_t = mint.transpose(y, 1, 2)

        # Use transposed tensor
        y_t = mint.nn.functional.relu(y_t)

        # Also use original for branch topology
        y_sum = mint.sum(y)

        # Reshape back
        out = y_t.reshape(batch_size, 64)
        out = mint.add(out, y_sum)

        return out


def test_auto_offload_transpose_view():
    """
    Feature: Test auto_offload with transpose view.
    Description: transpose creates a view with same device_address but different tensor_id.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = TransposeViewNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = TransposeViewNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_transpose_view passed successfully")


class MultiOutputNet(nn.Cell):
    """Network with multi-output operator (split)."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((64, 128)).astype(np.float32)), name='weight')

    def construct(self, x):
        """Forward pass with split (multi-output) operation."""
        # x: [batch, 64]
        y = mint.matmul(x, self.weight)  # [batch, 128]
        y = mint.nn.functional.relu(y)

        # Split into 2 parts (multi-output)
        y1, y2 = mint.split(y, 64, dim=1)  # Each [batch, 64]

        # Process each part differently
        y1 = mint.tanh(y1)
        y2 = mint.sigmoid(y2)

        # Combine
        out = mint.add(y1, y2)

        return out


def test_auto_offload_multi_output():
    """
    Feature: Test auto_offload with multi-output operator.
    Description: Split produces multiple outputs from single input.
                 Tests TupleGetItem tracking in the tracer.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = MultiOutputNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = MultiOutputNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_multi_output passed successfully")


class DiamondTopologyNet(nn.Cell):
    """Network with diamond topology: A -> B -> D and A -> C -> D."""

    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight2')

    def construct(self, x):
        """Forward pass with diamond topology."""
        # A: common input
        a = mint.matmul(x, self.weight1)
        a = mint.nn.functional.relu(a)

        # B: first branch
        b = mint.add(a, 1.0)
        b = mint.tanh(b)

        # C: second branch
        c = mint.matmul(a, self.weight2)
        c = mint.sigmoid(c)

        # D: merge both branches
        d = mint.mul(b, c)

        return d


def test_auto_offload_diamond_topology():
    """
    Feature: Test auto_offload with diamond topology.
    Description: Diamond pattern (A -> B -> D, A -> C -> D) is common in attention mechanisms.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = DiamondTopologyNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = DiamondTopologyNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_diamond_topology passed successfully")


class LargeTensorNet(nn.Cell):
    """Network with large tensors to trigger actual offload."""

    def __init__(self):
        super().__init__()
        # Large weight to increase memory usage
        self.weight = Parameter(Tensor(np.ones((512, 512)).astype(np.float32) * 0.01), name='weight')

    def construct(self, x):
        """Forward pass with large tensors."""
        # x: [batch, 512]
        y = mint.matmul(x, self.weight)
        y = mint.nn.functional.relu(y)

        # Multiple operations to create more intermediate tensors
        y = mint.add(y, 1.0)
        y = mint.mul(y, 2.0)
        y = mint.tanh(y)

        y = mint.matmul(y, self.weight)
        y = mint.nn.functional.relu(y)

        return y


def test_auto_offload_large_tensor():
    """
    Feature: Test auto_offload with large tensors.
    Description: Large tensors should trigger actual offload behavior.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = LargeTensorNet()
    net.auto_offload()

    # Large batch to increase memory usage
    x = Tensor(np.random.randn(64, 512).astype(np.float32))

    net_ref = LargeTensorNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_large_tensor passed successfully")


# ============================================================================
# P0 Priority Test Cases - Cross-Feature Tests
# ============================================================================

class CustomFunctionWithActivation(_Function):
    """Custom _Function that saves activation for backward."""

    @staticmethod
    def forward(ctx, x, weight):
        """Forward: y = relu(x @ weight), saves activation for backward."""
        out = mint.matmul(x, weight)
        activated = mint.nn.functional.relu(out)
        # Save tensors needed for backward
        ctx.save_for_backward(x, weight, activated)
        return activated

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: compute gradients using saved activation."""
        x, weight, activated = ctx.saved_tensors
        # ReLU backward: grad is 0 where activated <= 0
        relu_mask = (activated > 0).astype(ms.float32)
        grad_relu = grad_output * relu_mask
        # Matmul backward
        grad_x = mint.matmul(grad_relu, weight.T)
        grad_weight = mint.matmul(x.T, grad_relu)
        return grad_x, grad_weight


class CustomFunctionNet(nn.Cell):
    """Network using custom _Function with activation values needed in backward."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight')

    def construct(self, x):
        """Forward pass using custom _Function."""
        out = CustomFunctionWithActivation.apply(x, self.weight)
        out = mint.sum(out)
        return out


def test_auto_offload_custom_function():
    """
    Feature: Test auto_offload with _Function custom operator.
    Description: Custom _Function that saves activation values for backward.
                 This tests that offload correctly handles ctx.save_for_backward tensors.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = CustomFunctionNet()
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = CustomFunctionNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_custom_function passed successfully")


class RecomputeBlockNet(nn.Cell):
    """Block that uses activation values in backward (relu, matmul)."""

    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight2')

    def construct(self, x):
        """Forward with operators that need activation for backward."""
        # matmul backward needs x and weight
        y = mint.matmul(x, self.weight1)
        # relu backward needs forward output to compute mask
        y = mint.nn.functional.relu(y)
        # another matmul
        y = mint.matmul(y, self.weight2)
        y = mint.tanh(y)
        return y


class RecomputeNet(nn.Cell):
    """Network with recompute enabled for some blocks."""

    def __init__(self, use_reentrant=True):
        super().__init__()
        self.block1 = RecomputeBlockNet()
        self.block2 = RecomputeBlockNet()
        self.use_reentrant = use_reentrant
        # Set recompute for block1
        self.block1.recompute(use_reentrant=use_reentrant)

    def construct(self, x):
        """Forward pass with recompute block."""
        x = self.block1(x)
        x = self.block2(x)
        return x


def test_auto_offload_with_recompute_reentrant():
    """
    Feature: Test auto_offload with recompute (use_reentrant=True).
    Description: Recompute with use_reentrant=True uses checkpointing to save memory.
                 Tests that offload works correctly with recomputed activations.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = RecomputeNet(use_reentrant=True)
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = RecomputeNet(use_reentrant=True)
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_recompute_reentrant passed successfully")


def test_auto_offload_with_recompute_non_reentrant():
    """
    Feature: Test auto_offload with recompute (use_reentrant=False).
    Description: Recompute with use_reentrant=False uses different mechanism.
                 Tests that offload works correctly with non-reentrant recomputation.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = RecomputeNet(use_reentrant=False)
    net.auto_offload()

    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = RecomputeNet(use_reentrant=False)
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print("test_auto_offload_with_recompute_non_reentrant passed successfully")


if __name__ == "__main__":
    # Original tests
    test_auto_offload_with_gradient()
    test_auto_offload_with_view_ops()
    test_auto_offload_with_inplace_ops()
    test_auto_offload_with_chained_view_ops()
    test_auto_offload_view_then_inplace()
    test_auto_offload_with_multi_stream()
    test_auto_offload_with_multi_stream_chain()
    test_auto_offload_with_simple_multi_stream()

    # P0 tests
    test_auto_offload_branch_topology()
    test_auto_offload_contiguous_after_view()
    test_auto_offload_residual_connection()
    test_auto_offload_custom_bprop()
    test_auto_offload_multiple_iterations()

    # P0 cross-feature tests
    test_auto_offload_custom_function()
    test_auto_offload_with_recompute_reentrant()
    test_auto_offload_with_recompute_non_reentrant()

    # P1 tests
    test_auto_offload_conv_net()
    test_auto_offload_transpose_view()
    test_auto_offload_multi_output()
    test_auto_offload_diamond_topology()
    test_auto_offload_large_tensor()
