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
"""
Test auto_offload functionality for view/inplace scenarios in PyNative mode.

This test file covers:
- View operations: Reshape, Transpose, View, ExpandDims, Unsqueeze, Squeeze, Flatten, Slice, StridedSlice, Narrow, Permute, Expand, ExpandAs
- Inplace operations: InplaceAdd, InplaceMul, InplaceIndexAdd
- Chained view operations
- View + Inplace combinations
"""

import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, Tensor, context, ops
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter
from mindspore._c_expression import OffloadManager


def sync_weights(net_with_offload, net_ref):
    """Sync weights from net_with_offload to net_ref."""
    for p1, p2 in zip(net_with_offload.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())


def verify_gradients(exec_grads, ref_grads, rtol=1e-4, atol=1e-4):
    """Verify that execution gradients match reference gradients."""
    # Verify input gradients
    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(),
        rtol=rtol, atol=atol,
        err_msg="Input gradient mismatch"
    )
    # Verify parameter gradients
    for i, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(
            eg.asnumpy(), rg.asnumpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Param[{i}] gradient mismatch"
        )


# =============================================================================
# 1.1 基础view算子测试 (Reshape, Transpose, View)
# =============================================================================

class ReshapeNet(nn.Cell):
    """Network with reshape operations for testing auto_offload."""

    def __init__(self):
        super(ReshapeNet, self).__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with reshape operations."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)
        batch_size = x.shape[0]
        # Reshape to 3D
        x = x.reshape(batch_size, 16, 16)  # [batch, 16, 16]
        # Use reshaped tensor
        x = x * 2.0
        # Reshape back to 2D
        x = x.reshape(batch_size, 256)  # [batch, 256]
        x = self.fc2(x)  # [batch, 128]
        x = self.relu(x)
        return x


class TransposeNet(nn.Cell):
    """Network with transpose operations for testing auto_offload."""

    def __init__(self):
        super(TransposeNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                               weight_init='ones', bias_init='zeros')
        self.fc = nn.Dense(16 * 32 * 32, 10, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with transpose operations."""
        x = self.conv(x)  # [batch, 16, 32, 32]
        x = self.relu(x)
        # Transpose: [batch, 16, 32, 32] -> [batch, 32, 16, 32]
        x = ops.transpose(x, (0, 2, 1, 3))
        # Use transposed tensor
        x = x * 1.5
        # Transpose back
        x = ops.transpose(x, (0, 2, 1, 3))
        # Flatten for fully connected
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


def test_auto_offload_view_reshape():
    """
    Feature: Test auto_offload with reshape operations.
    Description: Reshape operation shares underlying data without copying.
                 The tracer should correctly handle the tensor_id change while device_address stays same.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ReshapeNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = ReshapeNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_view_reshape passed successfully")


def test_auto_offload_view_transpose():
    """
    Feature: Test auto_offload with transpose operations.
    Description: Transpose operation changes dimension order but shares underlying data.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = TransposeNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(4, 3, 32, 32).astype(np.float32))

    # Reference without auto_offload
    net_ref = TransposeNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads, rtol=1e-3, atol=1e-3)
    print("test_auto_offload_view_transpose passed successfully")


# =============================================================================
# 1.2 Slice算子测试 (Slice, StridedSlice)
# =============================================================================

class SliceNet(nn.Cell):
    """Network with slice operations for testing auto_offload."""

    def __init__(self):
        super(SliceNet, self).__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with slice operations."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        # Slice: extract first 64 columns
        x_slice = x[:, :64]  # [batch, 64]
        x_slice = self.relu(x_slice)

        # Slice with step: every other row
        x_step = x[::2, :]  # [batch/2, 128]
        x_step = self.fc2(x_step)

        # Combine results
        x = x_slice + x_step
        return x


class StridedSliceNet(nn.Cell):
    """Network with strided_slice operations for testing auto_offload."""

    def __init__(self):
        super(StridedSliceNet, self).__init__()
        self.fc = nn.Dense(32, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with strided_slice operations."""
        x = self.fc(x)  # [batch, 64]
        x = self.relu(x)

        # Strided slice with start, end, and stride
        # Slice: rows 0:8:2, cols 0:16:1
        x_sliced = ops.strided_slice(x, (0, 0), (8, 16), (2, 1))
        x_sliced = self.relu(x_sliced)

        return x_sliced


def test_auto_offload_slice():
    """
    Feature: Test auto_offload with slice operations.
    Description: Slice operations (using tensor slicing syntax) share underlying data.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = SliceNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = SliceNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_slice passed successfully")


def test_auto_offload_strided_slice():
    """
    Feature: Test auto_offload with strided_slice operations.
    Description: Strided_slice operation with explicit start, end, stride parameters.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = StridedSliceNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = StridedSliceNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_strided_slice passed successfully")


# =============================================================================
# 1.3 Expand类算子测试 (Expand, ExpandAs, ExpandDims, Unsqueeze)
# =============================================================================

class ExpandDimsNet(nn.Cell):
    """Network with ExpandDims operations for testing auto_offload."""

    def __init__(self):
        super(ExpandDimsNet, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                               weight_init='ones', bias_init='zeros')
        self.fc = nn.Dense(16 * 32 * 32, 10, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with ExpandDims operations."""
        # x: [batch, 32, 32] -> [batch, 1, 32, 32]
        x = ops.expand_dims(x, 1)
        x = self.conv(x)  # [batch, 16, 32, 32]
        x = self.relu(x)
        # Flatten for fully connected
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


class ExpandNet(nn.Cell):
    """Network with Expand operations for testing auto_offload."""

    def __init__(self):
        super(ExpandNet, self).__init__()
        self.fc = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with Expand operations."""
        # x: [batch, 32]
        # Expand to [batch, 64] by broadcasting
        expanded = ops.expand(x, (-1, 64))
        x = self.fc(expanded)  # [batch, 128]
        x = self.relu(x)
        return x


class UnsqueezeNet(nn.Cell):
    """Network with Unsqueeze operations for testing auto_offload."""

    def __init__(self):
        super(UnsqueezeNet, self).__init__()
        self.conv = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                               weight_init='ones', bias_init='zeros')
        self.fc = nn.Dense(8 * 32 * 32 * 32, 10, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with Unsqueeze operations."""
        # x: [batch, 32, 32, 32] -> [batch, 1, 32, 32, 32]
        x = ops.unsqueeze(x, 1)
        x = self.conv(x)  # [batch, 8, 32, 32, 32]
        x = self.relu(x)
        # Flatten for fully connected
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


def test_auto_offload_expand_dims():
    """
    Feature: Test auto_offload with ExpandDims operations.
    Description: ExpandDims adds a dimension at specified position, sharing underlying data.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ExpandDimsNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(4, 32, 32).astype(np.float32))

    # Reference without auto_offload
    net_ref = ExpandDimsNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads, rtol=1e-3, atol=1e-3)
    print("test_auto_offload_expand_dims passed successfully")


def test_auto_offload_expand():
    """
    Feature: Test auto_offload with Expand operations.
    Description: Expand broadcasts tensor to new shape, sharing underlying data.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ExpandNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 32).astype(np.float32))

    # Reference without auto_offload
    net_ref = ExpandNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_expand passed successfully")


def test_auto_offload_unsqueeze():
    """
    Feature: Test auto_offload with Unsqueeze operations.
    Description: Unsqueeze adds a dimension at specified position (same as ExpandDims).
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = UnsqueezeNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(4, 32, 32, 32).astype(np.float32))

    # Reference without auto_offload
    net_ref = UnsqueezeNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads, rtol=1e-3, atol=1e-3)
    print("test_auto_offload_unsqueeze passed successfully")


# =============================================================================
# 1.4 Narrow操作测试
# =============================================================================

class NarrowNet(nn.Cell):
    """Network with narrow operations for testing auto_offload."""

    def __init__(self):
        super(NarrowNet, self).__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(32, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with narrow operations."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        # Narrow along dim 1: start=0, length=32
        x_narrow = ops.narrow(x, 1, 0, 32)  # [batch, 32]
        x_narrow = self.relu(x_narrow)

        # Narrow along dim 0: start=0, length=16 (for batch dimension)
        x_narrow2 = ops.narrow(x, 0, 0, 16)  # [16, 128]

        # Combine
        x = self.fc2(x_narrow)  # [batch, 64]
        return x


def test_auto_offload_narrow():
    """
    Feature: Test auto_offload with narrow operations.
    Description: Narrow operation extracts a slice along specified dimension.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = NarrowNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = NarrowNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_narrow passed successfully")


# =============================================================================
# 1.5 Permute操作测试
# =============================================================================

class PermuteNet(nn.Cell):
    """Network with permute operations for testing auto_offload."""

    def __init__(self):
        super(PermuteNet, self).__init__()
        self.fc1 = nn.Dense(128 * 64, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 10, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with permute operations."""
        x = x.reshape(32, 2, 64)  # [batch=32, 2, 64]
        x = self.relu(x)

        # Permute: [32, 2, 64] -> [32, 64, 2]
        x = ops.permute(x, (0, 2, 1))

        # Use permuted tensor
        x = x * 2.0

        # Permute back: [32, 64, 2] -> [32, 2, 64]
        x = ops.permute(x, (0, 2, 1))

        # Flatten and process
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_auto_offload_permute():
    """
    Feature: Test auto_offload with permute operations.
    Description: Permute operation rearranges dimensions, sharing underlying data.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = PermuteNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = PermuteNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_permute passed successfully")


# =============================================================================
# 1.6 Flatten/Squeeze操作测试
# =============================================================================

class FlattenSqueezeNet(nn.Cell):
    """Network with Flatten and Squeeze operations for testing auto_offload."""

    def __init__(self):
        super(FlattenSqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                                weight_init='ones', bias_init='zeros')
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                                weight_init='ones', bias_init='zeros')
        self.fc = nn.Dense(8 * 32 * 32, 10, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with Flatten and Squeeze operations."""
        # x: [batch, 1, 32, 32]
        x = self.conv1(x)
        x = self.relu(x)

        # Flatten: [batch, 1, 32, 32] -> [batch, 1024]
        x = ops.flatten(x, 1)

        x = self.fc(x)
        x = self.relu(x)

        # Reshape for next conv (not ideal but tests reshape tracking)
        x = x.reshape(32, 8, 4, 4)

        # Squeeze: remove dimensions of size 1
        x = ops.squeeze(x)  # [batch, 8, 4, 4] -> [batch, 8, 4, 4] (no change if no dim of size 1)

        return x


def test_auto_offload_flatten_squeeze():
    """
    Feature: Test auto_offload with Flatten and Squeeze operations.
    Description: Flatten and Squeeze operations modify tensor shape while potentially sharing data.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = FlattenSqueezeNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 1, 32, 32).astype(np.float32))

    # Reference without auto_offload
    net_ref = FlattenSqueezeNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads, rtol=1e-3, atol=1e-3)
    print("test_auto_offload_flatten_squeeze passed successfully")


# =============================================================================
# 1.7 链式view测试 (Reshape->Transpose->Reshape)
# =============================================================================

class ChainedViewNet(nn.Cell):
    """Network with chained view operations for testing auto_offload."""

    def __init__(self):
        super(ChainedViewNet, self).__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with chained view operations."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        batch_size = x.shape[0]

        # Chained view: A -> reshape -> B -> transpose -> C
        a = x  # Original: [batch, 128]
        b = a.reshape(batch_size, 16, 8)  # View1: [batch, 16, 8]
        c = b.transpose(0, 2, 1)  # View2: [batch, 8, 16]

        # Use all views
        a_sum = ops.reduce_mean(a)
        b_sum = ops.reduce_mean(b)
        c_sum = ops.reduce_mean(c)

        # Combine
        x = a + a_sum + b_sum + c_sum

        # Continue processing
        x = self.fc2(x)  # [batch, 64]
        x = self.relu(x)

        return x


def test_auto_offload_chained_view():
    """
    Feature: Test auto_offload with chained view operations.
    Description: Multiple views from the same tensor (A -> B -> C) all share device_address.
                 The tracer should track each separately.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ChainedViewNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = ChainedViewNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_chained_view passed successfully")


# =============================================================================
# 1.8 View + Inplace组合测试
# =============================================================================

class ViewInplaceNet(nn.Cell):
    """Network with view followed by inplace operations for testing auto_offload."""

    def __init__(self):
        super(ViewInplaceNet, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight2')
        self.fc = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with view followed by inplace operations."""
        x = self.fc(x)  # [batch, 64]
        x = self.relu(x)

        batch_size = x.shape[0]

        # View operation
        y = x.reshape(batch_size, 8, 8)  # [batch, 8, 8]

        # Inplace-like operation on the view (using operations that may modify)
        # Note: InplaceAdd and InplaceMul directly modify input
        y = ops.InplaceAdd(self.weight1[:8, :8], y)  # Inplace add
        y = ops.InplaceMul(self.weight2[:8, :8], y)  # Inplace mul

        # Reshape back
        y = y.reshape(batch_size, 64)  # [batch, 64]

        # Continue
        x = x + y
        x = self.fc(x)
        return x


def test_auto_offload_view_inplace():
    """
    Feature: Test auto_offload with view followed by inplace operations.
    Description: After view, perform inplace-like operation. The tracer should
                 correctly track the modified view tensor.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ViewInplaceNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = ViewInplaceNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_view_inplace passed successfully")


# =============================================================================
# 1.9 跨view操作测试 (多view共享源tensor)
# =============================================================================

class CrossViewNet(nn.Cell):
    """Network with multiple views used by different operators for testing auto_offload."""

    def __init__(self):
        super(CrossViewNet, self).__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with cross view operations."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        batch_size = x.shape[0]

        # Create multiple views from same tensor
        view1 = x.reshape(batch_size, 32, 4)  # View 1
        view2 = x.reshape(batch_size, 16, 8)  # View 2
        view3 = x.transpose(0, 1)  # Transpose view

        # Use each view with different operations
        out1 = self.fc2(view1)  # Uses view1
        out2 = self.fc2(view2)  # Uses view2 (same source as view1)

        # Combine
        x = out1 + out2
        x = self.fc3(x)
        x = self.relu(x)

        return x


def test_auto_offload_cross_view():
    """
    Feature: Test auto_offload with cross view operations.
    Description: Multiple views created from same tensor are used by different operators.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = CrossViewNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = CrossViewNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_cross_view passed successfully")


# =============================================================================
# 1.10 Inplace算子测试 (InplaceAdd, InplaceMul, InplaceIndexAdd)
# =============================================================================

class InplaceOpsNet(nn.Cell):
    """Network with inplace operations for testing auto_offload."""

    def __init__(self):
        super(InplaceOpsNet, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight2')
        self.weight3 = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight3')
        self.fc = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with inplace operations."""
        x = self.fc(x)  # [batch, 64]
        x = self.relu(x)

        # InplaceAdd: x += weight1
        x = ops.InplaceAdd(self.weight1, x)

        # InplaceMul: x *= weight2
        x = ops.InplaceMul(self.weight2, x)

        # Another InplaceAdd
        x = ops.InplaceAdd(self.weight3, x)

        x = self.relu(x)
        return x


class InplaceIndexAddNet(nn.Cell):
    """Network with InplaceIndexAdd operations for testing auto_offload."""

    def __init__(self):
        super(InplaceIndexAddNet, self).__init__()
        self.index = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.int32))
        self.weight = Parameter(Tensor(np.ones((64, 64)).astype(np.float32) * 0.1), name='weight')
        self.fc = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with InplaceIndexAdd operations."""
        x = self.fc(x)  # [batch, 64]
        x = self.relu(x)

        # InplaceIndexAdd: x[:, self.index] += weight[:, :8]
        x = ops.InplaceIndexAdd(self.index, x, self.weight)

        x = self.relu(x)
        return x


def test_auto_offload_inplace_ops():
    """
    Feature: Test auto_offload with inplace operations (InplaceAdd, InplaceMul).
    Description: Inplace operations modify input tensor in place.
                 The tracer should correctly track the modified tensor.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = InplaceOpsNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = InplaceOpsNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_inplace_ops passed successfully")


def test_auto_offload_inplace_index_add():
    """
    Feature: Test auto_offload with InplaceIndexAdd operations.
    Description: InplaceIndexAdd modifies tensor at specified indices.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = InplaceIndexAddNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = InplaceIndexAddNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_inplace_index_add passed successfully")


# =============================================================================
# 1.11 深层嵌套view测试 (多层Reshape/Transpose)
# =============================================================================

class DeepNestedViewNet(nn.Cell):
    """Network with deep nested view operations for testing auto_offload."""

    def __init__(self):
        super(DeepNestedViewNet, self).__init__()
        self.fc1 = nn.Dense(64, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with deep nested view operations."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        batch_size = x.shape[0]

        # Level 1: reshape
        v1 = x.reshape(batch_size, 4, 32)  # [batch, 4, 32]
        # Level 2: transpose
        v2 = v1.transpose(0, 2, 1)  # [batch, 32, 4]
        # Level 3: reshape
        v3 = v2.reshape(batch_size, 8, 4, 4)  # [batch, 8, 4, 4]
        # Level 4: transpose
        v4 = v3.transpose(0, 2, 1, 3)  # [batch, 4, 8, 4]
        # Level 5: reshape back
        v5 = v4.reshape(batch_size, 128)  # [batch, 128]

        # Use deepest level
        x = v5 * 0.5 + v1[:, 0, :] * 0.5

        x = self.fc2(x)
        x = self.relu(x)

        return x


def test_auto_offload_deep_nested_view():
    """
    Feature: Test auto_offload with deep nested view operations (5+ levels).
    Description: Multiple levels of view operations chain from same tensor.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = DeepNestedViewNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = DeepNestedViewNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    warmup_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_deep_nested_view passed successfully")


# =============================================================================
# 完整view/inplace场景测试套件
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    print("=" * 60)
    print("Running view/inplace auto_offload tests...")
    print("=" * 60)

    # 1.1 基础view算子
    test_auto_offload_view_reshape()
    test_auto_offload_view_transpose()

    # 1.2 slice算子
    test_auto_offload_slice()
    test_auto_offload_strided_slice()

    # 1.3 expand类
    test_auto_offload_expand_dims()
    test_auto_offload_expand()
    test_auto_offload_unsqueeze()

    # 1.4 narrow
    test_auto_offload_narrow()

    # 1.5 permute
    test_auto_offload_permute()

    # 1.6 flatten/squeeze
    test_auto_offload_flatten_squeeze()

    # 1.7 链式view
    test_auto_offload_chained_view()

    # 1.8 view+inplace组合
    test_auto_offload_view_inplace()

    # 1.9 跨view操作
    test_auto_offload_cross_view()

    # 1.10 inplace算子
    test_auto_offload_inplace_ops()
    test_auto_offload_inplace_index_add()

    # 1.11 深层嵌套view
    test_auto_offload_deep_nested_view()

    print("=" * 60)
    print("All view/inplace auto_offload tests passed!")
    print("=" * 60)
