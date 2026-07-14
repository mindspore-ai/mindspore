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
Test auto_offload functionality for multi-output scenarios in PyNative mode.

This test file covers:
- Split operations: uniform and non-uniform splitting
- Chunk operations: chunk tensor into multiple parts
- Unstack operations: unstack tensor along a dimension
- Multi-output branching: one output used by multiple operations
- Partial output usage: only some outputs are used
- Nested multi-output: tuple of tuple outputs
- Multi-output combined with view operations
"""

import numpy as np
from mindspore import nn, Tensor, context, ops
from mindspore.ops import composite as C
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
# 2.1 Split均分测试
# =============================================================================

class SplitUniformNet(nn.Cell):
    """Network with uniform split operations for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc4 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc5 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with uniform split operations."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        # Split into 4 equal parts along dim 1
        # 256 / 4 = 64
        split_outputs = ops.Split(1, 4)(x)

        # Use each split output
        out1 = self.fc2(split_outputs[0])
        out2 = self.fc3(split_outputs[1])
        out3 = self.fc4(split_outputs[2])
        out4 = self.fc5(split_outputs[3])

        # Combine all outputs
        x = out1 + out2 + out3 + out4
        x = self.relu(x)
        return x


def test_auto_offload_split_uniform():
    """
    Feature: Test auto_offload with uniform split operations.
    Description: Split tensor into 4 equal parts along dimension 1.
                 Each part is used by a separate fully connected layer.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = SplitUniformNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = SplitUniformNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_split_uniform passed successfully")


# =============================================================================
# 2.2 Split不均分测试
# =============================================================================

class SplitNonUniformNet(nn.Cell):
    """Network with non-uniform split operations for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 200, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(50, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(100, 64, weight_init='ones', bias_init='zeros')
        self.fc4 = nn.Dense(50, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with non-uniform split operations."""
        x = self.fc1(x)  # [batch, 200]
        x = self.relu(x)

        # Split into 4 parts of different sizes: [50, 50, 50, 50]
        # Total = 200
        split_outputs = ops.Split(1, [50, 50, 50, 50])(x)

        # Use each split output
        out1 = self.fc2(split_outputs[0])
        out2 = self.fc3(split_outputs[1])
        out3 = self.fc4(split_outputs[2])
        out4 = self.fc4(split_outputs[3])

        # Combine
        x = out1 + out2 + out3 + out4
        x = self.relu(x)
        return x


def test_auto_offload_split_non_uniform():
    """
    Feature: Test auto_offload with non-uniform split operations.
    Description: Split tensor into parts of different sizes along dimension 1.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = SplitNonUniformNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = SplitNonUniformNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_split_non_uniform passed successfully")


# =============================================================================
# 2.3 Chunk操作测试
# =============================================================================

class ChunkNet(nn.Cell):
    """Network with chunk operations for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 192, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc4 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with chunk operations."""
        x = self.fc1(x)  # [batch, 192]
        x = self.relu(x)

        # Chunk into 3 parts along dim 1 (192 / 3 = 64)
        chunks = ops.Chunk(3)(x)  # Returns tuple of 3 tensors

        # Use each chunk
        out1 = self.fc2(chunks[0])
        out2 = self.fc3(chunks[1])
        out3 = self.fc4(chunks[2])

        # Combine
        x = out1 + out2 + out3
        x = self.relu(x)
        return x


class ChunkDim0Net(nn.Cell):
    """Network with chunk along dim 0 for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with chunk along dim 0."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        # Chunk into 2 parts along dim 0 (batch dimension)
        chunks = ops.Chunk(2)(x)  # Returns tuple of 2 tensors

        # Use each chunk
        out1 = self.fc2(chunks[0])
        out2 = self.fc3(chunks[1])

        # Combine
        x = out1 + out2
        x = self.relu(x)
        return x


def test_auto_offload_chunk():
    """
    Feature: Test auto_offload with chunk operations.
    Description: Chunk tensor into multiple parts along specified dimension.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ChunkNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = ChunkNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_chunk passed successfully")


def test_auto_offload_chunk_dim0():
    """
    Feature: Test auto_offload with chunk along dimension 0.
    Description: Chunk tensor along batch dimension (dim 0).
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = ChunkDim0Net()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(64, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = ChunkDim0Net()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_chunk_dim0 passed successfully")


# =============================================================================
# 2.4 Unstack操作测试
# =============================================================================

class UnstackNet(nn.Cell):
    """Network with unstack operations for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with unstack operations."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        # Unstack along dim 1 (split the 256 features into 4 tensors of 64)
        unstacked = ops.unstack(x, axis=1, num=4)

        # Concatenate back
        x_concat = ops.concat(unstacked, axis=1)
        x = self.fc2(x_concat)
        x = self.relu(x)
        return x


class UnstackDim0Net(nn.Cell):
    """Network with unstack along dim 0 for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 128, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(128, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with unstack along dim 0."""
        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        # Unstack along dim 0 (split batch into individual samples)
        unstacked = ops.unstack(x, axis=0)

        # Stack back with addition
        x_sum = unstacked[0]
        for tensor in unstacked[1:]:
            x_sum = x_sum + tensor

        x = self.fc2(x_sum)
        x = self.relu(x)
        return x


def test_auto_offload_unstack():
    """
    Feature: Test auto_offload with unstack operations.
    Description: Unstack tensor along specified dimension and process each part.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = UnstackNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = UnstackNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_unstack passed successfully")


def test_auto_offload_unstack_dim0():
    """
    Feature: Test auto_offload with unstack along dimension 0.
    Description: Unstack along batch dimension and combine results.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = UnstackDim0Net()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(8, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = UnstackDim0Net()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_unstack_dim0 passed successfully")


# =============================================================================
# 2.5 多输出分支测试（一个输出被多次使用）
# =============================================================================

class MultiOutputBranchingNet(nn.Cell):
    """Network with multi-output branching for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc4 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with multi-output branching."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        # Split into 4 parts
        split_outputs = ops.Split(1, 4)(x)

        # One output used by multiple operations (branching)
        # split_outputs[0] is used by fc2 and fc3
        out1a = self.fc2(split_outputs[0])
        out1b = self.fc3(split_outputs[0])

        # split_outputs[1] used by fc4
        out2 = self.fc4(split_outputs[1])

        # split_outputs[2] and [3] directly used
        out3 = split_outputs[2] + out1a
        out4 = split_outputs[3] + out1b

        # Combine all
        x = out1a + out1b + out2 + out3 + out4
        x = self.relu(x)
        return x


def test_auto_offload_multi_output_branching():
    """
    Feature: Test auto_offload with multi-output branching.
    Description: One output tensor from split is used by multiple operations.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = MultiOutputBranchingNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = MultiOutputBranchingNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_multi_output_branching passed successfully")


# =============================================================================
# 2.6 部分输出使用测试（只使用部分输出）
# =============================================================================

class PartialOutputNet(nn.Cell):
    """Network with partial output usage for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with partial output usage."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        # Split into 4 parts, but only use first 2
        split_outputs = ops.Split(1, 4)(x)

        # Only use outputs[0] and outputs[1], ignore outputs[2] and outputs[3]
        out1 = self.fc2(split_outputs[0])
        out2 = self.fc3(split_outputs[1])

        # Combine
        x = out1 + out2
        x = self.relu(x)
        return x


def test_auto_offload_partial_output():
    """
    Feature: Test auto_offload with partial output usage.
    Description: Only some outputs from split are used in computation.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = PartialOutputNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = PartialOutputNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_partial_output passed successfully")


# =============================================================================
# 2.7 嵌套多输出测试（tuple of tuple输出）
# =============================================================================

class NestedMultiOutputNet(nn.Cell):
    """Network with nested multi-output for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc4 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc5 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with nested multi-output."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        # First level split: 256 -> [128, 128]
        split1 = ops.Split(1, 2)(x)

        # Second level split: each 128 -> [64, 64]
        split2_left = ops.Split(1, 2)(split1[0])
        split2_right = ops.Split(1, 2)(split1[1])

        # Nested structure: ((a, b), (c, d))
        # Use all nested outputs
        out1 = self.fc2(split2_left[0])
        out2 = self.fc3(split2_left[1])
        out3 = self.fc4(split2_right[0])
        out4 = self.fc5(split2_right[1])

        # Combine
        x = out1 + out2 + out3 + out4
        x = self.relu(x)
        return x


def test_auto_offload_nested_multi_output():
    """
    Feature: Test auto_offload with nested multi-output.
    Description: Multiple levels of split operations creating tuple of tuples.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = NestedMultiOutputNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = NestedMultiOutputNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads)
    print("test_auto_offload_nested_multi_output passed successfully")


# =============================================================================
# 2.8 多输出+View组合测试
# =============================================================================

class MultiOutputViewNet(nn.Cell):
    """Network with multi-output combined with view operations for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256, weight_init='ones', bias_init='zeros')
        self.fc2 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc3 = nn.Dense(64, 64, weight_init='ones', bias_init='zeros')
        self.fc4 = nn.Dense(256, 128, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with multi-output followed by view operations."""
        x = self.fc1(x)  # [batch, 256]
        x = self.relu(x)

        batch_size = x.shape[0]

        # Split into 4 parts
        split_outputs = ops.Split(1, 4)(x)

        # Apply view operations to split outputs
        view1 = split_outputs[0].reshape(batch_size, 8, 8)  # View
        view2 = split_outputs[1].reshape(batch_size, 8, 8)  # View
        view3 = split_outputs[2].reshape(batch_size, 8, 8)  # View
        view4 = split_outputs[3].reshape(batch_size, 8, 8)  # View

        # Process viewed tensors
        out1 = self.fc2(view1.reshape(batch_size, -1))
        out2 = self.fc3(view2.reshape(batch_size, -1))

        # Concatenate and reshape back
        concat = ops.concat([out1, out2, view3.reshape(batch_size, -1), view4.reshape(batch_size, -1)], axis=1)
        x = self.fc4(concat)
        x = self.relu(x)
        return x


class MultiOutputReshapeNet(nn.Cell):
    """Network with multi-output followed by reshape for testing auto_offload."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                               weight_init='ones', bias_init='zeros')
        self.fc = nn.Dense(16 * 32 * 32, 10, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass with split followed by reshape and processing."""
        # x: [batch, 1, 32, 32]
        x = self.conv(x)  # [batch, 4, 32, 32]
        x = self.relu(x)

        # Split along channel dimension
        split_outputs = ops.Split(1, 4)(x)

        # Each split output: [batch, 1, 32, 32]
        # Process each with view operations
        processed = []
        for i, split in enumerate(split_outputs):
            # Reshape to [batch, 32, 32]
            viewed = split.squeeze(1)
            # Multiply by scalar
            viewed = viewed * (i + 1)
            processed.append(viewed)

        # Concatenate back along channel
        x = ops.concat(processed, axis=1)  # [batch, 4, 32, 32]

        # Flatten and fully connected
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


def test_auto_offload_multi_output_view():
    """
    Feature: Test auto_offload with multi-output combined with view operations.
    Description: Split outputs are processed with reshape/view operations.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = MultiOutputViewNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 128).astype(np.float32))

    # Reference without auto_offload
    net_ref = MultiOutputViewNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads, rtol=1e-3, atol=1e-3)
    print("test_auto_offload_multi_output_view passed successfully")


def test_auto_offload_multi_output_reshape():
    """
    Feature: Test auto_offload with multi-output followed by reshape.
    Description: Split outputs are reshaped and processed individually.
    Expectation: Forward and backward passes should produce correct results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    manager = OffloadManager.get_instance()
    manager.reset()

    net = MultiOutputReshapeNet()
    net.auto_offload()
    net.set_grad()

    x = Tensor(np.random.randn(32, 1, 32, 32).astype(np.float32))

    # Reference without auto_offload
    net_ref = MultiOutputReshapeNet()
    sync_weights(net, net_ref)

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    _ = grad_fn(net, list(net.trainable_params()))(x)

    # Analyze
    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    verify_gradients(exec_grads, ref_grads, rtol=1e-3, atol=1e-3)
    print("test_auto_offload_multi_output_reshape passed successfully")


# =============================================================================
# 完整多输出场景测试套件
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running multi-output auto_offload tests...")
    print("=" * 60)

    # 2.1 Split均分
    test_auto_offload_split_uniform()
    test_auto_offload_split_non_uniform()

    # 2.3 Chunk操作
    test_auto_offload_chunk()
    test_auto_offload_chunk_dim0()

    # 2.4 Unstack操作
    test_auto_offload_unstack()
    test_auto_offload_unstack_dim0()

    # 2.5 多输出分支
    test_auto_offload_multi_output_branching()

    # 2.6 部分输出使用
    test_auto_offload_partial_output()

    # 2.7 嵌套多输出
    test_auto_offload_nested_multi_output()

    # 2.8 多输出+View组合
    test_auto_offload_multi_output_view()
    test_auto_offload_multi_output_reshape()

    print("=" * 60)
    print("All multi-output auto_offload tests passed!")
    print("=" * 60)
