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
"""Test hyper_offload functionality in PyNative mode."""

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, context, ops
from mindspore._c_expression import OffloadManager


class SimpleNet(nn.Cell):
    """A simple network for testing hyper offload."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256)
        self.fc2 = nn.Dense(256, 128)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestHyperOffload:
    """Test cases for hyper offload functionality."""

    def setup_method(self):
        """Set up test environment."""
        context.set_context(mode=context.PYNATIVE_MODE)
        # Reset offload manager before each test
        manager = OffloadManager.get_instance()
        manager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        manager = OffloadManager.get_instance()
        manager.reset()

    def test_offload_manager_singleton(self):
        """Test that OffloadManager is a singleton."""
        manager1 = OffloadManager.get_instance()
        manager2 = OffloadManager.get_instance()
        assert manager1 is manager2, "OffloadManager should be a singleton"

    def test_offload_context_creation(self):
        """Test creating OffloadContext."""
        manager = OffloadManager.get_instance()

        # Initially no contexts
        assert manager.context_count() == 0, "Should start with 0 contexts"
        assert not OffloadManager.is_offload_ready(), "Should not be ready initially"

        # Create a context (warm-up phase) with key 0
        ctx = manager.get_or_create_offload_ctx(0)
        assert ctx is not None, "Context should be created"
        assert manager.context_count() == 1, "Should have 1 context"

    def test_offload_warmup_and_analyse(self):
        """Test warm-up phase and analyse."""
        manager = OffloadManager.get_instance()

        # Create context and start tracing with key 0
        ctx = manager.get_or_create_offload_ctx(0)
        ctx.start_offload()
        assert ctx.is_active(), "Context should be active after start_offload"

        # End tracing
        ctx.end_offload()
        assert not ctx.is_active(), "Context should not be active after end_offload"

        # Call analyse to mark warm-up as complete
        manager.analyse()
        assert OffloadManager.is_offload_ready(), "Should be ready after analyse"

    def test_offload_with_simple_forward(self):
        """Test offload tracing with a simple forward pass."""
        manager = OffloadManager.get_instance()
        net = SimpleNet()
        offload_key = id(net)

        # Create input
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        # Step 1: Warm-up phase
        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()

        # Call analyse
        manager.analyse()
        assert OffloadManager.is_offload_ready(), "Should be ready after analyse"

        # Step 2: Replay phase - should reuse the same context
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()
        _ = net(x)
        ctx2.end_offload()

        # Verify context count
        assert manager.context_count() == 1, "Should still have 1 context"

    def test_offload_with_forward_backward(self):
        """Test offload tracing with forward and backward pass."""
        manager = OffloadManager.get_instance()
        net = SimpleNet()
        offload_key = id(net)

        # Create input
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        # Define loss function
        def forward_fn(x):
            output = net(x)
            loss = ops.ReduceSum()(output)
            return loss

        grad_fn = ms.grad(forward_fn)

        # Step 1: Warm-up phase
        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = grad_fn(x)
        ctx.end_offload()

        # Call analyse
        manager.analyse()
        assert OffloadManager.is_offload_ready(), "Should be ready after analyse"

        # Step 2: Replay phase
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()
        _ = grad_fn(x)
        ctx2.end_offload()

        print("Forward-backward test passed!")

    def test_multiple_contexts(self):
        """Test multiple OffloadContext creation during warm-up."""
        manager = OffloadManager.get_instance()

        # Create multiple contexts during warm-up with different keys
        ctx1 = manager.get_or_create_offload_ctx(100)
        ctx1.start_offload()
        ctx1.end_offload()

        ctx2 = manager.get_or_create_offload_ctx(200)
        ctx2.start_offload()
        ctx2.end_offload()

        assert manager.context_count() == 2, "Should have 2 contexts"

        # Call analyse
        manager.analyse()
        assert OffloadManager.is_offload_ready(), "Should be ready after analyse"

        # Replay phase - should return existing contexts by key
        ctx3 = manager.get_or_create_offload_ctx(100)
        ctx3.start_offload()
        ctx3.end_offload()

        ctx4 = manager.get_or_create_offload_ctx(200)
        ctx4.start_offload()
        ctx4.end_offload()

        # Should still have 2 contexts
        assert manager.context_count() == 2, "Should still have 2 contexts after replay"
        # Verify same context is returned for same key
        assert ctx1 is ctx3, "Same key should return same context"
        assert ctx2 is ctx4, "Same key should return same context"

    def test_reset(self):
        """Test resetting OffloadManager."""
        manager = OffloadManager.get_instance()

        # Create context and analyse with key 0
        ctx = manager.get_or_create_offload_ctx(0)
        ctx.start_offload()
        ctx.end_offload()
        manager.analyse()

        assert OffloadManager.is_offload_ready(), "Should be ready"
        assert manager.context_count() == 1, "Should have 1 context"

        # Reset
        manager.reset()

        assert not OffloadManager.is_offload_ready(), "Should not be ready after reset"
        assert manager.context_count() == 0, "Should have 0 contexts after reset"


if __name__ == "__main__":
    # Run basic tests
    test = TestHyperOffload()
    test.setup_method()

    print("Testing offload manager singleton...")
    test.test_offload_manager_singleton()
    print("PASSED")

    test.setup_method()
    print("Testing offload context creation...")
    test.test_offload_context_creation()
    print("PASSED")

    test.setup_method()
    print("Testing offload warmup and analyse...")
    test.test_offload_warmup_and_analyse()
    print("PASSED")

    test.setup_method()
    print("Testing offload with simple forward...")
    test.test_offload_with_simple_forward()
    print("PASSED")

    test.setup_method()
    print("Testing offload with forward backward...")
    test.test_offload_with_forward_backward()
    print("PASSED")

    test.setup_method()
    print("Testing multiple contexts...")
    test.test_multiple_contexts()
    print("PASSED")

    test.setup_method()
    print("Testing reset...")
    test.test_reset()
    print("PASSED")

    print("\n=== All tests passed! ===")
