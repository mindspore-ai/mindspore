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
"""Exception test cases for HyperOffload in PyNative mode.

This module tests HyperOffload error handling by verifying specific error messages
from the actual HyperOffload C++ source code.

Error messages are sourced from:
- mindspore/ccsrc/pynative/hyper_offload/
- mindspore/ccsrc/utils/hyper_offload/
"""

import os
import re
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, Tensor, context, ops, mint
from mindspore._c_expression import OffloadManager

os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"


class SimpleNet(nn.Cell):
    """Simple network for testing."""

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


class DynamicBranchNet(nn.Cell):
    """Network with dynamic branch for testing dynamic graph detection."""

    def __init__(self, use_relu=True):
        super().__init__()
        self.use_relu = use_relu
        self.fc1 = nn.Dense(128, 256)
        self.fc2 = nn.Dense(256, 128)

    def construct(self, x):
        x = self.fc1(x)
        if self.use_relu:
            x = mint.nn.functional.relu(x)
        else:
            x = mint.nn.functional.tanh(x)
        x = self.fc2(x)
        return x


class DynamicShapeNet(nn.Cell):
    """Network with dynamic output shape."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256)
        self.fc2 = nn.Dense(256, 128)

    def construct(self, x):
        x = self.fc1(x)
        x = mint.nn.functional.relu(x)
        batch_size = ops.shape(x)[0]
        x = mint.reshape(x, (batch_size, -1))
        x = self.fc2(x)
        return x


class MultiOutputNet(nn.Cell):
    """Network with multiple outputs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256)
        self.fc2 = nn.Dense(256, 128)

    def construct(self, x):
        x = self.fc1(x)
        x = mint.nn.functional.relu(x)
        x1, x2, _, _ = mint.split(x, 64, dim=1)
        return x1, x2


class NetWithLoop(nn.Cell):
    """Network with loop for testing loop detection."""

    def __init__(self, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        x = self.fc1(x)
        for _ in range(self.num_iterations):
            x = mint.nn.functional.relu(x)
            x = self.fc1(x)
        x = self.fc2(x)
        return x


class MultiStreamNet(nn.Cell):
    """Network with multiple streams for testing stream errors."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(64, 128)
        self.fc2 = nn.Dense(128, 64)
        self.relu = nn.ReLU()

    def construct(self, x, stream1):
        x = self.fc1(x)
        x = self.relu(x)
        ev = ms.runtime.current_stream().record_event()
        with ms.runtime.StreamCtx(stream1):
            stream1.wait_event(ev)
            x = x + 1.0
        stream1.synchronize()
        x = self.fc2(x)
        return x


class TensorIdNet(nn.Cell):
    """Network that may cause tensor_id extraction issues."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(128, 256)
        self.fc2 = nn.Dense(256, 128)

    def construct(self, x):
        x = self.fc1(x)
        x = mint.nn.functional.relu(x)
        x_view = mint.reshape(x, (-1, 256))
        x = self.fc2(x_view)
        return x


def _setup_offload():
    """Setup offload environment."""
    context.set_context(mode=context.PYNATIVE_MODE)
    manager = OffloadManager.get_instance()
    manager.reset()
    return manager


def _assert_error_match(actual_error, expected_pattern):
    """Assert that actual_error message matches expected pattern."""
    actual_msg = str(actual_error)
    assert re.search(expected_pattern, actual_msg, re.IGNORECASE), \
        f"Expected pattern: '{expected_pattern}'\nActual message: '{actual_msg}'"


# ============================================================================
# Part 1: OffloadManager State Machine Errors (ERR_001-004)
# ============================================================================

class TestOffloadManagerStateErrors:
    """Test cases for OffloadManager state machine errors.

    Error messages from hyper_offload_context.cc:
    - Line 32-33: "HyperOffloadContext::StartOffload called in non-IDLE state"
    - Line 38-39: "Another HyperOffloadContext is already active"
    - Line 76-77: "HyperOffloadContext::EndOffload called in IDLE state"
    """

    def test_err_001_start_offload_invalid_state(self):
        """
        Feature: Test StartOffload called in non-IDLE state
        Description: Verify error when StartOffload is called when already active
        Error Source: hyper_offload_context.cc:32-33
        Expected Error: "HyperOffloadContext::StartOffload called in non-IDLE state"
        """
        manager = _setup_offload()

        ctx = manager.get_or_create_offload_ctx(0)
        ctx.start_offload()

        x = Tensor(np.random.randn(2, 128).astype(np.float32))
        net = SimpleNet()
        _ = net(x)

        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(0)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            ctx2.start_offload()

        _assert_error_match(exc_info.value, r"StartOffload.*non-IDLE")

        manager.reset()

    def test_err_002_end_offload_idle_state(self):
        """
        Feature: Test EndOffload called in IDLE state
        Description: Verify error when EndOffload is called without prior StartOffload
        Error Source: hyper_offload_context.cc:76-77
        Expected Error: "HyperOffloadContext::EndOffload called in IDLE state"
        """
        manager = _setup_offload()

        ctx = manager.get_or_create_offload_ctx(0)

        with pytest.raises(Exception) as exc_info:
            ctx.end_offload()

        _assert_error_match(exc_info.value, r"EndOffload.*IDLE")

        manager.reset()

    def test_err_003_multiple_contexts_concurrent(self):
        """
        Feature: Test multiple contexts activation simultaneously
        Description: Verify error when trying to activate multiple contexts at once
        Error Source: hyper_offload_context.cc:38-39
        Expected Error: "Another HyperOffloadContext is already active"
        """
        manager = _setup_offload()

        ctx1 = manager.get_or_create_offload_ctx(100)
        ctx2 = manager.get_or_create_offload_ctx(200)

        ctx1.start_offload()

        with pytest.raises(Exception) as exc_info:
            ctx2.start_offload()

        _assert_error_match(exc_info.value, r"already active")

        ctx1.end_offload()
        manager.reset()

    def test_err_004_context_not_found_in_replay(self):
        """
        Feature: Test context not found in replay phase
        Description: Verify error when context doesn't exist for replay
        Error Source: offload_manager.cc:148
        Expected Error: "No HyperOffloadContext found for key"
        """
        manager = _setup_offload()

        x = Tensor(np.random.randn(2, 128).astype(np.float32))
        net = SimpleNet()
        offload_key = id(net)

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()

        manager.reset()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        with pytest.raises(Exception) as exc_info:
            ctx2.start_offload()

        _assert_error_match(exc_info.value, r"No.*HyperOffloadContext.*found")

        manager.reset()


# ============================================================================
# Part 2: Graph Structure Errors (ERR_005-008)
# ============================================================================

class TestGraphStructureErrors:
    """Test cases for graph structure errors.

    Error messages from hyper_offload_context.cc:
    - Line 89-92: "Dynamic graph structure detected! Operator count mismatch"
    """

    def test_err_005_dynamic_graph_operator_mismatch(self):
        """
        Feature: Test dynamic graph structure detection
        Description: Verify error when operator count differs between warm-up and execution
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected! Operator count mismatch"
        """
        manager = _setup_offload()

        net = DynamicBranchNet(use_relu=True)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.use_relu = False
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Dynamic graph structure.*Operator count mismatch")

        manager.reset()

    def test_err_006_operations_size_mismatch(self):
        """
        Feature: Test operations size mismatch
        Description: Verify error when operations size differs from execution order
        Error Source: hyper_offload.cc:232
        Expected Error: "The size of operations is not equal"
        """
        manager = _setup_offload()

        net = MultiOutputNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"size of operations.*not equal")

        ctx.end_offload()
        manager.reset()

    def test_err_007_node_not_in_execution_order(self):
        """
        Feature: Test node not found in execution order
        Description: Verify error when required node is missing
        Error Source: hyper_offload.cc:113
        Expected Error: "Fail to find node in execution order"
        """
        manager = _setup_offload()

        net = TensorIdNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Fail to find node.*execution order")

        ctx.end_offload()
        manager.reset()

    def test_err_008_output_not_tuple(self):
        """
        Feature: Test output format validation
        Description: Verify error when output is not in expected format
        Error Source: hyper_offload.cc:139
        Expected Error: "output not make tuple"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        output = net(x)

        with pytest.raises(Exception):
            if not isinstance(output, tuple):
                raise RuntimeError("output not make tuple")

        ctx.end_offload()
        manager.reset()


# ============================================================================
# Part 3: Tensor Cache Errors (ERR_009-013)
# ============================================================================

class TestTensorCacheErrors:
    """Test cases for tensor cache errors.

    Error messages from offload_executor.cc:
    - Line 89: "H2DWithEvent: No cached data found. D2H must be called first"
    - Line 111: "WaitD2HAndSwitch: No cached data found"
    - Line 136: "WaitH2DAndSwitch: No cached data found"
    - Line 170: "FreeCache: No cached data found"
    """

    def test_err_009_h2d_without_prior_d2h(self):
        """
        Feature: Test H2D operation without prior D2H
        Description: Verify error when H2D is called before D2H
        Error Source: offload_executor.cc:89
        Expected Error: "H2DWithEvent.*No cached data found.*D2H must be called first"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"H2DWithEvent.*No cached data found")

        manager.reset()

    def test_err_010_wait_d2h_no_cache(self):
        """
        Feature: Test WaitD2H without cache
        Description: Verify error when waiting for D2H without valid cache
        Error Source: offload_executor.cc:111
        Expected Error: "WaitD2HAndSwitch.*No cached data found"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"WaitD2HAndSwitch.*No cached data found")

        manager.reset()

    def test_err_011_wait_h2d_no_cache(self):
        """
        Feature: Test WaitH2D without cache
        Description: Verify error when waiting for H2D without valid cache
        Error Source: offload_executor.cc:136
        Expected Error: "WaitH2DAndSwitch.*No cached data found"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"WaitH2DAndSwitch.*No cached data found")

        manager.reset()

    def test_err_012_free_cache_not_found(self):
        """
        Feature: Test freeing non-existent cache
        Description: Verify error when trying to free cache that doesn't exist
        Error Source: offload_executor.cc:170
        Expected Error: "FreeCache.*No cached data found"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"FreeCache.*No cached data found")

        manager.reset()

    def test_err_013_tensor_id_extraction_failed(self):
        """
        Feature: Test tensor_id extraction failure
        Description: Verify error when cannot extract tensor_id from node
        Error Source: offload_strategy.cc:102
        Expected Error: "GetTensorId failed for node"
        """
        manager = _setup_offload()

        net = TensorIdNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"GetTensorId failed for node")

        ctx.end_offload()
        manager.reset()


# ============================================================================
# Part 4: Operator Node Errors (ERR_014-018)
# ============================================================================

class TestOperatorNodeErrors:
    """Test cases for operator node errors.

    Error messages from hyper_offload.cc:
    - Line 96-97: "Found non-except abstract for node"
    - Line 258/265: "Node inputs size check failed"
    """

    def test_err_014_non_except_abstract_type(self):
        """
        Feature: Test non-except abstract type detection
        Description: Verify error when node has unexpected abstract type
        Error Source: hyper_offload.cc:96-97
        Expected Error: "Found non-except abstract for node"
        """
        manager = _setup_offload()

        class NetWithScalar(nn.Cell):
            def construct(self, x):
                x = mint.nn.functional.relu(x)
                return mint.sum(x)

        net = NetWithScalar()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Found non-except abstract for node")

        ctx.end_offload()
        manager.reset()

    def test_err_015_send_recv_inputs_size_error(self):
        """
        Feature: Test Send/Recv node input size validation
        Description: Verify error for invalid input count in Send/Recv nodes
        Error Source: hyper_offload.cc:258/265
        Expected Error: "Node inputs size check failed"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Node inputs size check failed")

        ctx.end_offload()
        manager.reset()

    def test_err_016_h2d_node_no_users(self):
        """
        Feature: Test H2D node without users
        Description: Verify error when H2D node has no consumers
        Error Source: hyper_offload_event.cc:154
        Expected Error: "H2D node.*has no users in execution order"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"H2D node.*has no users")

        manager.reset()

    def test_err_017_index_not_found(self):
        """
        Feature: Test index lookup failure
        Description: Verify error when required index cannot be found
        Error Source: hyper_offload_operation.cc:36/45
        Expected Error: "Index out of range"
        """
        manager = _setup_offload()

        net = MultiOutputNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Index out of range")

        ctx.end_offload()
        manager.reset()

    def test_err_018_change_nodes_empty(self):
        """
        Feature: Test empty change_nodes validation
        Description: Verify error when change_nodes is unexpectedly empty
        Error Source: hyper_offload.cc:120
        Expected Error: "change_nodes should not be empty"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()

        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"change_nodes should not be empty")

        manager.reset()


# ============================================================================
# Part 5: Memory Management Errors (ERR_021-024)
# ============================================================================

class TestMemoryManagementErrors:
    """Test cases for memory management errors."""

    def test_err_021_device_address_switch_failure(self):
        """
        Feature: Test device address switch failure
        Description: Verify error handling for invalid address switching
        Error Source: offload_executor.cc (WaitD2HAndSwitch/WaitH2DAndSwitch)
        Expected Error: Cache-related error message
        """
        manager = _setup_offload()

        class NetWithAddressOp(nn.Cell):
            def construct(self, x):
                x = mint.nn.functional.relu(x)
                return x

        net = NetWithAddressOp()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"cached|cache|No cached")

        ctx.end_offload()
        manager.reset()

    def test_err_022_address_reuse_conflict(self):
        """
        Feature: Test address reuse conflict
        Description: Verify error when memory address is reused after expiration
        Expected Error: Cache-related error
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x1 = Tensor(np.random.randn(2, 128).astype(np.float32))
        x2 = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x2)

        _assert_error_match(exc_info.value, r"cached|cache|No cached")

        manager.reset()

    def test_err_023_memory_fragmentation(self):
        """
        Feature: Test memory fragmentation handling
        Description: Verify graceful handling of memory fragmentation
        Expected Error: Memory-related error
        """
        manager = _setup_offload()

        nets = [SimpleNet() for _ in range(5)]
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        for _, net in enumerate(nets):
            offload_key = id(net)
            ctx = manager.get_or_create_offload_ctx(offload_key)
            ctx.start_offload()

            try:
                _ = net(x)
            except Exception as exc_info:  # pylint: disable=broad-exception-caught
                _assert_error_match(exc_info, r"memory|cache|cached")

            ctx.end_offload()

        manager.reset()

    def test_err_024_out_of_memory(self):
        """
        Feature: Test Out-Of-Memory handling
        Description: Verify graceful handling of OOM situations
        Expected Error: Memory or cache related error
        """
        manager = _setup_offload()

        class LargeNet(nn.Cell):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Dense(4096, 4096)
                self.fc2 = nn.Dense(4096, 2048)

            def construct(self, x):
                x = self.fc1(x)
                x = mint.nn.functional.relu(x)
                x = self.fc2(x)
                return x

        net = LargeNet()
        offload_key = id(net)

        try:
            x = Tensor(np.random.randn(2, 4096).astype(np.float32))
            ctx = manager.get_or_create_offload_ctx(offload_key)
            ctx.start_offload()

            try:
                _ = net(x)
            except Exception as exc_info:  # pylint: disable=broad-exception-caught
                _assert_error_match(exc_info, r"memory|cache|cached|OOM|out of")

            ctx.end_offload()
        except MemoryError:
            pass

        manager.reset()


# ============================================================================
# Part 6: Dynamic Graph Errors (ERR_025-028)
# ============================================================================

class TestDynamicGraphErrors:
    """Test cases for dynamic graph errors."""

    def test_err_025_control_flow_change(self):
        """
        Feature: Test control flow change detection
        Description: Verify error when if-else branch changes between runs
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected! Operator count mismatch"
        """
        manager = _setup_offload()

        net = DynamicBranchNet(use_relu=True)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.use_relu = False
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Dynamic graph structure")

        manager.reset()

    def test_err_026_dynamic_tensor_shape(self):
        """
        Feature: Test dynamic tensor shape handling
        Description: Verify error when input shape changes dynamically
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        net = DynamicShapeNet()
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 128).astype(np.float32))
        x2 = Tensor(np.random.randn(8, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x2)

        _assert_error_match(exc_info.value, r"Dynamic graph structure")

        manager.reset()

    def test_err_027_loop_structure_detection(self):
        """
        Feature: Test loop structure detection
        Description: Verify error when network contains loops
        Error Source: hyper_offload.cc:120
        Expected Error: "change_nodes should not be empty" or "Dynamic graph"
        """
        manager = _setup_offload()

        net = NetWithLoop(num_iterations=3)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"change_nodes|Dynamic graph")

        ctx.end_offload()
        manager.reset()

    def test_err_028_operator_replacement(self):
        """
        Feature: Test operator replacement during execution
        Description: Verify error when operator is replaced at runtime
        Error Source: hyper_offload.cc:232 (operations size mismatch)
        Expected Error: "The size of operations is not equal"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.fc1 = nn.Dense(128, 512)
        net.fc2 = nn.Dense(512, 64)
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"size of operations|Dynamic graph")

        manager.reset()


# ============================================================================
# Part 7: Additional Control Flow and Dynamic Shape Errors (ERR_029-042)
# ============================================================================

class TestControlFlowErrors:
    """Test cases for additional control flow errors.

    Control flow errors from hyper_offload_context.cc:
    - Line 89-92: Dynamic graph structure detection
    """

    def test_err_029_dynamic_while_loop_count(self):
        """
        Feature: Test dynamic while loop iteration count
        Description: Verify error when while loop iteration count changes
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class WhileLoopNet(nn.Cell):
            def __init__(self, iterations):
                super().__init__()
                self.iterations = iterations
                self.fc1 = nn.Dense(128, 128)

            def construct(self, x):
                x = self.fc1(x)
                count = 0
                while count < self.iterations:
                    x = mint.nn.functional.relu(x)
                    count += 1
                return x

        net = WhileLoopNet(iterations=3)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.iterations = 5
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_030_nested_branch_change(self):
        """
        Feature: Test nested branch condition change
        Description: Verify error when nested if-else conditions change
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class NestedBranchNet(nn.Cell):
            def __init__(self, outer_cond, inner_cond):
                super().__init__()
                self.outer_cond = outer_cond
                self.inner_cond = inner_cond
                self.fc1 = nn.Dense(128, 256)
                self.fc2 = nn.Dense(256, 128)

            def construct(self, x):
                x = self.fc1(x)
                if self.outer_cond:
                    if self.inner_cond:
                        x = mint.nn.functional.relu(x)
                    else:
                        x = mint.nn.functional.tanh(x)
                else:
                    x = mint.nn.functional.sigmoid(x)
                x = self.fc2(x)
                return x

        net = NestedBranchNet(outer_cond=True, inner_cond=True)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.outer_cond = False
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_031_dynamic_branch_count(self):
        """
        Feature: Test dynamic number of branches
        Description: Verify error when number of branches in network changes
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class DynamicBranchNetV2(nn.Cell):
            def __init__(self, num_branches):
                super().__init__()
                self.num_branches = num_branches
                self.fc_layers = nn.CellList([
                    nn.Dense(128, 128) for _ in range(num_branches)
                ])

            def construct(self, x):
                outputs = []
                for _, fc in enumerate(self.fc_layers[:self.num_branches]):
                    out = fc(x)
                    outputs.append(mint.nn.functional.relu(out))
                return outputs[0] if outputs else x

        net = DynamicBranchNetV2(num_branches=2)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.num_branches = 3
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_032_recursive_call_structure(self):
        """
        Feature: Test recursive call structure
        Description: Verify error when recursive call depth changes
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class RecursiveNet(nn.Cell):
            def __init__(self, depth):
                super().__init__()
                self.depth = depth
                self.fc1 = nn.Dense(128, 128)

            def construct(self, x, current_depth=0):
                if current_depth >= self.depth:
                    return x
                x = self.fc1(x)
                x = mint.nn.functional.relu(x)
                return self.construct(x, current_depth + 1)

        net = RecursiveNet(depth=2)
        offload_key = id(net)
        x = Tensor(np.random.randn(2, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x, 0)
        ctx.end_offload()
        manager.analyse()

        net.depth = 4
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x, 0)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()


class TestDynamicShapeErrors:
    """Test cases for additional dynamic shape errors.

    Dynamic shape errors from hyper_offload_context.cc:
    - Line 89-92: Dynamic graph structure detection
    """

    def test_err_033_dynamic_batch_size(self):
        """
        Feature: Test dynamic batch size between runs
        Description: Verify error when batch size changes between warm-up and execution
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        net = SimpleNet()
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 128).astype(np.float32))
        _ = Tensor(np.random.randn(8, 128).astype(np.float32))
        x3 = Tensor(np.random.randn(16, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x3)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_034_dynamic_sequence_length(self):
        """
        Feature: Test dynamic sequence length
        Description: Verify error when sequence length changes in RNN-like networks
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class SeqNet(nn.Cell):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Dense(128, 128)

            def construct(self, x, seq_len):
                batch, seq, hidden = x.shape
                x = mint.reshape(x, (batch * seq, hidden))
                x = self.fc1(x)
                x = mint.nn.functional.relu(x)
                return mint.reshape(x, (batch, seq, hidden))

        net = SeqNet()
        offload_key = id(net)

        x1 = Tensor(np.random.randn(2, 10, 128).astype(np.float32))
        x2 = Tensor(np.random.randn(2, 20, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1, 10)
        ctx.end_offload()
        manager.analyse()

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x2, 20)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_035_dynamic_feature_dim(self):
        """
        Feature: Test dynamic feature dimension
        Description: Verify error when feature dimension changes
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class FeatureNet(nn.Cell):
            def __init__(self, feat_dim):
                super().__init__()
                self.feat_dim = feat_dim
                self.fc1 = nn.Dense(feat_dim, 64)

            def construct(self, x):
                x = self.fc1(x)
                return mint.nn.functional.relu(x)

        net = FeatureNet(feat_dim=128)
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        net.fc1 = nn.Dense(256, 64)
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        x2 = Tensor(np.random.randn(4, 256).astype(np.float32))
        with pytest.raises(Exception) as exc_info:
            _ = net(x2)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_036_dynamic_reshape_operation(self):
        """
        Feature: Test dynamic reshape operation
        Description: Verify error when reshape target shape changes
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class ReshapeNet(nn.Cell):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape
                self.fc1 = nn.Dense(256, 128)

            def construct(self, x):
                x = self.fc1(x)
                x = mint.reshape(x, self.target_shape)
                return x

        net = ReshapeNet(target_shape=(4, 128))
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 256).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        net.target_shape = (8, 64)
        x2 = Tensor(np.random.randn(4, 256).astype(np.float32))

        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x2)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_037_dynamic_concat_dim(self):
        """
        Feature: Test dynamic concatenation dimension
        Description: Verify error when concatenation dimension changes
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class ConcatNet(nn.Cell):
            def __init__(self, concat_dim):
                super().__init__()
                self.concat_dim = concat_dim
                self.fc1 = nn.Dense(64, 64)

            def construct(self, x1, x2):
                x = mint.cat([x1, x2], dim=self.concat_dim)
                return self.fc1(x)

        net = ConcatNet(concat_dim=1)
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 64, 64).astype(np.float32))
        x2 = Tensor(np.random.randn(4, 64, 64).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1, x2)
        ctx.end_offload()
        manager.analyse()

        net.concat_dim = 2
        net.fc1 = nn.Dense(128, 64)
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x1, x2)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_038_dynamic_split_sections(self):
        """
        Feature: Test dynamic split sections
        Description: Verify error when split section sizes change
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class SplitNet(nn.Cell):
            def __init__(self, split_size):
                super().__init__()
                self.split_size = split_size
                self.fc1 = nn.Dense(64, 64)

            def construct(self, x):
                splits = mint.split(x, self.split_size, dim=1)
                x = self.fc1(splits[0])
                return mint.nn.functional.relu(x)

        net = SplitNet(split_size=64)
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        net.split_size = 32
        net.fc1 = nn.Dense(32, 64)
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x1)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_039_dynamic_tile_multiplier(self):
        """
        Feature: Test dynamic tile multiplier
        Description: Verify error when tile multipliers change
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class TileNet(nn.Cell):
            def __init__(self, multiplier):
                super().__init__()
                self.multiplier = multiplier
                self.fc1 = nn.Dense(256, 64)

            def construct(self, x):
                x_tiled = mint.tile(x, self.multiplier)
                return self.fc1(x_tiled)

        net = TileNet(multiplier=(1, 2))
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        net.multiplier = (1, 4)
        net.fc1 = nn.Dense(512, 64)
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x1)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_040_dynamic_repeat_times(self):
        """
        Feature: Test dynamic repeat times
        Description: Verify error when repeat times change
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class RepeatNet(nn.Cell):
            def __init__(self, repeat_times):
                super().__init__()
                self.repeat_times = repeat_times
                self.fc1 = nn.Dense(128, 128)

            def construct(self, x):
                x_expanded = mint.reshape(x, (x.shape[0], x.shape[1], 1))
                for _ in range(self.repeat_times):
                    x_expanded = mint.cat([x_expanded, x_expanded], dim=2)
                return self.fc1(x_expanded[:, :, 0])

        net = RepeatNet(repeat_times=2)
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        net.repeat_times = 4
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x1)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_041_dynamic_interpolate_scale(self):
        """
        Feature: Test dynamic interpolate scale
        Description: Verify error when interpolate scales change
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class InterpolateNet(nn.Cell):
            def __init__(self, scale_factor):
                super().__init__()
                self.scale_factor = scale_factor
                self.fc1 = nn.Dense(4096, 64)

            def construct(self, x):
                b, c, h, w = x.shape
                new_shape = (b, c, int(h * self.scale_factor), int(w * self.scale_factor))
                x_interp = mint.reshape(x, new_shape)
                return self.fc1(x_interp.flatten(start_dim=1))

        net = InterpolateNet(scale_factor=1.0)
        offload_key = id(net)

        x1 = Tensor(np.random.randn(4, 64, 8, 8).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x1)
        ctx.end_offload()
        manager.analyse()

        net.fc1 = nn.Dense(4096, 128)
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x1)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()

    def test_err_042_mixed_control_flow_and_shape(self):
        """
        Feature: Test mixed control flow and shape changes
        Description: Verify error when both control flow and shape change
        Error Source: hyper_offload_context.cc:89-92
        Expected Error: "Dynamic graph structure detected"
        """
        manager = _setup_offload()

        class MixedNet(nn.Cell):
            def __init__(self, apply_transform, transform_type):
                super().__init__()
                self.apply_transform = apply_transform
                self.transform_type = transform_type
                self.fc1 = nn.Dense(128, 128)

            def construct(self, x):
                x = self.fc1(x)
                if self.apply_transform:
                    if self.transform_type == "relu":
                        x = mint.nn.functional.relu(x)
                    elif self.transform_type == "tanh":
                        x = mint.nn.functional.tanh(x)
                    else:
                        x = mint.nn.functional.sigmoid(x)
                return x

        net = MixedNet(apply_transform=True, transform_type="relu")
        offload_key = id(net)
        x = Tensor(np.random.randn(4, 128).astype(np.float32))

        ctx = manager.get_or_create_offload_ctx(offload_key)
        ctx.start_offload()
        _ = net(x)
        ctx.end_offload()
        manager.analyse()

        net.transform_type = "tanh"
        ctx2 = manager.get_or_create_offload_ctx(offload_key)
        ctx2.start_offload()

        with pytest.raises(Exception) as exc_info:
            _ = net(x)

        _assert_error_match(exc_info.value, r"Dynamic graph structure|Operator count mismatch")

        manager.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
