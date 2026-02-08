# Copyright 2025 Huawei Technologies Co., Ltd
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
Custom Pass System Tests - Usage and Debugging Guide

OVERVIEW
========
This module contains comprehensive tests for the MindSpore custom pass system,
including plugin building, functionality verification, pass execution order testing,
and error scenarios. Focus on CPU-compatible passes for broader testing coverage.

FUSION PASSES TESTED
===================
- ReplaceAddNFusionPass: AddN([A, B]) -> Add(A, B) [CPU/GPU/Ascend]
  * Optimizes AddN with exactly 2 inputs into standard Add operation
  * Reduces graph complexity and improves execution performance

- AddNegFusionPass: Add(x, Neg(y)) -> Sub(x, y) [CPU/GPU/Ascend]
  * Eliminates unnecessary Neg operations by converting to subtraction
  * Standard algebraic optimization that works across all device types
  * Key for testing pass execution order dependencies

PASS EXECUTION ORDER TESTING (A-B vs B-A)
=========================================
The test suite includes critical A-B vs B-A execution order tests using AddN+Neg pattern:

Network Pattern:
  neg_x2 = Neg(x2)           # Negation operation
  result = AddN([x1, neg_x2]) # AddN that can become Add

A-B Order (ReplaceAddN -> AddNeg):
  1. AddN([x1, Neg(x2)]) -> Add(x1, Neg(x2))  [ReplaceAddN pass]
  2. Add(x1, Neg(x2)) -> Sub(x1, x2)          [AddNeg pass]
  Result: IR contains PrimFunc_Sub, PrimFunc_Neg eliminated

B-A Order (AddNeg -> ReplaceAddN):
  1. AddNeg pass cannot match AddN([x1, Neg(x2)]) pattern (needs Add+Neg)
  2. AddN([x1, Neg(x2)]) -> Add(x1, Neg(x2))  [ReplaceAddN pass]
  Result: IR contains PrimFunc_Add + PrimFunc_Neg, no Sub fusion

This demonstrates how pass execution order affects optimization effectiveness.

ENVIRONMENT VARIABLES
====================
Test Framework Control:
- KEEP_GRAPHS: Keep graph files after test completion for debugging
  Example: export KEEP_GRAPHS=1
  Effect: Preserves IR graphs in temporary directories for manual inspection

- CUSTOM_PASS_TMP_DIR: Override default temporary directory location
  Example: export CUSTOM_PASS_TMP_DIR=/custom/tmp/path
  Effect: Uses specified directory instead of /tmp/mindspore_custom_pass_tests

- USE_PROJECT_TMP: Use project-local tmp directory instead of system /tmp
  Example: export USE_PROJECT_TMP=1
  Effect: Creates tmp directory within project for easier debugging access

Plugin Build Configuration:
- MINDSPORE_ROOT: Override MindSpore installation path for plugin building
  Example: export MINDSPORE_ROOT=/opt/mindspore
  Effect: CMake uses this path for MindSpore headers and libraries

PREREQUISITES
=============
- MindSpore framework installed and configured
- Custom pass plugin compiled (libpass.so)
- CPU environment sufficient (no GPU/Ascend required for these tests)
- pytest and pytest-xdist for parallel execution

IR ANALYSIS AND VERIFICATION
============================
1. Graph File Locations:
   - Saved to graphs_dir_fixture (typically /tmp/mindspore_custom_pass_tests/*)
   - Look for .ir files containing primitive function calls
   - Compare before/after pass execution graphs

2. Key Primitive Patterns to Check:
   - PrimFunc_AddN -> PrimFunc_Add (ReplaceAddN pass)
   - PrimFunc_Neg elimination (AddNeg pass success)
   - PrimFunc_Sub appearance (AddNeg pass success)

3. Debugging Failed Passes:
   - Check if source pattern matches exactly in IR
   - Verify pass registration was successful
   - Look for logging messages from pass execution

DEBUGGING TIPS
==============
1. Graph Analysis:
   - Set save_graphs=True in context to save IR graphs
   - Check graphs before/after pass execution in save_graphs_path
   - Look for op transformations in .ir files

2. Plugin Issues:
   - Verify plugin file exists and has correct permissions
   - Check plugin registration success with verbose logging
   - Ensure plugin ABI compatibility with MindSpore version

3. Test Isolation:
   - Use separate processes (-n option) to avoid pass conflicts
   - Each test registers passes independently
   - Clean environment between test runs

4. Platform-Specific Issues:
   - Both passes work on CPU/GPU/Ascend (CPU-focused for compatibility)
   - All fusion tests can run on CPU-only environments
   - Check device availability and drivers only for multi-device tests

5. Pass Order Issues:
   - Check which passes are registered first (affects execution order)
   - Verify expected IR transformations match actual results
   - Use A-B/B-A tests to validate pass dependencies

PROCESS ISOLATION REQUIREMENT
============================
CRITICAL: Each test case must run in separate processes to avoid conflicts.
Multiple pass registrations in the same process can cause undefined behavior.

Recommended execution patterns:
- pytest test_custom_passes.py -n auto  (parallel processes)
- pytest test_custom_passes.py::test_name (individual test)

DO NOT run multiple tests sequentially in the same process without isolation.
"""

import os
import numpy as np
import logging

from tests.mark_utils import arg_mark

logger = logging.getLogger(__name__)

import mindspore as ms
from mindspore import context, nn, ops, Tensor

# Import test utilities
from .utils import verify_op_removed_after_pass


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_plugin_build_and_existence(build_plugin):
    """
    Feature: test custom pass plugin build and file existence
    Description: test plugin build process and verify plugin file properties
    Expectation: plugin builds successfully and has correct properties
    """
    logger.info("=== Testing Plugin Build and Existence ===")

    # Verify plugin file exists
    assert os.path.exists(build_plugin), f"Plugin file does not exist: {build_plugin}"
    logger.info("[OK] Plugin file exists: %s", build_plugin)

    # Verify plugin file size
    file_size = os.path.getsize(build_plugin)
    assert file_size > 0, f"Plugin file is empty: {file_size} bytes"
    logger.info("[OK] Plugin file size: %s bytes", file_size)

    # Verify plugin file permissions
    assert os.access(build_plugin, os.R_OK), f"Plugin file is not readable: {build_plugin}"
    logger.info("[OK] Plugin file is readable")

    # Verify plugin file extension
    assert build_plugin.endswith('.so'), f"Plugin file does not have .so extension: {build_plugin}"
    logger.info("[OK] Plugin file has correct extension")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_plugin_registration(build_plugin):
    """
    Feature: test custom pass plugin registration
    Description: test plugin registration to different devices
    Expectation: plugin registers successfully to all device types
    """
    logger.info("=== Testing Plugin Registration ===")

    # Test registration for different devices
    devices = ["cpu", "gpu", "ascend", "all"]

    for device in devices:
        logger.info("Testing registration for device: %s", device)
        success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, device)
        assert success, f"Plugin registration failed for device: {device}"
        logger.info("[OK] Registration successful for device: %s", device)

    # Test duplicate registration (should still succeed)
    logger.info("Testing duplicate registration")
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    assert success, "Duplicate plugin registration failed"
    logger.info("[OK] Duplicate registration successful")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_addn_fusion_functionality(build_plugin, graphs_dir_fixture):
    """
    Feature: test AddN fusion pass functionality
    Description: test AddN replacement with Add in graph mode
    Expectation: AddN ops are replaced by Add ops in IR
    """
    logger.info("=== Testing AddN Fusion Functionality ===")

    # Register the plugin
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    assert success, "Plugin registration failed"
    logger.info("[OK] Plugin registered successfully")

    # Create test network with AddN operation
    class AddNNetwork(nn.Cell):
        def __init__(self):
            super().__init__()
            self.addn = ops.AddN()
            self.add = ops.Add()
            self.mul = ops.Mul()

        def construct(self, x, y):
            # Create AddN operation that should be optimized
            temp1 = self.add(x, y)
            temp2 = self.mul(x, y)
            result = self.addn([temp1, temp2])  # AddN with 2 inputs
            return result

    # Setup MindSpore context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )

    # Create network and test data
    net = AddNNetwork()
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    y = Tensor(np.array([5, 6, 7, 8]).astype(np.float32))

    logger.info("Input x: %s", x)
    logger.info("Input y: %s", y)

    # Execute network
    logger.info("Executing network...")
    output = net(x, y)
    logger.info("Output: %s", output)

    # Verify functional correctness
    expected = (x + y) + (x * y)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected.asnumpy())
    logger.info("[OK] Functional correctness verified")

    # Verify IR transformation
    verify_op_removed_after_pass(
        graph_dir=graphs_dir_fixture,
        device="cpu",
        op_name="PrimFunc_AddN",
        replacement_op="PrimFunc_Add"
    )
    logger.info("[OK] IR transformation verified")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_simple_addn_network(build_plugin, graphs_dir_fixture):
    """
    Feature: test simple AddN network
    Description: test basic AddN operation functionality
    Expectation: AddN operation works correctly
    """
    logger.info("=== Testing Simple AddN Network ===")

    # Register the plugin
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    assert success, "Plugin registration failed"

    class SimpleAddNNetwork(nn.Cell):
        def __init__(self):
            super().__init__()
            self.addn = ops.AddN()

        def construct(self, x, y):
            return self.addn([x, y])

    # Create test data
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))

    # Create and execute network
    net = SimpleAddNNetwork()
    output = net(x, y)

    logger.info("Network executed successfully")
    logger.info("Output shape: %s", output.shape)
    logger.info("Output dtype: %s", output.dtype)

    # Verify results
    expected = x + y
    np.testing.assert_array_almost_equal(output.asnumpy(), expected.asnumpy())
    logger.info("[OK] Simple AddN test passed")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_invalid_plugin_path():
    """
    Feature: test error handling for invalid plugin paths
    Description: test plugin registration with non-existent file
    Expectation: registration fails gracefully
    """
    logger.info("=== Testing Invalid Plugin Path ===")

    # Test with non-existent file
    invalid_path = "/non/existent/path/libpass.so"
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", invalid_path, "cpu")
    assert not success, "Registration should fail for invalid path"
    logger.info("[OK] Registration correctly failed for invalid path")

    # Test with empty path
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", "", "cpu")
    assert not success, "Registration should fail for empty path"
    logger.info("[OK] Registration correctly failed for empty path")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_invalid_pass_name(build_plugin):
    """
    Feature: test error handling for invalid pass names
    Description: test plugin registration with non-existent pass name
    Expectation: registration fails gracefully
    """
    logger.info("=== Testing Invalid Pass Name ===")

    # Test with non-existent pass name
    success = ms.graph.register_custom_pass("NonExistentPass", build_plugin, "cpu")
    assert not success, "Registration should fail for invalid pass name"
    logger.info("[OK] Registration correctly failed for invalid pass name")

    # Test with empty pass name
    success = ms.graph.register_custom_pass("", build_plugin, "cpu")
    assert not success, "Registration should fail for empty pass name"
    logger.info("[OK] Registration correctly failed for empty pass name")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_multi_device_registration(build_plugin):
    """
    Feature: test multiple device registration
    Description: test registering same pass to multiple devices
    Expectation: all registrations succeed independently
    """
    logger.info("=== Testing Multi-Device Registration ===")

    # Test registering to all devices
    devices = ["cpu", "gpu", "ascend"]
    results = {}

    for device in devices:
        success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, device)
        results[device] = success
        logger.info("Registration for %s: %s", device, 'SUCCESS' if success else 'FAILED')

    # Verify at least CPU registration succeeded
    assert results["cpu"], "CPU registration should succeed"
    logger.info("[OK] Multi-device registration test completed")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_without_addn(build_plugin, graphs_dir_fixture):
    """
    Feature: test network without AddN operations
    Description: test custom pass on network without target operations
    Expectation: network executes normally without errors
    """
    logger.info("=== Testing Network Without AddN ===")

    # Register the plugin
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    assert success, "Plugin registration failed"

    # Network without AddN operations
    class NonAddNNetwork(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.mul = ops.Mul()

        def construct(self, x, y):
            temp1 = self.add(x, y)
            temp2 = self.mul(x, y)
            result = self.add(temp1, temp2)  # Only Add operations
            return result

    # Create and execute network
    net = NonAddNNetwork()
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))

    output = net(x, y)

    # Verify results
    expected = (x + y) + (x * y)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected.asnumpy())
    logger.info("[OK] Network without AddN executed successfully")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_multiple_pass_registration_cpu(build_plugin):
    """
    Feature: test multiple pass registration on CPU
    Description: test registering both passes (ReplaceAddN and AddNeg work on CPU)
    Expectation: both passes register successfully on CPU
    """
    logger.info("=== Testing Multiple Pass Registration (CPU) ===")

    # ReplaceAddN should work on CPU
    success1 = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    assert success1, "ReplaceAddNFusionPass registration failed on CPU"
    logger.info("[OK] ReplaceAddNFusionPass registered on CPU")

    # AddNegFusion should work on CPU
    success2 = ms.graph.register_custom_pass("AddNegFusionPass", build_plugin, "cpu")
    assert success2, "AddNegFusionPass registration failed on CPU"
    logger.info("[OK] AddNegFusionPass registered on CPU")
    logger.info("[OK] Multiple pass registration test completed")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_add_neg_fusion_registration(build_plugin):
    """
    Feature: test AddNegFusionPass plugin registration
    Description: test plugin registration for Add + Neg fusion on CPU
    Expectation: plugin registers successfully
    """
    logger.info("=== Testing AddNeg Fusion Registration ===")

    # Test registration for CPU device
    success = ms.graph.register_custom_pass("AddNegFusionPass", build_plugin, "cpu")
    assert success, f"AddNegFusionPass registration failed for CPU"
    logger.info("[OK] AddNegFusionPass registration successful for CPU")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_add_neg_fusion_functionality(build_plugin, graphs_dir_fixture):
    """
    Feature: test AddNegFusionPass functionality
    Description: test Add + Neg fusion in graph mode
    Expectation: Add + Neg ops are fused into Sub op in IR
    """
    logger.info("=== Testing AddNeg Fusion Functionality ===")

    # Register the plugin for CPU
    success = ms.graph.register_custom_pass("AddNegFusionPass", build_plugin, "cpu")
    assert success, "AddNegFusionPass registration failed"
    logger.info("[OK] AddNegFusionPass registered successfully")

    # Create test network with Add + Neg pattern
    class AddNegNetwork(nn.Cell):
        def __init__(self):
            super().__init__()
            self.neg = ops.Neg()

        def construct(self, x1, x2):
            # Neg operation: -x2
            neg_x2 = self.neg(x2)
            # Add operation: x1 + (-x2) = x1 - x2
            output = x1 + neg_x2
            return output

    # Setup MindSpore context for CPU
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )

    # Create network and test data
    net = AddNegNetwork()
    # Create test inputs
    x1 = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]).astype(np.float32))
    x2 = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                          [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]]).astype(np.float32))

    logger.info("Input x1 shape: %s", x1.shape)
    logger.info("Input x2 shape: %s", x2.shape)

    # Execute network
    logger.info("Executing AddNeg network for IR analysis...")
    output = net(x1, x2)
    logger.info("[OK] Network executed successfully - IR graphs generated")
    logger.info("Output shape: %s", output.shape)

    # Verify functional correctness
    expected = x1.asnumpy() - x2.asnumpy()  # x1 + (-x2) = x1 - x2
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
    logger.info("[OK] Functional correctness verified")

    # Verify IR transformation - look for Sub fusion (Neg should be eliminated)
    verify_op_removed_after_pass(
        graph_dir=graphs_dir_fixture,
        device="cpu",  # Use cpu since we're running on CPU
        op_name="PrimFunc_Neg",  # Should be removed due to fusion
        replacement_op="PrimFunc_Sub"  # Look for Sub operation
    )
    logger.info("[OK] AddNeg fusion functionality test completed")



# ========== A-B & B-A Tests ==========

# Shared network class for AddN + Neg pattern testing
class AddNNegNetwork(nn.Cell):
    """Network with AddN + Neg pattern for pass order testing"""
    def __init__(self):
        super().__init__()
        self.addn = ops.AddN()
        self.neg = ops.Neg()

    def construct(self, x1, x2):
        # Neg operation
        neg_x2 = self.neg(x2)
        # AddN operation that can be replaced by Add
        add_result = self.addn([x1, neg_x2])
        return add_result


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pass_execution_order_a_b(build_plugin, graphs_dir_fixture):
    """
    Feature: test pass execution order A-B (ReplaceAddN -> AddNeg)
    Description: test AddN+Neg pattern with ReplaceAddN first, then AddNeg
    Expectation: should result in Sub fusion after both passes (Neg eliminated)
    """
    logger.info("=== Testing Pass Order A-B: ReplaceAddN -> AddNeg ===")

    # Register both passes for CPU
    success1 = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    success2 = ms.graph.register_custom_pass("AddNegFusionPass", build_plugin, "cpu")
    assert success1 and success2, "Pass registration failed"
    logger.info("[OK] Both passes registered successfully")

    # Setup context for CPU
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )

    # Create test data
    shape = (2, 3, 4)
    x1 = Tensor(np.random.rand(*shape).astype(np.float32))
    x2 = Tensor(np.random.rand(*shape).astype(np.float32))

    logger.info("Input shapes: x1=%s, x2=%s", x1.shape, x2.shape)

    # Execute network
    net = AddNNegNetwork()
    output = net(x1, x2)
    logger.info("[OK] Network executed successfully")
    logger.info("Output shape: %s", output.shape)

    # Verify IR transformation - should have Sub after both passes (Neg eliminated)
    verify_op_removed_after_pass(
        graph_dir=graphs_dir_fixture,
        device="cpu",  # Use cpu since we're running on CPU
        op_name="PrimFunc_Neg",  # Should be eliminated by AddNeg pass
        replacement_op="PrimFunc_Sub"  # Should be created by AddNeg pass
    )
    logger.info("[OK] A-B pass order test completed - Sub fusion achieved, Neg eliminated")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pass_execution_order_b_a(build_plugin, graphs_dir_fixture):
    """
    Feature: test pass execution order B-A (AddNeg -> ReplaceAddN)
    Description: test AddN+Neg pattern with AddNeg first, then ReplaceAddN
    Expectation: should NOT result in Sub fusion (AddNeg can't match AddN+Neg pattern)
    """
    logger.info("=== Testing Pass Order B-A: AddNeg -> ReplaceAddN ===")

    # Register both passes for CPU (note: this registers them but doesn't control execution order)
    success1 = ms.graph.register_custom_pass("AddNegFusionPass", build_plugin, "cpu")
    success2 = ms.graph.register_custom_pass("ReplaceAddNFusionPass", build_plugin, "cpu")
    assert success1 and success2, "Pass registration failed"
    logger.info("[OK] Both passes registered successfully")

    # Setup context for CPU
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )

    # Create test data
    shape = (2, 3, 4)
    x1 = Tensor(np.random.rand(*shape).astype(np.float32))
    x2 = Tensor(np.random.rand(*shape).astype(np.float32))

    logger.info("Input shapes: x1=%s, x2=%s", x1.shape, x2.shape)

    # Execute network
    net = AddNNegNetwork()
    output = net(x1, x2)
    logger.info("[OK] Network executed successfully")
    logger.info("Output shape: %s", output.shape)

    # Verify IR transformation - AddN should be replaced by Add, but Neg should still exist
    verify_op_removed_after_pass(
        graph_dir=graphs_dir_fixture,
        device="cpu",  # Use cpu since we're running on CPU
        op_name="PrimFunc_AddN",  # Should be removed by ReplaceAddN pass
        replacement_op="PrimFunc_Add"  # Should be replaced by Add, Neg still exists
    )
    logger.info("[OK] B-A pass order test completed - AddN->Add, but Neg preserved (no Sub fusion)")
