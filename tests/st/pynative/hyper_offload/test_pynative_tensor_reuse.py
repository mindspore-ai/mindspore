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
"""Test tensor reuse scenarios for HyperOffload in PyNative mode.

This module tests scenarios where a tensor is used by multiple operators,
especially with long distances between uses (> select_distance).

Key principle: Input data must be identical for accuracy comparison.
"""

import os
import random
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


def _setup_offload_config(select_distance=1, prefetch_distance=0):
    """Setup HyperOffload configuration parameters."""
    origin_select = compile_config.HYPER_OFFLOAD_SELECT_DISTANCE
    origin_prefetch = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = select_distance
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = prefetch_distance
    return origin_select, origin_prefetch


def _restore_offload_config(origin_select, origin_prefetch):
    """Restore HyperOffload configuration."""
    compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = origin_select
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = origin_prefetch


def _verify_gradients(grads_with, grads_no):
    """Verify gradient consistency."""
    if isinstance(grads_no, tuple):
        for i, (gw, gn) in enumerate(zip(grads_with, grads_no)):
            if isinstance(gn, (tuple, list)):
                _verify_gradients(gw, gn)
            elif hasattr(gw, 'asnumpy') and hasattr(gn, 'asnumpy'):
                np.testing.assert_allclose(
                    gw.asnumpy(), gn.asnumpy(), rtol=RTOL, atol=ATOL,
                    err_msg=f"Gradient mismatch at index {i}"
                )


def _copy_weights(net_from, net_to):
    """Copy weights from one network to another."""
    for p1, p2 in zip(net_from.trainable_params(), net_to.trainable_params()):
        if hasattr(p1, 'data') and hasattr(p2, 'data'):
            try:
                p2.set_data(p1.data.copy())
            except:
                pass


def _set_weights(net, weights_dict):
    """Set network weights from dictionary."""
    for name, param in net.parameters_and_names():
        if name in weights_dict:
            try:
                param.set_data(weights_dict[name])
            except:
                pass


def _get_weights(net):
    """Get network weights as dictionary."""
    weights = {}
    for name, param in net.parameters_and_names():
        weights[name] = param.data.copy()
    return weights


# ============================================================================
# Network Definitions (Same as before)
# ============================================================================

class ShortDistanceNet(nn.Cell):
    """Tensor used by multiple operators with short distance."""

    def __init__(self):
        super(ShortDistanceNet, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 128)
        self.fc3 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x1 = mint.nn.functional.relu(t0)
        x2 = mint.nn.functional.tanh(t0)
        x3 = mint.nn.functional.sigmoid(t0)
        out = self.fc2(x3)
        return self.fc3(out)


class ThresholdNet(nn.Cell):
    """Tensor used by operators at threshold distance."""

    def __init__(self):
        super(ThresholdNet, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(5):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class MediumDistanceNet(nn.Cell):
    """Tensor used by operators with medium distance."""

    def __init__(self):
        super(MediumDistanceNet, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 128)
        self.fc3 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(8):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class LongDistanceNet(nn.Cell):
    """Tensor used by operators with long distance (15+ operators)."""

    def __init__(self):
        super(LongDistanceNet, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(15):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class BasicResidualNet(nn.Cell):
    """Basic residual connection: output = F(input) + input."""

    def __init__(self):
        super(BasicResidualNet, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 128)
        self.fc3 = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        identity = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = mint.add(x, identity)
        return self.fc3(out)


class MultiLayerResidualNet(nn.Cell):
    """Multi-layer residual blocks."""

    def __init__(self, num_blocks=3):
        super(MultiLayerResidualNet, self).__init__()
        self.num_blocks = num_blocks
        self.fc_input = nn.Dense(128, 128)
        self.fc_blocks = nn.CellList([nn.Dense(128, 128) for _ in range(num_blocks)])
        self.fc_output = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        x = self.fc_input(x)
        x = self.relu(x)
        identity = x
        for i, fc in enumerate(self.fc_blocks):
            out = fc(x)
            out = self.relu(out)
            if i == 0:
                identity = out
            out = mint.add(out, identity)
            out = self.relu(out)
            identity = out
        return self.fc_output(x)


class SkipConnectionNet(nn.Cell):
    """Skip connection where early tensor is used much later."""

    def __init__(self):
        super(SkipConnectionNet, self).__init__()
        self.fc_early = nn.Dense(128, 128)
        self.fc_middle = nn.Dense(128, 128)
        self.fc_late = nn.Dense(128, 128)
        self.fc_output = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        skip_tensor = self.fc_early(x)
        x = self.relu(skip_tensor)
        for _ in range(5):
            x = self.relu(self.fc_middle(x))
        out = mint.add(x, skip_tensor)
        return self.fc_output(self.relu(self.fc_late(out)))


class DualBranchNet(nn.Cell):
    """Dual branch network with shared tensor."""

    def __init__(self):
        super(DualBranchNet, self).__init__()
        self.fc_shared = nn.Dense(128, 256)
        self.fc1 = nn.Dense(256, 128)
        self.fc2 = nn.Dense(256, 128)
        self.fc_merge = nn.Dense(256, 64)

    def construct(self, x):
        t0 = self.fc_shared(x)
        branch1 = mint.nn.functional.relu(self.fc1(t0))
        branch2 = mint.nn.functional.tanh(self.fc2(t0))
        merged = mint.cat([branch1, branch2], dim=1)
        return self.fc_merge(merged)


class QuadBranchNet(nn.Cell):
    """Four branch network with shared tensor."""

    def __init__(self):
        super(QuadBranchNet, self).__init__()
        self.fc_shared = nn.Dense(128, 128)
        self.fc_branches = nn.CellList([nn.Dense(128, 64) for _ in range(4)])
        self.fc_merge = nn.Dense(256, 64)
        self.relu = mint.nn.functional.relu
        self.tanh = mint.nn.functional.tanh

    def construct(self, x):
        t0 = self.fc_shared(x)
        branches = []
        for i, fc in enumerate(self.fc_branches):
            if i % 2 == 0:
                branch = self.relu(t0)
            else:
                branch = self.tanh(t0)
            branch = fc(branch)
            branches.append(branch)
        merged = mint.cat(branches, dim=1)
        return self.fc_merge(merged)


class SkipBranchNet(nn.Cell):
    """Branch network with skip connections."""

    def __init__(self):
        super(SkipBranchNet, self).__init__()
        self.fc_input = nn.Dense(128, 128)
        self.fc_branches = nn.CellList([nn.Dense(128, 128) for _ in range(3)])  # Output 128
        self.fc_skip_project = nn.Dense(128, 128)  # Project skip to 128
        self.fc_output = nn.Dense(384, 64)  # 128 * 3 = 384
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        skip = self.fc_input(x)  # (batch, 128)
        skip = self.fc_skip_project(skip)  # (batch, 128)
        x = self.relu(skip)
        branches = []
        for i, fc in enumerate(self.fc_branches):
            branch = x
            for _ in range(2):
                branch = self.relu(fc(branch))  # (batch, 128)
            branches.append(branch)
        merged = mint.cat(branches, dim=1)  # (batch, 128*3) = (batch, 384)
        return self.fc_output(merged)


class Distance10Net(nn.Cell):
    """Tensor with 10 intermediate operators."""

    def __init__(self):
        super(Distance10Net, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(10):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class Distance20Net(nn.Cell):
    """Tensor with 20 intermediate operators."""

    def __init__(self):
        super(Distance20Net, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(20):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class Distance50Net(nn.Cell):
    """Tensor with 50 intermediate operators (stress test)."""

    def __init__(self):
        super(Distance50Net, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(50):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class Distance100Net(nn.Cell):
    """Tensor with 100 intermediate operators (boundary test)."""

    def __init__(self):
        super(Distance100Net, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 64)

    def construct(self, x):
        t0 = self.fc1(x)
        x_intermediate = t0
        for _ in range(100):
            x_intermediate = mint.nn.functional.relu(self.fc1(x_intermediate))
        out = mint.add(x_intermediate, t0)
        return self.fc2(out)


class TupleShareNet(nn.Cell):
    """Tuple outputs shared by multiple operators."""

    def __init__(self):
        super(TupleShareNet, self).__init__()
        self.fc1 = nn.Dense(128, 256)
        self.fc2 = nn.Dense(128, 128)
        self.fc3 = nn.Dense(128, 128)
        self.fc_out = nn.Dense(256, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x1, x2 = mint.split(x, 128, dim=1)
        out1 = self.fc2(x1)
        out2 = self.fc3(x2)
        merged = mint.cat([out1, out2], dim=1)
        return self.fc_out(merged)


class BroadcastShareNet(nn.Cell):
    """Broadcast tensor shared by multiple operations."""

    def __init__(self):
        super(BroadcastShareNet, self).__init__()
        self.fc_shared = nn.Dense(128, 128)
        self.fc_branches = nn.CellList([nn.Dense(128, 128) for _ in range(3)])
        self.fc_output = nn.Dense(384, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        shared = self.fc_shared(x)  # (batch, 128) -> (batch, 128)
        shared = self.relu(shared)
        
        branches = []
        for fc in self.fc_branches:
            branch = shared
            for _ in range(2):
                branch = self.relu(fc(branch))  # (batch, 128) -> (batch, 128)
            branches.append(branch)
        
        merged = mint.cat(branches, dim=1)  # (batch, 128*3) = (batch, 384)
        return self.fc_output(merged)


class InplaceShareNet(nn.Cell):
    """Inplace operations sharing tensor."""

    def __init__(self):
        super(InplaceShareNet, self).__init__()
        self.fc1 = nn.Dense(128, 128)
        self.fc2 = nn.Dense(128, 128)
        self.fc3 = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        t0 = self.fc1(x)
        t0 = self.relu(t0)
        t1 = self.fc2(t0)
        t1 = self.relu(t1)
        out = mint.add(t0, t1)
        return self.fc3(out)


class SplitNet(nn.Cell):
    """Split operator with multiple tensor outputs."""

    def __init__(self):
        super(SplitNet, self).__init__()
        self.fc1 = nn.Dense(256, 256)
        self.fc2 = nn.Dense(128, 64)
        self.fc3 = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x1, x2 = mint.split(x, 128, dim=1)
        x1 = self.relu(self.fc2(x1))
        x2 = self.relu(self.fc3(x2))
        merged = mint.cat([x1, x2], dim=1)
        return merged


class SplitFirstNet(nn.Cell):
    """Split with TupleGetItem to get first tensor only."""

    def __init__(self):
        super(SplitFirstNet, self).__init__()
        self.fc1 = nn.Dense(256, 256)
        self.fc2 = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        splits = mint.split(x, 128, dim=1)
        x1 = splits[0]
        x1 = self.relu(self.fc2(x1))
        return x1


class SplitSecondNet(nn.Cell):
    """Split with TupleGetItem to get second tensor only."""

    def __init__(self):
        super(SplitSecondNet, self).__init__()
        self.fc1 = nn.Dense(256, 256)
        self.fc2 = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        splits = mint.split(x, 128, dim=1)
        x2 = splits[1]
        x2 = self.relu(self.fc2(x2))
        return x2


class MultiSplitNet(nn.Cell):
    """Split into multiple parts with TupleGetItem for some."""

    def __init__(self):
        super(MultiSplitNet, self).__init__()
        self.fc1 = nn.Dense(256, 256)
        self.fc2 = nn.Dense(64, 64)
        self.fc3 = nn.Dense(64, 64)
        self.fc4 = nn.Dense(128, 64)
        self.relu = mint.nn.functional.relu

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        parts = mint.split(x, 64, dim=1)
        p0 = self.relu(self.fc2(parts[0]))
        p2 = self.relu(self.fc3(parts[2]))
        merged = mint.cat([p0, p2], dim=1)
        merged = self.fc4(merged)
        return merged


# ============================================================================
# Test Cases with Consistent Input for Accuracy
# ============================================================================

def _run_accuracy_test(net_class, input_data, select_distance, prefetch_distance, test_name):
    """
    Run accuracy test with consistent input data.
    
    Key: Uses SAME input tensor for both with/without offload comparison.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    origin_select, origin_prefetch = _setup_offload_config(select_distance, prefetch_distance)

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = net_class()
    net_no_offload.set_grad()
    weights = _get_weights(net_no_offload)

    output_no_offload = net_no_offload(input_data)
    grads_no_offload = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(input_data)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = net_class()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()
    _set_weights(net_with_offload, weights)

    _ = net_with_offload(input_data)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(input_data)
    manager.analyse()

    output_with_offload = net_with_offload(input_data)
    grads_with_offload = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(input_data)

    np.testing.assert_allclose(
        output_with_offload.asnumpy(), output_no_offload.asnumpy(),
        rtol=RTOL, atol=ATOL, err_msg=f"{test_name}: Output mismatch"
    )
    _verify_gradients(grads_with_offload, grads_no_offload)

    _restore_offload_config(origin_select, origin_prefetch)
    
    return output_no_offload, output_with_offload, grads_no_offload, grads_with_offload


def _run_memory_test(net_class, input_data, select_distance, prefetch_distance, test_name):
    """
    Run memory test to measure peak memory usage.
    
    Key: Separate test from accuracy to avoid resource interference.
    Key: Verify that peak memory decreases with hyperoffload enabled.
    Key: Ensure weight consistency between with/without offload runs.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    origin_select, origin_prefetch = _setup_offload_config(select_distance, prefetch_distance)

    grad_net = ops.GradOperation(get_all=True, get_by_list=True)

    net_no_offload = net_class()
    net_no_offload.set_grad()
    weights = _get_weights(net_no_offload)
    
    hal.reset_peak_memory_stats()
    _ = net_no_offload(input_data)
    _ = grad_net(net_no_offload, list(net_no_offload.trainable_params()))(input_data)
    mem_no_offload = hal.max_memory_allocated()

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    manager = OffloadManager.get_instance()
    manager.reset()

    net_with_offload = net_class()
    net_with_offload.auto_offload()
    net_with_offload.set_grad()
    _set_weights(net_with_offload, weights)

    _ = net_with_offload(input_data)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(input_data)
    manager.analyse()

    hal.reset_peak_memory_stats()
    _ = net_with_offload(input_data)
    _ = grad_net(net_with_offload, list(net_with_offload.trainable_params()))(input_data)
    mem_with_offload = hal.max_memory_allocated()

    print(f"{test_name}: No-offload={mem_no_offload}, With-offload={mem_with_offload}")
    
    assert mem_with_offload < mem_no_offload, \
        f"{test_name}: Memory with offload ({mem_with_offload}) should be less than without offload ({mem_no_offload})"

    _restore_offload_config(origin_select, origin_prefetch)
    
    return mem_no_offload, mem_with_offload


# ============================================================================
# Accuracy Test Cases (REUSE_ACC_001-017)
# ============================================================================

class TestTensorReuseAccuracy:
    """Test accuracy with consistent input data."""

    def test_reuse_acc_001_short_distance(self):
        """ACC_001: Short distance accuracy with same input."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(ShortDistanceNet, input_data, 5, 2, "ACC_001")

    def test_reuse_acc_002_threshold_distance(self):
        """ACC_002: Threshold distance accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(ThresholdNet, input_data, 5, 2, "ACC_002")

    def test_reuse_acc_003_medium_distance(self):
        """ACC_003: Medium distance accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(MediumDistanceNet, input_data, 5, 2, "ACC_003")

    def test_reuse_acc_004_long_distance(self):
        """ACC_004: Long distance accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(LongDistanceNet, input_data, 5, 2, "ACC_004")

    def test_reuse_acc_005_basic_residual(self):
        """ACC_005: Basic residual accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(BasicResidualNet, input_data, 5, 2, "ACC_005")

    def test_reuse_acc_006_multi_layer_residual(self):
        """ACC_006: Multi-layer residual accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(MultiLayerResidualNet, input_data, 5, 2, "ACC_006")

    def test_reuse_acc_007_skip_connection(self):
        """ACC_007: Skip connection accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(SkipConnectionNet, input_data, 5, 2, "ACC_007")

    def test_reuse_acc_008_dual_branch(self):
        """ACC_008: Dual branch accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(DualBranchNet, input_data, 5, 2, "ACC_008")

    def test_reuse_acc_009_quad_branch(self):
        """ACC_009: Quad branch accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(QuadBranchNet, input_data, 5, 2, "ACC_009")

    def test_reuse_acc_010_skip_branch(self):
        """ACC_010: Skip branch accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(SkipBranchNet, input_data, 5, 2, "ACC_010")

    def test_reuse_acc_011_distance_10(self):
        """ACC_011: Distance 10 accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(Distance10Net, input_data, 5, 2, "ACC_011")

    def test_reuse_acc_012_distance_20(self):
        """ACC_012: Distance 20 accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(Distance20Net, input_data, 10, 3, "ACC_012")

    def test_reuse_acc_013_distance_50(self):
        """ACC_013: Distance 50 accuracy."""
        input_data = Tensor(np.random.randn(8, 128).astype(np.float32))
        _run_accuracy_test(Distance50Net, input_data, 20, 5, "ACC_013")

    def test_reuse_acc_014_distance_100(self):
        """ACC_014: Distance 100 accuracy."""
        input_data = Tensor(np.random.randn(8, 128).astype(np.float32))
        _run_accuracy_test(Distance100Net, input_data, 50, 10, "ACC_014")

    def test_reuse_acc_015_tuple_share(self):
        """ACC_015: Tuple share accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(TupleShareNet, input_data, 5, 2, "ACC_015")

    def test_reuse_acc_016_broadcast_share(self):
        """ACC_016: Broadcast share accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(BroadcastShareNet, input_data, 5, 2, "ACC_016")

    def test_reuse_acc_017_inplace_share(self):
        """ACC_017: Inplace share accuracy."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_accuracy_test(InplaceShareNet, input_data, 5, 2, "ACC_017")

    def test_reuse_acc_018_split_multi_output(self):
        """ACC_018: Split multi-output accuracy."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_accuracy_test(SplitNet, input_data, 5, 2, "ACC_018")

    def test_reuse_acc_019_split_first_tuplegetitem(self):
        """ACC_019: Split with TupleGetItem(0) to get first tensor."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_accuracy_test(SplitFirstNet, input_data, 5, 2, "ACC_019")

    def test_reuse_acc_020_split_second_tuplegetitem(self):
        """ACC_020: Split with TupleGetItem(1) to get second tensor."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_accuracy_test(SplitSecondNet, input_data, 5, 2, "ACC_020")

    def test_reuse_acc_021_multi_split_tuplegetitem(self):
        """ACC_021: Multi-split with TupleGetItem for some parts."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_accuracy_test(MultiSplitNet, input_data, 5, 2, "ACC_021")


# ============================================================================
# Memory Test Cases (REUSE_MEM_001-021)
# ============================================================================

class TestTensorReuseMemory:
    """Test memory usage with separate runs."""

    def test_reuse_mem_001_short_distance(self):
        """MEM_001: Short distance memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(ShortDistanceNet, input_data, 5, 2, "MEM_001")

    def test_reuse_mem_002_threshold_distance(self):
        """MEM_002: Threshold distance memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(ThresholdNet, input_data, 5, 2, "MEM_002")

    def test_reuse_mem_003_medium_distance(self):
        """MEM_003: Medium distance memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(MediumDistanceNet, input_data, 5, 2, "MEM_003")

    def test_reuse_mem_004_long_distance(self):
        """MEM_004: Long distance memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(LongDistanceNet, input_data, 5, 2, "MEM_004")

    def test_reuse_mem_005_basic_residual(self):
        """MEM_005: Basic residual memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(BasicResidualNet, input_data, 5, 2, "MEM_005")

    def test_reuse_mem_006_multi_layer_residual(self):
        """MEM_006: Multi-layer residual memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(MultiLayerResidualNet, input_data, 5, 2, "MEM_006")

    def test_reuse_mem_007_skip_connection(self):
        """MEM_007: Skip connection memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(SkipConnectionNet, input_data, 5, 2, "MEM_007")

    def test_reuse_mem_008_dual_branch(self):
        """MEM_008: Dual branch memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(DualBranchNet, input_data, 5, 2, "MEM_008")

    def test_reuse_mem_009_quad_branch(self):
        """MEM_009: Quad branch memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(QuadBranchNet, input_data, 5, 2, "MEM_009")

    def test_reuse_mem_010_skip_branch(self):
        """MEM_010: Skip branch memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(SkipBranchNet, input_data, 5, 2, "MEM_010")

    def test_reuse_mem_011_distance_10(self):
        """MEM_011: Distance 10 memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(Distance10Net, input_data, 5, 2, "MEM_011")

    def test_reuse_mem_012_distance_20(self):
        """MEM_012: Distance 20 memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(Distance20Net, input_data, 10, 3, "MEM_012")

    def test_reuse_mem_013_distance_50(self):
        """MEM_013: Distance 50 memory."""
        input_data = Tensor(np.random.randn(8, 128).astype(np.float32))
        _run_memory_test(Distance50Net, input_data, 20, 5, "MEM_013")

    def test_reuse_mem_014_distance_100(self):
        """MEM_014: Distance 100 memory."""
        input_data = Tensor(np.random.randn(8, 128).astype(np.float32))
        _run_memory_test(Distance100Net, input_data, 50, 10, "MEM_014")

    def test_reuse_mem_015_tuple_share(self):
        """MEM_015: Tuple share memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(TupleShareNet, input_data, 5, 2, "MEM_015")

    def test_reuse_mem_016_broadcast_share(self):
        """MEM_016: Broadcast share memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(BroadcastShareNet, input_data, 5, 2, "MEM_016")

    def test_reuse_mem_017_inplace_share(self):
        """MEM_017: Inplace share memory."""
        input_data = Tensor(np.random.randn(16, 128).astype(np.float32))
        _run_memory_test(InplaceShareNet, input_data, 5, 2, "MEM_017")

    def test_reuse_mem_018_split_multi_output(self):
        """MEM_018: Split multi-output memory."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_memory_test(SplitNet, input_data, 5, 2, "MEM_018")

    def test_reuse_mem_019_split_first_tuplegetitem(self):
        """MEM_019: Split with TupleGetItem(0) to get first tensor."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_memory_test(SplitFirstNet, input_data, 5, 2, "MEM_019")

    def test_reuse_mem_020_split_second_tuplegetitem(self):
        """MEM_020: Split with TupleGetItem(1) to get second tensor."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_memory_test(SplitSecondNet, input_data, 5, 2, "MEM_020")

    def test_reuse_mem_021_multi_split_tuplegetitem(self):
        """MEM_021: Multi-split with TupleGetItem for some parts."""
        input_data = Tensor(np.random.randn(16, 256).astype(np.float32))
        _run_memory_test(MultiSplitNet, input_data, 5, 2, "MEM_021")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
