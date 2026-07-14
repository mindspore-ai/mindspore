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
"""Test auto_offload with communication operators in PyNative mode.

This test requires multi-card environment (2P).
CI entry: test_entry_auto_offload_comm.py

Manual run:
#msrun --worker_num=2 --local_worker_num=2 --master_port=10971 --join=True --cluster_time_out=300 pytest -sv --disable-warnings test_pynative_auto_offload_comm.py
"""

import numpy as np
from mindspore import nn, Tensor, context, mint
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter
from mindspore.communication.management import init, get_rank
from mindspore.communication.comm_func import all_reduce, all_gather_into_tensor
from mindspore._c_expression import OffloadManager


def set_device():
    """Set device for distributed training."""
    context.set_context(mode=context.PYNATIVE_MODE)
    init()


class AllReduceNet(nn.Cell):
    """Network with AllReduce communication operator."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((64, 64)).astype(np.float32)), name='weight')

    def construct(self, x):
        """Forward pass with AllReduce."""
        # x: [batch, 64]
        y = mint.matmul(x, self.weight)
        y = mint.nn.functional.relu(y)

        # AllReduce: sum gradients across all ranks
        y, _ = all_reduce(y)

        # Continue computation
        y = mint.add(y, 1.0)
        y = mint.tanh(y)

        return y


def test_auto_offload_allreduce():
    """
    Feature: Test auto_offload with AllReduce communication operator.
    Description: AllReduce is a common communication pattern in data parallel training.
                 Tests that the tracer correctly handles communication operators.
    Expectation: Forward and backward passes should produce correct results across all ranks.
    """
    set_device()

    manager = OffloadManager.get_instance()
    manager.reset()

    net = AllReduceNet()
    net.auto_offload()

    rank = get_rank()

    # Each rank uses different input
    np.random.seed(rank)
    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    # Reference without auto_offload
    net_ref = AllReduceNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    # Warm-up phase
    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    # Execution phase
    exec_grads = grad_fn(net, list(net.trainable_params()))(x)

    # Verify
    np.testing.assert_allclose(
        exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(), rtol=1e-4, atol=1e-4
    )
    for _, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
        np.testing.assert_allclose(eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4)

    print(f"[Rank {rank}] test_auto_offload_allreduce passed successfully")


class AllGatherNet(nn.Cell):
    """Network with AllGather communication operator."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones((64, 64)).astype(np.float32)), name='weight')

    def construct(self, x):
        """Forward pass with AllGather."""
        # x: [batch, 64]
        y = mint.matmul(x, self.weight)
        y = mint.nn.functional.relu(y)

        # AllGather: gather tensors from all ranks
        y, _ = all_gather_into_tensor(y)  # Output shape: [batch * world_size, 64]

        # Continue computation with gathered tensor
        y = mint.mean(y, dim=0, keepdim=True)  # [1, 64]
        y = mint.tanh(y)

        return y


def test_auto_offload_allgather():
    """
    Feature: Test auto_offload with AllGather communication operator.
    Description: AllGather gathers tensors from all ranks, changing tensor shape.
                 Tests that the tracer correctly handles shape-changing communication.
    Expectation: Forward and backward passes should produce correct results.
    """
    set_device()

    manager = OffloadManager.get_instance()
    manager.reset()

    net = AllGatherNet()
    net.auto_offload()

    rank = get_rank()

    np.random.seed(rank)
    x = Tensor(np.random.randn(8, 64).astype(np.float32))

    net_ref = AllGatherNet()
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

    print(f"[Rank {rank}] test_auto_offload_allgather passed successfully")


class MixedCommNet(nn.Cell):
    """Network with mixed computation and communication."""

    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.ones((64, 128)).astype(np.float32) * 0.1), name='weight1')
        self.weight2 = Parameter(Tensor(np.ones((128, 64)).astype(np.float32) * 0.1), name='weight2')

    def construct(self, x):
        """Forward pass with mixed computation and communication."""
        # x: [batch, 64]
        # Local computation
        y = mint.matmul(x, self.weight1)  # [batch, 128]
        y = mint.nn.functional.relu(y)

        # Communication: AllReduce
        y, _ = all_reduce(y)

        # More local computation
        y = mint.matmul(y, self.weight2)  # [batch, 64]
        y = mint.tanh(y)

        # Another communication
        y, _ = all_reduce(y)

        # Final computation
        y = mint.add(y, x)  # Residual connection

        return y


def test_auto_offload_mixed_comm():
    """
    Feature: Test auto_offload with mixed computation and communication.
    Description: Real training scenarios have interleaved local computation and communication.
                 Tests that offload correctly handles the mixed pattern.
    Expectation: Forward and backward passes should produce correct results.
    """
    set_device()

    manager = OffloadManager.get_instance()
    manager.reset()

    net = MixedCommNet()
    net.auto_offload()

    rank = get_rank()

    np.random.seed(rank)
    x = Tensor(np.random.randn(16, 64).astype(np.float32))

    net_ref = MixedCommNet()
    for p1, p2 in zip(net.trainable_params(), net_ref.trainable_params()):
        p2.set_data(p1.data.copy())

    grad_fn = C.GradOperation(get_all=True, get_by_list=True)
    ref_grads = grad_fn(net_ref, list(net_ref.trainable_params()))(x)

    grad_fn(net, list(net.trainable_params()))(x)

    manager.analyse()

    # Test multiple iterations
    for iter_idx in range(3):
        exec_grads = grad_fn(net, list(net.trainable_params()))(x)

        np.testing.assert_allclose(
            exec_grads[0][0].asnumpy(), ref_grads[0][0].asnumpy(),
            rtol=1e-4, atol=1e-4,
            err_msg=f"[Rank {rank}] Input gradient mismatch at iteration {iter_idx}"
        )
        for i, (eg, rg) in enumerate(zip(exec_grads[1], ref_grads[1])):
            np.testing.assert_allclose(
                eg.asnumpy(), rg.asnumpy(), rtol=1e-4, atol=1e-4,
                err_msg=f"[Rank {rank}] Param[{i}] mismatch at iteration {iter_idx}"
            )

    print(f"[Rank {rank}] test_auto_offload_mixed_comm passed successfully")


if __name__ == "__main__":
    test_auto_offload_allreduce()
    test_auto_offload_allgather()
    test_auto_offload_mixed_comm()
