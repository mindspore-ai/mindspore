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
test compile cache with control flow.
"""
import numpy as np
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore import nn, Tensor, context, jit


class ForwardNet(nn.Cell):
    """
    ForForWhileForwardNet
    Args:
        max_cycles, default 10
    Inputs:
        x (Tensor): Input tensor of integer type.
        y (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> net = ForwardNet()
        >>> x = Tensor(np.array(1), mstype.int32)
        >>> y = Tensor(np.array(3), mstype.int32)
        >>> forward_net = ForwardNet(max_cycles=3)
        >>> output = forward_net(x, y)
    """
    def __init__(self, max_cycles=10):
        super().__init__()
        self.max_cycles = max_cycles
        self.zero = Tensor(np.array(0), mstype.int32)
        self.i = Tensor(np.array(0), mstype.int32)

    @jit(backend="ms_backend")
    def construct(self, x, y):
        out = self.zero
        for _ in range(0, self.max_cycles):
            out = x * y + out
        i = self.i
        while i < self.max_cycles:
            out = x * y + out
            i = i + 1
        return out


@jit
class BackwardNet(nn.Cell):
    def __init__(self, net):
        super().__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation()

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


def run_net_for_n_while():
    """
    Test ForForAfterWhileNet run in different mode (graph mode and pynative mode).
    It tests both forward pass and backward gradient computation to ensure
    compilation cache works correctly with complex control flow networks.

    Steps:
    1. Run in GRAPH_MODE
    2. Run in PYNATIVE_MODE
    3. Compare backward results between both mode
    4. Print results for verification
    """
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=3)
    backward_net = BackwardNet(forward_net)
    graph_grads = backward_net(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=3)
    backward_net = BackwardNet(forward_net)
    pynative_grads = backward_net(x, y)
    assert graph_grads == pynative_grads
    print("RUNTIME_COMPILE", graph_grads, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", graph_grads.asnumpy().shape, "RUNTIME_CACHE")


if __name__ == "__main__":
    run_net_for_n_while()
