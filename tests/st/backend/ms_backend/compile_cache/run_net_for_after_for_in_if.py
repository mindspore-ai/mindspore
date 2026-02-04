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
from mindspore import context, Tensor, nn, jit
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter

grad_all = C.GradOperation(get_all=True)


@jit
class ForAfterForInIfNet(nn.Cell):
    """
    ForAfterForInIfNet
    Args:
        None
    Inputs:
        x (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> net = ForAfterForInIfNet()
        >>> x = Tensor(5, mstype.int32)
        >>> output = net(x)
    """
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
        self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

    def construct(self, x):
        """Build control flow net."""
        out = self.param_a
        if self.param_a > self.param_b:
            for _ in range(0, 4):
                self.param_a = self.param_a + 1
                self.param_b = self.param_b - 3
        self.param_b = self.param_b + 10
        for _ in range(0, 5):
            x = x + self.param_b
        out = out * x
        return out


class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    @jit
    def construct(self, *inputs):
        return grad_all(self.net)(*inputs)


def run_for_after_for_in_if():
    """
    Test ForAfterForInIfNet run in different optimization levels (O0 and O1) in graph mode.
    It tests both forward pass and backward gradient computation to ensure
    compilation cache works correctly with complex control flow networks.

    Steps:
    1. Run in GRAPH_MODE with JIT level O0
    2. Run in GRAPH_MODE with JIT level O1
    3. Compare forward and backward results between both levels
    4. Print results for verification
    """
    x = Tensor(5, mstype.int32)

    # graph mode O0
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O0"})
    for_after_for_in_if_net = ForAfterForInIfNet()
    net = GradNet(for_after_for_in_if_net)

    forward_net = ForAfterForInIfNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    # jit mode O1
    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(jit_config={"jit_level": "O1"})
    for_after_for_in_if_net = ForAfterForInIfNet()
    net = GradNet(for_after_for_in_if_net)

    forward_net = ForAfterForInIfNet()
    pynative_forward_res = forward_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
    print("RUNTIME_COMPILE", graph_forward_res, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", graph_forward_res.asnumpy().shape, "RUNTIME_CACHE")


if __name__ == "__main__":
    run_for_after_for_in_if()
