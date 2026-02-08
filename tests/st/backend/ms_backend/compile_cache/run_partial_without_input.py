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
from mindspore import Tensor, Parameter, ops, lazy_inline, nn, context, jit
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)

@jit
class SubNet(Cell):
    """
    SubNet: layer_norm with @lazy_inline
    Args:
        None
    Returns:
        Tensor, output tensor

    Examples:
        >>> output = SubNet()
    """
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.layer_norm = ops.LayerNorm()
        self.gamma = Tensor(np.ones([3]).astype(np.float32))
        self.beta = Tensor(np.ones([3]).astype(np.float32))

    def construct(self, sub_input):
        output, _, _ = self.layer_norm(sub_input, self.gamma, self.beta)
        return output


@jit
class SingleIfNet(nn.Cell):
    """
    SingleIfNet
    Args:
        None
    Inputs:
        x (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> input_x = Tensor(2, mstype.int32)
        >>> input_y = Tensor(5, mstype.int32)
        >>> forward_net = SingleIfNet()
        >>> graph_forward_res = forward_net(input_x, input_y)
    """
    def __init__(self):
        super().__init__()
        self.sub_net = SubNet()
        self.y = Parameter(Tensor(1))
        self.z = Parameter(Tensor(2))

    def construct(self, x, y):
        x += 1
        out = self.func(x, y)
        out = out * 2
        if self.y > self.z:
            out = out * 2
        return out

    def func(self, a, b):
        if a < b:
            b = b + a
        else:
            b = b - a
        b = b + 5
        return b


@jit
class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        return grad_all(self.net)(*inputs)


if __name__ == "__main__":
    context.set_context(jit_level='O0')

    input_x = Tensor(2, mstype.int32)
    input_y = Tensor(5, mstype.int32)
    forward_net = SingleIfNet()
    grad_net = GradNet(forward_net)
    expect1 = Tensor(26, mstype.int32)
    expect2 = (Tensor(2, mstype.int32), Tensor(2, mstype.int32))

    graph_forward_res = forward_net(input_x, input_y)
    assert graph_forward_res == expect1

    graph_backward_res = grad_net(input_x, input_y)
    assert graph_backward_res == expect2
    print("RUNTIME_COMPILE", graph_forward_res, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", graph_forward_res.asnumpy().shape, "RUNTIME_CACHE")
