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
from mindspore import Tensor, ops, nn, lazy_inline, context, jit
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell
import mindspore.ops.operations as P


@jit
class Grad(Cell):
    def __init__(self, net):
        super().__init__()
        self.grad = ops.GradOperation()
        self.net = net

    def construct(self, grad_x):
        grad_net = self.grad(self.net)
        return grad_net(grad_x)


class Block(Cell):
    """Build block for BaseBlock"""
    def __init__(self):
        super().__init__()
        self.batch_matmul = P.BatchMatMul()
        self.expand_dims = P.ExpandDims()
        self.y = Parameter(Tensor(np.ones((8)).astype(np.float32)))

    def construct(self, block_x):
        z1 = self.batch_matmul(block_x, block_x)
        z2 = self.expand_dims(self.y, 1)
        return z1 + z2


class BaseBlock(Cell):
    """BaseBlock with @lazy_inline"""
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.block = Block()

    def construct(self, base_block_x):
        return self.block(base_block_x)

class LazyInlineNet(Cell):
    """
    LazyInlineNet
    Args:
        None
    Inputs:
        x (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> net = LazyInlineNet()
        >>> x = Tensor(np.array(1), mstype.int32)
        >>> output = LazyInlineNet(x)
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.CellList()
        b = BaseBlock()
        self.blocks.append(b)

    @jit
    def construct(self, lazy_net_x):
        out = lazy_net_x
        for _ in range(5):
            out = self.blocks[0](out)
        return out


@jit
class GradNet(Cell):
    """
    GradNet for LazyInlineNet
    Args:
        None
    Inputs:
        net

    Returns:
        Tensor, output tensors

    Examples:
        >>> input_x = Tensor(np.ones((8, 8)).astype(np.float32))
        >>> input_y = Tensor(6)
        >>> lazy_inline_net = LazyInlineNet()
        >>> lazy_inline_grad_net = GradNet(lazy_inline_net)
        >>> output = lazy_inline_grad_net(input_x, input_y)
    """
    def __init__(self, input_net):
        super().__init__()
        self.grad_net = Grad(input_net)
        self.a = Parameter(Tensor(np.ones((8)).astype(np.float32)))
        self.b = Parameter(Tensor(np.ones((8)).astype(np.float32)))

    def construct(self, x, y):
        out = self.grad_net(x)
        if y > 3:
            return out * 2, self.a
        return out, self.b


if __name__ == "__main__":
    # graph mode
    context.set_context(jit_config={"jit_level": "O0"})
    input_x = Tensor(np.ones((8, 8)).astype(np.float32))
    input_y = Tensor(6)
    lazy_inline_net = LazyInlineNet()
    lazy_inline_grad_net = GradNet(lazy_inline_net)
    output = lazy_inline_grad_net(input_x, input_y)
    print("RUNTIME_COMPILE", output[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", output[0].asnumpy().shape, "RUNTIME_CACHE")
