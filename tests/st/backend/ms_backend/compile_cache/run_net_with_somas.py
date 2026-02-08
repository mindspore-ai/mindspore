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
test compile cache in jit mode with somas
"""
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor
import mindspore.ops.operations as P
from mindspore import context, ops, lazy_inline, nn, jit

@jit
class Grad(Cell):
    def __init__(self, input_net):
        super().__init__()
        self.grad = ops.GradOperation()
        self.net = input_net

    def construct(self, grad_input_x):
        inner_grad_net = self.grad(self.net)
        return inner_grad_net(grad_input_x)

@jit
class Block(Cell):
    def __init__(self):
        super().__init__()
        self.transpose1 = P.Transpose()
        self.transpose2 = P.Transpose()
        self.transpose3 = P.Transpose()
        self.transpose4 = P.Transpose()
        self.real_div1 = P.RealDiv()
        self.real_div2 = P.RealDiv()
        self.batch_matmul1 = P.BatchMatMul()
        self.batch_matmul2 = P.BatchMatMul()
        self.add = P.Add()
        self.softmax = P.Softmax(-1)
        self.expand_dims = P.ExpandDims()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.y = Tensor(np.ones((8, 128, 128)).astype(np.float16))

    def construct(self, block_input):
        transpose1 = self.transpose1(block_input, (0, 2, 1, 3))
        real_div1 = self.real_div1(transpose1, Tensor(2.37891).astype(np.float16))
        transpose2 = self.transpose2(block_input, (0, 2, 3, 1))
        real_div2 = self.real_div2(transpose2, Tensor(2.37891).astype(np.float16))
        batch_matmul1 = self.batch_matmul1(real_div1, real_div2)
        expand_dims = self.expand_dims(self.y, 1)
        sub = self.sub(Tensor([1.0]).astype(np.float16), expand_dims)
        mul = self.mul(sub, Tensor([-0.0001]).astype(np.uint8))
        add = self.add(mul, batch_matmul1)
        soft_max = self.softmax(add)
        transpose3 = self.transpose3(block_input, (0, 2, 1, 3))
        batch_matmul2 = self.batch_matmul2(soft_max, transpose3)
        transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
        return transpose4

@jit
class TestBlock(Cell):
    def construct(self, x):
        x = x + x
        x = x + 2
        x = x - 9
        return x

@jit
class OuterBlock(Cell):
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.test_block = TestBlock()

    def construct(self, outer_block_x):
        return ((self.test_block(outer_block_x), outer_block_x), self.block(outer_block_x))

@jit
class ForBlockNet(Cell):
    def __init__(self):
        super().__init__()
        self.blocks = nn.CellList()
        for _ in range(3):
            b = OuterBlock()
            self.blocks.append(b)

    def construct(self, net_input):
        out = net_input
        for i in range(3):
            out = self.blocks[i](out)[0][1] + self.blocks[i](out)[0][0] + self.blocks[i](out)[1]
        return out


if __name__ == "__main__":
    context.set_context(jit_level='O1', memory_optimize_level="O0")
    net = ForBlockNet()
    grad_net = Grad(net)
    output = grad_net(Tensor(np.ones((8, 128, 16, 32)).astype(np.float16)))
    print("RUNTIME_COMPILE", output[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", output[0].asnumpy().shape, "RUNTIME_CACHE")
