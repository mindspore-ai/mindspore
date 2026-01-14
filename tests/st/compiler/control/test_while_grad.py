# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test_while_grad.py """
from tests.mark_utils import arg_mark
import numpy as np
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import Parameter
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore import nn

context.set_context(jit_config={"jit_level": "O0"})

class Net(nn.Cell):
    def construct(self, x, y):
        while x < y:
            x = x * x + 1
        return x


class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = C.GradOperation(get_all=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_while_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor([2.0], dtype=mstype.float32)
    y = Tensor([2.0], dtype=mstype.float32)
    GradNet(Net())(x, y)


class WhileSpecTwiceNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([(- 3)], mstype.float32), name='w')
        self.b = Parameter(Tensor([(- 2)], mstype.float32), name='b')

    def construct(self, x, y):
        x = self.b
        while y > x:
            x = y + 2
        return y


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_while_header_spec_twice():
    """
    Feature: FuncGraph Cloner.
    Description: While header will be specialized to 2 graphs, because common call header is RefTensor but body call
    header is Tensor.Related issue:I5HVPJ.
    Expectation: No error raised.
    """
    x = Tensor(np.array([3], np.float32))
    y = Tensor(np.array([1], np.float32))
    net = WhileSpecTwiceNet()
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(x, y)
    print('ms backward: ', fgrad)
