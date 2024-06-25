# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test while with grad in PIJit and pynative mode"""
import numpy as np

from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import Parameter
from mindspore import Tensor, jit, context
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from tests.mark_utils import arg_mark


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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_grad():
    """
    Feature: while with grad.
    Description: set while in network and do backward grad
    when all the branches can not be inferred.
    Expectation: No error raised.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([2.0], dtype=mstype.float32)
    y = Tensor([2.0], dtype=mstype.float32)
    ms_net = GradNet(Net())
    jit(GradNet.construct, mode="PIJit")(ms_net, x, y)
    GradNet(ms_net)(x, y)


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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_header_spec_twice():
    """
    Feature: FuncGraph Cloner.
    Description: While header will be specialized to 2 graphs, because common call header is RefTensor but body call
    header is Tensor.Related issue:I5HVPJ.
    Expectation: No error raised.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([3], np.float32))
    y = Tensor(np.array([1], np.float32))
    net = WhileSpecTwiceNet()
    jit(WhileSpecTwiceNet.construct, mode="PIJit")(net, x, y)
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(x, y)
    print('ms backward: ', fgrad)
