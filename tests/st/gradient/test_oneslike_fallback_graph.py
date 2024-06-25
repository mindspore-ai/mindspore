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
"""test function grad in graph mode"""
import random
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.functional import grad
from mindspore.common import dtype as mstype
from mindspore import Parameter
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_oneslike_fallback_with_tensor():
    """
    Features: Multitype Funcgraph oneslike by jit fallback.
    Description: Test oneslike with jit fallback.
    Expectation: No exception.
    """
    class SubClass:
        x = 2

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self, sub):
            super(ParamMultipleInputNet, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.subclass = sub

        def construct(self, x, y):
            outputs = x * y * self.w * self.subclass.x
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, self.weights, return_ids=True)(x, y)
            return res

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet(SubClass())
    grad_net = GradNet(inner_net)
    grad_out = grad_net(x, y)
    assert np.allclose(grad_out[0][1].asnumpy(),
                       np.array([13, 13]).astype(np.float32))
    assert np.allclose(grad_out[1][0][1].asnumpy(),
                       np.array([7, 13]).astype(np.float32))


class GradNet1(nn.Cell):
    def __init__(self, net, grad_position=0):
        super().__init__()
        self.grad = grad
        self.grad_net = self.grad(net, grad_position=grad_position)

    def construct(self, *x):
        return self.grad_net(*x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_oneslike_fallback_with_empty_grad():
    """
    Features: Multitype Funcgraph oneslike by jit fallback.
    Description: Test oneslike with jit fallback.
    Expectation: No exception.
    """
    class SubClass:
        x = 2
        y = 15

    class InnerClass(nn.Cell):
        def __init__(self, sub):
            super().__init__()
            self.subclass = sub

        def construct(self, x):
            b = self.subclass.x * self.subclass.y * self.subclass.y
            c = b - self.subclass.x
            d = c / self.subclass.y
            e = d // self.subclass.x
            f = e % self.subclass.y
            g = f**self.subclass.x
            return g
    x = random.randint(-5, 10)
    ms_grad = GradNet1(InnerClass(SubClass()))(x)
    assert ms_grad == ()
