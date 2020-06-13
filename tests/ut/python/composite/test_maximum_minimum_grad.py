# Copyright 2020 Huawei Technologies Co., Ltd
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

import logging
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell, Composite
from mindspore.ops import operations as P

log = logging.getLogger("ME")
log.setLevel(level=logging.DEBUG)
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="Ascend")

class MaximumGrad(Composite):
    def __init__(self, grad_x=True, grad_y=True):
        super(MaximumGrad, self).__init__()
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.select = P.Select()
        self.greater_equal = P.GreaterEqual()
        self.zeros_like = P.ZerosLike()
        self.sub = P.Sub()

    def construct(self, x, y, dout):
        cmp_result = self.greater_equal(x, y)
        dx = self.select(cmp_result, dout, self.zeros_like(dout))
        dy = self.select(cmp_result, self.zeros_like(dout), dout)

        outs = []
        if self.grad_x and self.grad_y:
            outs = [dx, dy]
        elif self.grad_x and not self.grad_y:
            outs = dx
        elif self.grad_y and not self.grad_x:
            outs = dy
        return outs

class MinimumGrad(Composite):
    def __init__(self, grad_x=True, grad_y=True):
        super(MinimumGrad, self).__init__()
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.select = P.Select()
        self.less_equal = P.LessEqual()
        self.zeros_like = P.ZerosLike()
        self.sub = P.Sub()

    def construct(self, x, y, dout):
        cmp_result = self.less_equal(x, y)
        dx = self.select(cmp_result, dout, self.zeros_like(dout))
        dy = self.select(cmp_result, self.zeros_like(dout), dout)

        outs = []
        if self.grad_x and self.grad_y:
            outs = [dx, dy]
        elif self.grad_x and not self.grad_y:
            outs = dx
        elif self.grad_y and not self.grad_x:
            outs = dy
        return outs

class Net(Cell):
    def __init__(self, maximum=True):
        super(Net, self).__init__()
        self.maximum = maximum
        if self.maximum:
            self.grad = MaximumGrad()
        else:
            self.grad = MinimumGrad()

    def construct(self, x, y, z):
        return self.grad(x, y, z)

def vm_impl(x, y, z, maximum=True, grad_x=True, grad_y=True):
    if maximum:
        dx = np.where(x >= y, z, 0)
        dy = np.where(x >= y, 0, z)
    else:
        dx = np.where(x <= y, z, 0)
        dy = np.where(x <= y, 0, z)
    outs = []
    if grad_x and grad_y:
        outs = [dx, dy]
    elif grad_x and not grad_y:
        outs = dx
    elif grad_y and not grad_x:
        outs = dy
    return outs

# composite not inline funcGraph
def test_maximum_grad():
    grad_x = True
    grad_y = True
    x = np.random.normal(0, 1, [32]).astype(np.float32)
    y = np.random.normal(0, 1, [32]).astype(np.float32)
    z = np.random.normal(0, 1, [32]).astype(np.float32)
    net = Net()

    res = net(Tensor(x), Tensor(y), Tensor(z))
    vm_res = vm_impl(x, y, z)
    print("=================maximum_grad test======================")
    print("x:\n{}".format(x))
    print("y:\n{}".format(y))
    print("res:\n{}".format(res))
    print("vm_res:\n{}".format(vm_res))
    print("=======================================")

def test_minimum_grad():
    grad_x = True
    grad_y = True
    x = np.random.normal(0, 1, [32]).astype(np.float32)
    y = np.random.normal(0, 1, [32]).astype(np.float32)
    z = np.random.normal(0, 1, [32]).astype(np.float32)
    net = Net(False)

    res = net(Tensor(x), Tensor(y), Tensor(z))
    vm_res = vm_impl(x, y, z, False)
    print("=================minimum_grad test======================")
    print("x:\n{}".format(x))
    print("y:\n{}".format(y))
    print("res:\n{}".format(res))
    print("vm_res:\n{}".format(vm_res))
    print("=======================================")

#test_maximum_grad()
test_minimum_grad()
