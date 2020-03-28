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
import numpy as np
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
import mindspore.ops.composite as C
from mindspore.common.api import _executor
from mindspore.common.parameter import ParameterTuple
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)

def test_net_vargs_expand():
    class AddNet(Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.w = Parameter(Tensor(np.ones((3, 4, 5), np.float32)), "w2", requires_grad=True)
        def construct(self, x, y):
            return x + y
    x = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    sens = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    net = AddNet()
    out = C.grad_all_with_sens(net, net.trainable_params())(x, y, sens)

class VarNet(Cell):
    def __init__(self, net):
        super(VarNet, self).__init__()
        self.b = Parameter(Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b", requires_grad=True)
        self.w = Parameter(Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "w", requires_grad=True)
        self.net = net
    def construct(self, *args):
        return self.net(*args)*self.w + self.b
    
class SecondNet(Cell):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.b2 = Parameter(Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b2", requires_grad=True)
    def construct(self, *args):
        res = args[0] + args[1]
        return res + self.b2
def test_all_var_args_grad_with_sens():
    """"test grad_by_list_with_sens with all var args input"""
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
        def construct(self, *inputs):
            return C.grad_by_list_with_sens(self.net, self.weights)(*inputs)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(1.0, dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y, sens)

def test_grad_list_var_args():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
        def construct(self, *inputs):
            return C.grad_by_list(self.net, self.weights)(*inputs)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y)

def test_grad_all_var_args():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
        def construct(self, *inputs):
            return C.grad_all(self.net)(*inputs)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y)

def test_grad_all_var_args_with_sens():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
        def construct(self, *inputs):
            return C.grad_all_with_sens(self.net)(*inputs)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(1.0, dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y, sens)

def test_grad_var_args_with_sens():
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = ParameterTuple(net.trainable_params())
            self.net = net
        def construct(self, *inputs):
            return C.grad_with_sens(self.net)(*inputs)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(1.0, dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y, sens)

def test_var_args_grad():
    class VarNet(Cell):
        def __init__(self, net):
            super(VarNet, self).__init__()
            self.b = Parameter(Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b", requires_grad=True)
            self.net = net
        def construct(self, *args):
            return self.net(*args) + self.b
        
    class SecondNet(Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
            self.b2 = Parameter(Tensor(np.ones([3, 4, 5]), dtype=mstype.float32), "b2", requires_grad=True)
        def construct(self, *args):
            res = args[0] + args[1]
            return res + self.b2
    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())
        def construct(self, x, y, sens):
            return C.grad_by_list_with_sens(self.net, self.weights)(x, y, sens)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    sens = Tensor(1.0, dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y, sens)


def test_var_args_positional():
    """"test grad_all with var args in inner graph"""
    class VarNet(Cell):
        def __init__(self, net):
            super(VarNet, self).__init__()
            self.net = net
        def construct(self, x, y):
            return self.net(x, y)*x

    class SecondNet(Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
        def construct(self, *args):
            return args[0] + args[1]

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())
        def construct(self, x, y):
            return C.grad_all(self.net)(x, y)
    x = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    y = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
    net = VarNet(SecondNet())
    grad_net = GradNet(net)
    out = grad_net(x, y)
