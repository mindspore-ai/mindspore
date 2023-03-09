# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test grad ops """
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import jit, jit_class
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore import ParameterTuple, Parameter

one = Tensor([1], mstype.int32)
zero = Tensor([0], mstype.int32)


@jit
def local_pow(x, n):
    r = one
    while n > zero:
        n = n - one
        r = r * x
    return r


def test_pow_first_order():
    """
    Feature: pow first order test.
    Description: pow first order test.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    n = Tensor([3], mstype.int32)
    grad = ops.GradOperation()
    grad_net = grad(local_pow)
    res = grad_net(x, n)
    assert res == 75

def test_pow_second_order():
    """
    Feature: pow second order test.
    Description: pow second order test.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    n = Tensor([3], mstype.int32)
    grad = ops.GradOperation()
    grad_net = grad(local_pow)
    sec_grad_net = grad(grad_net)
    res = sec_grad_net(x, n)
    assert res == 30


def test_high_order_with_params():
    """
    Feature: second order test.
    Description: second order test with weight.
    Expectation: return expected value.
                 net: (x ** 3) * (self.weight ** 3)
                 first_grad: 3 * (x ** 2) * (self.weight ** 3)
                 second_grad: 3 * (x ** 2) * 3 * (self.weight * 2)
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.weight = Parameter(Tensor(Tensor([2], mstype.int32)), name="weight", requires_grad=True)
            self.n = Tensor([3], mstype.int32)

        def construct(self, x):
            r = one
            n = self.n
            while n > zero:
                n = n - one
                r = r * x * self.weight
            return r


    class Grad(nn.Cell):
        def __init__(self, network):
            super(Grad, self).__init__()
            self.grad = ops.GradOperation(get_all=True, sens_param=False)
            self.network = network

        def construct(self, x):
            output = self.grad(self.network)(x)
            return output


    class GradSec(nn.Cell):
        def __init__(self, network):
            super(GradSec, self).__init__()
            self.grad = ops.GradOperation(get_by_list=True, sens_param=False)
            self.network = network
            self.params = ParameterTuple(network.trainable_params())

        def construct(self, x):
            output = self.grad(self.network, self.params)(x)
            return output


    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    expected = Tensor([900], mstype.int32)
    net = Net()
    first_grad = Grad(net)
    second_grad = GradSec(first_grad)
    assert second_grad(x) == (expected,)


def test_reftoembed_with_two_weights():
    """
    Feature: RefToEmbed can be properly be evaluated.
    Description: Multiple weights with same shape and type can be evaluatd properly
                 even RewriterBeforeOptA (one more round of Renormalize) takes effect.
    Expectation: return expected value.
    """
    @jit_class
    class SimpleData:
        def __init__(self, a):
            self.a = a

        def get_data(self):
            return self.a

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.weight = Parameter(Tensor([2], mstype.int32), name="weight", requires_grad=True)
            self.bias = Parameter(Tensor([3], mstype.int32), name="bias", requires_grad=True)

        def construct(self, x):
            simple = SimpleData(x)
            r = self.weight * self.bias * simple.get_data()
            return r

    class Grad(nn.Cell):
        def __init__(self, network):
            super(Grad, self).__init__()
            self.grad = ops.GradOperation(get_by_list=True, sens_param=False)
            self.network = network
            self.params = ParameterTuple(network.trainable_params())

        def construct(self, x):
            output = self.grad(self.network, self.params)(x)
            return output

    context.set_context(mode=context.GRAPH_MODE)
    expected_weight_grad = Tensor([15], mstype.int32)
    expected_bias_grad = Tensor([10], mstype.int32)
    net = Net()
    first_grad = Grad(net)
    assert first_grad(5) == (expected_weight_grad, expected_bias_grad)
