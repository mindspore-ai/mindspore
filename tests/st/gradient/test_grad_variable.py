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
"""test getting gradient of Variable"""
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import Parameter, Variable


def compare(a, b):
    if isinstance(a, (list, tuple)):
        for aa, bb in zip(a, b):
            if not compare(aa, bb):
                return False
        return True

    return np.allclose(a.asnumpy(), b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_tuple_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to tuple tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0]
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_list_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to list tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0]
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_dict_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to dict tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t['a']
            y = t['b']
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                  'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_tuple_tuple_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested tuple tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0][0]
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable(((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                   Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_tuple_list_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested tuple and list tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0][0]
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable(([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                   Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)],
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_list_tuple_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested list and tuple tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0][0]
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable([(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                   Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_tuple_dict_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested tuple and dict tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0]['a']
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable(({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                   'b': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)},
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_dict_tuple_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested dict and tuple tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t['a'][0]
            y = t['b']
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable({'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                        Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                  'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_list_dict_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested list and dict tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t[0]['a']
            y = t[1]
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable([{'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                   'b': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)},
                  Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_variable_dict_list_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to nested dict and list tensor input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, t):
            x = t['a'][0]
            y = t['b']
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    t = Variable({'a': [Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                        Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)],
                  'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)
