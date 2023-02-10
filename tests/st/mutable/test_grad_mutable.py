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
"""test getting gradient of mutable input"""
import os
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context, Parameter, jit
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common import mutable


def compare(a, b):
    if isinstance(a, (list, tuple)):
        if not a and b:
            return False
        for aa, bb in zip(a, b):
            if not compare(aa, bb):
                return False
        return True

    return np.allclose(a.asnumpy(), b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_mutable_tuple_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
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
def test_grad_mutable_list_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
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
def test_grad_mutable_dict_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                 'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, dict)
    assert len(output.keys()) == 2
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output['a'], expect[0])
    assert compare(output['b'], expect[1])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_mutable_tuple_tuple_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable(((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
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
def test_grad_mutable_tuple_list_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable(([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
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
def test_grad_mutable_list_tuple_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable([(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
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
def test_grad_mutable_tuple_dict_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable(({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                  'b': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)},
                 Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
    output = GradNetWrtX(Net())(t)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], dict)
    assert len(output[0].keys()) == 2
    assert compare(output[0]['a'], expect[0][0])
    assert compare(output[0]['b'], expect[0][1])
    assert compare(output[1], expect[1])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_mutable_dict_tuple_tensor():
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

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    context.set_context(mode=context.GRAPH_MODE)
    t = mutable({'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                       Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                 'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
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
def test_grad_mutable_list_dict_tensor():
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

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable([{'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                  'b': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)},
                 Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
    output = GradNetWrtX(Net())(t)
    expect = [[np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32), np.array([[0, 0, 0],
                                                                                     [0, 0, 0]]).astype(np.float32)],
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], dict)
    assert len(output[0].keys()) == 2
    assert compare(output[0]['a'], expect[0][0])
    assert compare(output[0]['b'], expect[0][1])
    assert compare(output[1], expect[1])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_mutable_dict_list_tensor():
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

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    context.set_context(mode=context.GRAPH_MODE)
    t = mutable({'a': [Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                       Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)],
                 'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
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
def test_grad_mutable_tuple_tensor_jit_function():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to tuple tensor input.
    Expectation: Get the correct gradients.
    """

    @jit
    def net(t):
        x = t[0]
        y = t[1]
        out = P.MatMul()(x, y)
        return out

    z = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                 Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))

    context.set_context(mode=context.GRAPH_MODE)
    output = GradOperation()(net)(z)
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
def test_grad_mutable_list_tensor_jit_function():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to tuple tensor input.
    Expectation: Get the correct gradients.
    """

    @jit
    def net(t):
        x = t[0]
        y = t[1]
        out = P.MatMul()(x, y)
        return out

    z = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                 Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])

    context.set_context(mode=context.GRAPH_MODE)
    output = GradOperation()(net)(z)
    assert isinstance(output, tuple)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_grad_mutable_unused_tuple_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to tuple tensor input which is unused by backend nodes.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub = P.Sub()

        def construct(self, t):
            x1 = t[0]
            x2 = t[1]
            output = x1 + self.sub(x1, x2)
            return x1, x2, output

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable((Tensor([[4.0, 6.0, 6.0], [4.0, 6.0, 6.0]], dtype=mstype.float32),
                 Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=mstype.float32),
                 Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=mstype.float32)))
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [np.array([[3., 3., 3.],
                        [3., 3., 3.]]).astype(np.float32),
              np.array([[0., 0., 0.],
                        [0., 0., 0.]]).astype(np.float32),
              np.array([[0., 0., 0.],
                        [0., 0., 0.]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_grad_mutable_unused_list_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to list tensor input which is unused by backend nodes.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub = P.Sub()

        def construct(self, t):
            x1 = t[0]
            t[1] = Tensor([[3.0, 6.0, 6.0], [3.0, 6.0, 6.0]], dtype=mstype.float32)
            x2 = t[1]
            output = x1 + self.sub(x1, x2)
            return x1, x2, output

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable([Tensor([[4.0, 6.0, 6.0], [4.0, 6.0, 6.0]], dtype=mstype.float32),
                 Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=mstype.float32),
                 Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=mstype.float32)])
    output = GradNetWrtX(Net())(t)
    assert isinstance(output, tuple)
    expect = [np.array([[3., 3., 3.],
                        [3., 3., 3.]]).astype(np.float32),
              np.array([[0., 0., 0.],
                        [0., 0., 0.]]).astype(np.float32),
              np.array([[0., 0., 0.],
                        [0., 0., 0.]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_grad_mutable_unused_dict_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to dict tensor input which is unused by backend nodes.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub = P.Sub()

        def construct(self, t):
            x1 = t['x1']
            t['x2'] = Tensor([[3.0, 6.0, 6.0], [3.0, 6.0, 6.0]], dtype=mstype.float32)
            x2 = t['x2']
            output = x1 + self.sub(x1, x2)
            return x1, x2, output

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, z):
            gradient_function = self.grad_op(self.net)
            return gradient_function(z)

    context.set_context(mode=context.GRAPH_MODE)
    t = mutable({'x1': Tensor([[4.0, 6.0, 6.0], [4.0, 6.0, 6.0]], dtype=mstype.float32),
                 'x2': Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=mstype.float32),
                 'x3': Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(t)
    expect = [np.array([[3., 3., 3.],
                        [3., 3., 3.]]).astype(np.float32),
              np.array([[0., 0., 0.],
                        [0., 0., 0.]]).astype(np.float32),
              np.array([[0., 0., 0.],
                        [0., 0., 0.]]).astype(np.float32)]
    assert isinstance(output, dict)
    assert len(output.keys()) == 3
    assert compare(output['x1'], expect[0])
    assert compare(output['x2'], expect[1])
    assert compare(output['x3'], expect[2])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_mutable_single_element_dict_tensor():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the dict tensor input which has only one element.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, t):
            y = t['a']
            x = x * self.z
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation(get_all=True)

        def construct(self, x, t):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, t)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = mutable({'a': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
    output = GradNetWrtX(Net())(x, y)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert compare(output[0], expect[0])
    assert isinstance(output[1], dict)
    assert len(output[1].keys()) == 1
    assert compare(output[1]['a'], expect[1])
