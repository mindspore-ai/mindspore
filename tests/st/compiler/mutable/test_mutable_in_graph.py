# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""test the feature of mutable in graph"""
import os
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common import mutable
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, grad_for_scalar=True)


def compare(a, b):
    if isinstance(a, (list, tuple)):
        if not a and b:
            return False
        for aa, bb in zip(a, b):
            if not compare(aa, bb):
                return False
        return True

    return np.allclose(a.asnumpy(), b)


def seq_compare(a, b):
    if isinstance(a, (list, tuple)):
        if not a and b:
            return False
        for aa, bb in zip(a, b):
            if not seq_compare(aa, bb):
                return False
        return True
    if isinstance(a, Tensor):
        if isinstance(b, Tensor):
            return np.allclose(a.asnumpy(), b.asnumpy())
        return np.allclose(a.asnumpy(), b)
    if isinstance(a, float):
        return np.allclose(a, b)
    return a == b


def dict_compare(a, b):
    if isinstance(a, dict):
        if not a and b:
            return False
        for aa, bb in zip(a.values(), b.values()):
            if not seq_compare(aa, bb):
                return False
        return True
    if isinstance(a, Tensor):
        if isinstance(b, Tensor):
            return np.allclose(a.asnumpy(), b.asnumpy())
        return np.allclose(a.asnumpy(), b)
    if isinstance(a, float):
        return np.allclose(a, b)
    return a == b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_bool():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for bool which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable(True)

        def construct(self, x, y):
            z = mutable(False)
            out = (x, mutable(y), z, self.w)
            return out


    x = mutable(True)
    y = False
    net = Net()
    output = net(x, y)
    assert output[0]
    assert not output[1]
    assert not output[2]
    assert output[3]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_scalar():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for scalar which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable(1)

        def construct(self, x, y):
            z = mutable(2)
            out = x + mutable(y) + z + self.w
            return out


    x = mutable(3)
    y = 4
    net = Net()
    output = net(x, y)
    assert output == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_tuple_scalar():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for tuple which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable((1, 2, 3))

        def construct(self, x, y):
            z = mutable((2, 3, 4))
            out = x[0] + mutable(y)[0] + z[0] + self.w[0]
            return out


    x = mutable((3, 4, 5))
    y = (4, 5, 6)
    net = Net()
    output = net(x, y)
    assert output == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_dyn_tuple_scalar():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for dynamic tuple which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable((1, 2, 3), True)

        def construct(self, x, y):
            z = mutable((2, 3, 4), True)
            out = x[0] + mutable(y, True)[0] + z[0] + self.w[0]
            return out


    x = mutable((3, 4, 5), True)
    y = (4, 5, 6)
    net = Net()
    output = net(x, y)
    assert output == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_list_scalar():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for list which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable([1, 2, 3])

        def construct(self, x, y):
            z = mutable([2, 3, 4])
            out = x[0] + mutable(y)[0] + z[0] + self.w[0]
            return out


    x = mutable([3, 4, 5])
    y = [4, 5, 6]
    net = Net()
    output = net(x, y)
    assert output == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_dyn_list_scalar():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for dynamic list which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable([1, 2, 3], True)

        def construct(self, x, y):
            z = mutable([2, 3, 4], True)
            out = x[0] + mutable(y, True)[0] + z[0] + self.w[0]
            return out


    x = mutable([3, 4, 5], True)
    y = [4, 5, 6]
    net = Net()
    output = net(x, y)
    assert output == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_dict_any():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for dict which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable({'a': 1, 'b': Tensor([0]), 'c': (3, 4, 5), 'd': [1, 3, 5]})

        def construct(self, x, y):
            z = mutable({'a': 1, 'b': Tensor([0]), 'c': (3, 4, 5), 'd': [1, 3, 5]})
            out = x['a'] + mutable({'a': 1, 'b': Tensor([0]), 'c': (3, 4, 5), 'd': [1, 3, 5]})['a'] + \
                  z['c'][2] + self.w['c'][0]
            return out


    x = mutable({'a': 1, 'b': Tensor([0]), 'c': (3, 4, 5), 'd': [1, 3, 5]})
    y = {'a': 1, 'b': Tensor([0]), 'c': (3, 4, 5), 'd': [1, 3, 5]}
    net = Net()
    output = net(x, y)
    assert output == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cal_mutable_tensor():
    """
    Feature: Support mutable in graph.
    Description: Get the operation result for Tensor which is set mutable or not.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = mutable(Tensor([1]))

        def construct(self, x, y):
            z = mutable(Tensor([2]))
            out = x + mutable(y) + z + self.w
            return out


    x = mutable(Tensor([3]))
    y = Tensor([4])
    net = Net()
    output = net(x, y)
    assert output == Tensor([10])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_scalar_mul_grad_first():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the first scalar input.
    Expectation: Get the correct gradient.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x * y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = mutable(2)
    output = GradNet(Net())(x, 3)
    assert output == 3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_scalar_mul_grad_all():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to all scalar inputs.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x * y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = mutable(2)
    y = mutable(3)
    output = GradNet(Net())(x, y)
    assert output == (3, 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_tuple_or_list_scalar_mul_grad():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the tuple or list scalar input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x[0] * x[1]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    x = mutable((2, 3))
    output = GradNet(Net())(x)
    assert output == (3, 2)

    x = mutable([2, 3])
    output = GradNet(Net())(x)
    assert output == [3, 2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dict_scalar_mul_grad():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the dict scalar input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x['a'] * x['b']

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    x = mutable({'a': 2, 'b': 3})
    output = GradNet(Net())(x)
    assert output == (3, 2)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dict_scalar_mul_grad_with_fallback():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the dict scalar input.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return x['a'] * x['b']

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    x = mutable({'a': 2, 'b': 3})
    output = GradNet(Net())(x)
    assert output == {'a': 3, 'b': 2}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dict_mix_scalar_mul_grad_all():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the mix scalar input including dict and tuple.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x['a'] * x['b'] * y[0]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    x = mutable({'a': 2, 'b': 3})
    y = mutable((4, 5))
    output = GradNet(Net())(x, y)
    assert output == ((12, 8), (6, 0))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dict_mix_scalar_mul_grad_all_with_fallback():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the mix scalar input including dict and tuple.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x['a'] * x['b'] * y[0]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = mutable({'a': 2, 'b': 3})
    y = mutable((4, 5))
    output = GradNet(Net())(x, y)
    assert output == ({'a': 12, 'b': 8}, (6, 0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to tensor input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, x, y):
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32))
            y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32))

        def construct(self):
            y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x, y)

    grad_net = GradNetWrtX(Net())
    output = grad_net()
    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tensor input defined
                 outside the graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, x, y):
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x), y)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert np.allclose(output.asnumpy(), expect_output)
    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert np.allclose(output.asnumpy(), expect_output)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_tuple_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to tuple tensor input defined
                 in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0]
            y = z[1]
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                            dtype=mstype.float32)))
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                                dtype=mstype.float32)))

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert compare(output, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_list_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to list tensor input defined
                 in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0]
            y = z[1]
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                            dtype=mstype.float32)])
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                                dtype=mstype.float32)])

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert compare(output, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_list_and_tuple_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to list and tuple nested tensor input
                 defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0][0]
            y = z[1]
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable([(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                          Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                            dtype=mstype.float32)])
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable(([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                               Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)],
                              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                                dtype=mstype.float32)))

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    expect = [(np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
               np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert compare(output, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_tuple_or_list_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tuple or list tensor input
                 defined outside graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0]
            y = z[1]
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))


    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]

    x = (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, tuple)
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, tuple)
    assert compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)

    x = [Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)]
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, list)
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, list)
    assert compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_grad_const_dict_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to dict tensor input defined in graph
                 which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a']
            y = z['b']
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                            dtype=mstype.float32)})
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                                dtype=mstype.float32)})

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, dict)
    assert len(output.keys()) == 2
    assert compare(output['a'], expect[0])
    assert compare(output['b'], expect[1])
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, dict)
    assert len(output.keys()) == 2
    assert compare(output['a'], expect[0])
    assert compare(output['b'], expect[1])


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_grad_const_dict_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict tensor input defined outside graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a']
            y = z['b']
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    x = {'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)}
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, dict)
    assert len(output.keys()) == 2
    assert compare(output['a'], expect[0])
    assert compare(output['b'], expect[1])

    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, dict)
    assert len(output.keys()) == 2
    assert compare(output['a'], expect[0])
    assert compare(output['b'], expect[1])
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_dict_and_tuple_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict tuple nested tensor
                 input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a'][0]
            y = z['b']
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable({'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                               Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                            dtype=mstype.float32)})
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable(
                {'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                       Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                 'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                    dtype=mstype.float32)})

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)


    expect = [(np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
               np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert compare(output, expect)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_dict_and_tuple_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict tuple nested tensor
                 input defined outside graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a'][0]
            y = z['b']
            out = self.matmul(x, y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    expect = [(np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
               np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    x = {'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
               Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)}
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, tuple)
    assert compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, tuple)
    assert compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_tuple_any_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to tuple any input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0]
            y = z[3][1]
            out = self.matmul(x, y) * self.w  + z[1] + z[2][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         1.3,
                         [2, 3, 4],
                         (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                            dtype=mstype.float32))))
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              1.3,
                              [2, 3, 4],
                              (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                                    dtype=mstype.float32))))

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)


    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32), 6, [0, 6, 0],
              (0, np.array([[1.7, 1.7, 1.7],
                            [1.9, 1.9, 1.9],
                            [1.5, 1.5, 1.5]]).astype(np.float32))]
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_list_any_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to list any input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0]
            y = z[3][1]
            out = self.matmul(x, y) * self.w  + z[1] + z[2][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         1.3,
                         [2, 3, 4],
                         (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                            dtype=mstype.float32))])
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              1.3,
                              [2, 3, 4],
                              (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], \
                                    dtype=mstype.float32))])

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)


    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32), 6, [0, 6, 0],
              (0, np.array([[1.7, 1.7, 1.7],
                            [1.9, 1.9, 1.9],
                            [1.5, 1.5, 1.5]]).astype(np.float32))]
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert seq_compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert seq_compare(output, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_const_tuple_or_list_any_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tuple or list any input defined
                 outside graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z[0]
            y = z[3][1]
            out = self.matmul(x, y) * self.w  + z[1] + z[2][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))


    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32), 6, [0, 6, 0],
              (0, np.array([[1.7, 1.7, 1.7],
                            [1.9, 1.9, 1.9],
                            [1.5, 1.5, 1.5]]).astype(np.float32))]

    x = (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         1.3,
         [2, 3, 4],
         (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)

    x = [Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         1.3,
         [2, 3, 4],
         (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))]
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, list)
    assert seq_compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, list)
    assert seq_compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_grad_const_dict_any_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to dict any input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a']
            y = z['d'][1]
            out = self.matmul(x, y) * self.w  + z['b'] + z['c'][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         'b': 1.3,
                         'c': [2, 3, 4],
                         'd': (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                                dtype=mstype.float32))})
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              'b': 1.3,
                              'c': [2, 3, 4],
                              'd': (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                                    dtype=mstype.float32))})
        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    expect = (np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32), 6, [0, 6, 0],
              (0, np.array([[1.7, 1.7, 1.7],
                            [1.9, 1.9, 1.9],
                            [1.5, 1.5, 1.5]]).astype(np.float32)))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_grad_const_dict_any_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict any input defined
                 outside graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a']
            y = z['d'][1]
            out = self.matmul(x, y) * self.w  + z['b'] + z['c'][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    expect = (np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32), 6, [0, 6, 0],
              (0, np.array([[1.7, 1.7, 1.7],
                            [1.9, 1.9, 1.9],
                            [1.5, 1.5, 1.5]]).astype(np.float32)))
    x = {'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         'b': 1.3,
         'c': [2, 3, 4],
         'd': (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
            dtype=mstype.float32))}
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, tuple)
    assert seq_compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_grad_const_dict_any_to_mutable_with_fallback():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to dict any input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a']
            y = z['d'][1]
            out = self.matmul(x, y) * self.w  + z['b'] + z['c'][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         'b': 1.3,
                         'c': [2, 3, 4],
                         'd': (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                                dtype=mstype.float32))})
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              'b': 1.3,
                              'c': [2, 3, 4],
                              'd': (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
                                    dtype=mstype.float32))})
        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    expect = {'a': np.array([[1.4100001, 1.5999999, 6.6],
                             [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              'b': 6,
              'c': [0, 6, 0],
              'd': (0, np.array([[1.7, 1.7, 1.7],
                                 [1.9, 1.9, 1.9],
                                 [1.5, 1.5, 1.5]]).astype(np.float32))}

    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, dict)
    assert dict_compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, dict)
    assert dict_compare(output, expect)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_grad_const_dict_any_arg_to_mutable_with_fallback():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict any input defined outside
                 graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, z):
            x = z['a']
            y = z['d'][1]
            out = self.matmul(x, y) * self.w  + z['b'] + z['c'][1]
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    expect = {'a': np.array([[1.4100001, 1.5999999, 6.6],
                             [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              'b': 6,
              'c': [0, 6, 0],
              'd': (0, np.array([[1.7, 1.7, 1.7],
                                 [1.9, 1.9, 1.9],
                                 [1.5, 1.5, 1.5]]).astype(np.float32))}
    x = {'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         'b': 1.3,
         'c': [2, 3, 4],
         'd': (1, Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
            dtype=mstype.float32))}
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, dict)
    assert dict_compare(output, expect)

    grad_net = GradNetWrtX1(Net())
    m_x = mutable(x)
    output = grad_net(m_x)
    assert isinstance(output, dict)
    assert dict_compare(output, expect)
    assert m_x.__ms_origin_object__ is not None
    assert id(m_x.__ms_origin_object__) == id(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_mutable_in_primal():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tensor input defined outside the graph
                 which is set mutable and uses mutable in primal graph.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.w = Parameter(Tensor(np.array([1.0], np.float32)), name='w')

        def construct(self, x, y):
            out = self.matmul(mutable(x), y) * self.w
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x), y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_empty_list():
    """
    Feature: Support mutable in graph.
    Description: Support empty list with mutable.
    Expectation: No Expectation.
    """
    class Net(nn.Cell):
        def construct(self, x, index):
            list_out = x[:index]
            return list_out

    x = mutable([])
    index = mutable(3)
    net = Net()
    out = net(x, index)
    assert out == []
