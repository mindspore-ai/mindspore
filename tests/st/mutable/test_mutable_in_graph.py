# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common import mutable
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


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
def test_cal_mutable_tensor():
    """
    Feature: Support mutable in graph.
    Description: Get the matmul result for one tensor defined in graph which is set mutable.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x):
            y = mutable(Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))
            out = self.matmul(x, y)
            return out

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.y = mutable(Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))

        def construct(self, x):
            out = self.matmul(x, self.y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    net = Net()
    output = net(x)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    expect_output = P.MatMul()(p, q)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    net = Net1()
    output = net(x)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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

        def construct(self, x, y):
            out = self.matmul(x, y)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tensor input defined outside the graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_tuple_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to tuple tensor input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z[0]
            y = z[1]
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, tuple)
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_list_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to list tensor input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z[0]
            y = z[1]
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, list)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_tuple_or_list_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tuple or list tensor input defined outside graph which is
                 set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z[0]
            y = z[1]
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    x = (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    assert isinstance(output, tuple)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)
    x = [Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)]
    output = grad_net(x)
    assert isinstance(output, list)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_list_and_tuple_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to list and tuple nested tensor input defined in graph which is
                 set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z[0][0]
            y = z[1]
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable([(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                          Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                         Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable([(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                               Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    grad_net = GradNetWrtX(Net())
    output = grad_net()
    assert isinstance(output, list)
    expect = [(np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
               np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    assert isinstance(output, list)
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_const_dict_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to dict tensor input defined in graph which is set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z['a']
            y = z['b']
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                              'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    grad_net = GradNetWrtX(Net())
    output = grad_net()
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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

        def construct(self, z):
            x = z['a']
            y = z['b']
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    x = {'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)}
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert isinstance(output, dict)
    assert len(output.keys()) == 2
    assert compare(output['a'], expect[0])
    assert compare(output['b'], expect[1])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_dict_and_tuple_tensor_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict tuple nested tensor input defined in graph which is
                 set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z['a'][0]
            y = z['b']
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self):
            x = mutable({'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                               Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    class GradNetWrtX1(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX1, self).__init__()
            self.net = net
            self.grad_op = GradOperation()
            self.x = mutable({'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                                    Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
                              'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)})

        def construct(self):
            gradient_function = self.grad_op(self.net)
            return gradient_function(self.x)

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    grad_net = GradNetWrtX(Net())
    output = grad_net()
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
    assert isinstance(output, tuple)
    expect = [(np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
               np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    grad_net = GradNetWrtX1(Net())
    output = grad_net()
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
    assert isinstance(output, tuple)
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_const_dict_and_tuple_tensor_arg_to_mutable():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const dict tuple nested tensor input defined outside graph which is
                 set mutable.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, z):
            x = z['a'][0]
            y = z['b']
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(mutable(x))

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    x = {'a': (Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
               Tensor([[0.5, 0.6, 4.0], [1.2, 1.3, 1.1]], dtype=mstype.float32)),
         'b': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)}
    grad_net = GradNetWrtX(Net())
    output = grad_net(x)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
    assert isinstance(output, tuple)
    expect = [(np.array([[1.4100001, 1.5999999, 6.6],
                         [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
               np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert compare(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_mutable_in_primal():
    """
    Feature: Support mutable in graph.
    Description: Get gradient with respect to const tensor input defined outside the graph which is set mutable
                 and uses mutable in primal graph.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(mutable(x), y)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mutable_empty_list():
    """
    Feature: Support mutable in graph.
    Description: Support empty list with mutable.
    Expectation: Get the correct gradients.
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
