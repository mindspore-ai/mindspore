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
"""test mutable or constant tensor feature"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common import mutable
from mindspore import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cal_constant_tensor():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get the matmul result for two constant tensor.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    net = Net()
    output = net(x, y)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    expect_output = net(p, q)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cal_constant_tensor_jit_function():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get the matmul result for two constant tensor in @jit decorated function.
    Expectation: Get the correct result.
    """

    @jit
    def net(x, y):
        out = P.MatMul()(x, y)
        return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    output = net(x, y)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    expect_output = net(p, q)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_const_arg_tensor_to_mutable():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to constant tensor input.
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

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    grad_net = GradNetWrtX(Net())
    # mutable api
    output = grad_net(mutable(x), y)
    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    # tensor set_const_arg api
    x.set_const_arg(False)
    output = grad_net(x, y)
    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_function_grad_const_arg_tensor_to_mutable():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to constant tensor input for the function decorated with jit.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    @jit
    def fn(x, y):
        net = Net()
        grad_op = GradOperation()
        return grad_op(net)(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32, const_arg=True)
    # mutable api
    output = fn(mutable(x), y)
    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    # tensor set_const_arg api
    x.set_const_arg(False)
    output = fn(x, y)
    expect_output = np.array([[1.4100001, 1.5999999, 6.6],
                              [1.4100001, 1.5999999, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
