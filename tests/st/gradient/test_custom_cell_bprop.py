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
""" test_cell_bprop """
import numpy as np
import pytest

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Parameter, ParameterTuple
from mindspore import context
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

grad_all = C.GradOperation(get_all=True)


class MulAdd(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        # In this test case, The user defined bprop is wrong defined purposely to distinguish from ad result
        return 2 * dout, 2 * y


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_mul_add():
    mul_add = MulAdd()
    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    assert grad_all(mul_add)(x, y) == (2, 4)


class InlineMulADD(nn.Cell):
    def __init__(self):
        super(InlineMulADD, self).__init__()
        self.mul_add = MulAdd()
        self.param = 2

    def construct(self, x, y):
        return self.mul_add(x, y) + x + self.param * y


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_inline_mul_add():
    inline_mul_add = InlineMulADD()
    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    assert grad_all(inline_mul_add)(x, y) == (3, 6)


class WithParameter(nn.Cell):
    def __init__(self):
        super(WithParameter, self).__init__()
        self.param1 = Parameter(1, 'param1')
        self.param2 = Parameter(2, 'param2')

    def construct(self, x, y):
        return self.param1 * self.param2 * x + y

    def bprop(self, x, y, out, dout):
        # In this test case, The user defined bprop is wrong defined purposely to distinguish from ad result
        return self.param1 * self.param2 * dout, 2 * y


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_param():
    with_param = WithParameter()
    with pytest.raises(RuntimeError):
        grad_all(with_param)(1, 2)


class WithNoBprop(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_no_bprop():
    with_no_bprop = WithNoBprop()
    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    assert grad_all(with_no_bprop)(x, y) == (2, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_in_bprop_1():
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super(GradInBprop_1, self).__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super(GradInBprop_2, self).__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0], grads[1]

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super(GradInBprop_3, self).__init__()
            self.f = GradInBprop_2()

        def construct(self, x, y):
            return self.f(x, y)

    grad_in_bprop = GradInBprop_3()
    grads = grad_all(grad_in_bprop)(Tensor(np.ones([2, 2]).astype(np.float32)),
                                    Tensor(np.ones([2, 2]).astype(np.float32)))
    assert (grads[0].asnumpy() == np.ones([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.zeros([2, 2]).astype(np.float32)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_in_bprop_2():
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super(GradInBprop_1, self).__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

        def bprop(self, x, y, out, dout):
            return x * y, y + x

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super(GradInBprop_2, self).__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0], grads[1]

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super(GradInBprop_3, self).__init__()
            self.f = GradInBprop_2()

        def construct(self, x, y):
            return self.f(x, y)

    grad_in_bprop = GradInBprop_3()
    grads = grad_all(grad_in_bprop)(Tensor(np.ones([2, 2]).astype(np.float32)),
                                    Tensor(np.ones([2, 2]).astype(np.float32)))
    assert (grads[0].asnumpy() == np.ones([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([[2, 2], [2, 2]]).astype(np.float32)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_in_bprop_3():
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super(GradInBprop_1, self).__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super(GradInBprop_2, self).__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0], grads[1]

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super(GradInBprop_3, self).__init__()
            self.f = GradInBprop_2()

        def construct(self, x, y):
            return self.f(x, y)

        def bprop(self, x, y, out, dout):
            return x + y + y + out[0], x + x + y + y + dout[0]

    grad_in_bprop = GradInBprop_3()
    grads = grad_all(grad_in_bprop)(Tensor(np.ones([2, 2]).astype(np.float32)),
                                    Tensor(np.ones([2, 2]).astype(np.float32)))
    assert (grads[0].asnumpy() == np.array([[4, 4], [4, 4]]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([[5, 5], [5, 5]]).astype(np.float32)).all()


class OneInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.ReLU()

    def construct(self, x):
        return self.op(x)

    def bprop(self, x, out, dout):
        return (5 * x,)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_one_input_bprop():
    net = OneInputBprop()
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    grad = grad_all(net)(input1)
    assert (grad[0].asnumpy() == np.array([5, 5]).astype(np.float32)).all()


class TwoInput(nn.Cell):
    def construct(self, x, y):
        return x * y


class InlineBpropTwoInput(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f = TwoInput()

    def construct(self, x, y):
        return self.f(x, y), grad_all(self.f)(x, y)

    def bprop(self, x, y, out, dout):
        grads = grad_all(self.f)(x, y)
        return grads[0] * 2, grads[1] * 2


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_inline_bprop_two_input():
    net = InlineBpropTwoInput()
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    input2 = Tensor(np.ones([2, 2]).astype(np.float32))
    grads = grad_all(net)(input1, input2)
    assert (grads[0].asnumpy() == np.array([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([2, 2]).astype(np.float32)).all()
    assert len(grads) == 2


class TwoInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Mul()

    def construct(self, x, y):
        return self.op(x, y)

    def bprop(self, x, y, out, dout):
        return 5 * x, 8 * y


class TwoInputWithParameter(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Mul()
        self.inputdata = Parameter(initializer(1, (2, 2), mstype.float32), name="global_step")

    def construct(self, x, y):
        x = self.inputdata + x
        return self.op(x, y)


class TwoInputWithOnlyInitParameterBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Mul()
        self.inputdata = Parameter(initializer(1, (2, 2), mstype.float32), name="global_step")

    def construct(self, x, y):
        return self.op(x, y)

    def bprop(self, x, y, out, dout):
        return 5 * x, 8 * y


class InlineMutilTwoInputParameterCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = TwoInputBprop()
        self.f2 = TwoInput()
        self.f3 = TwoInputWithParameter()
        self.f4 = TwoInputWithOnlyInitParameterBprop()

    def construct(self, x, y):
        output = self.f1(x, y) + self.f2(x, y) + self.f3(x, y) + self.f4(x, y)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_inline_bprop_multi_input():
    net = InlineMutilTwoInputParameterCell()
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    input2 = Tensor(np.ones([2, 2]).astype(np.float32))
    net.init_parameters_data()
    grads = grad_all(net)(input1, input2)
    assert (grads[0].asnumpy() == np.array([[12, 12], [12, 12]]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([[19, 19], [19, 19]]).astype(np.float32)).all()
    assert len(grads) == 2


class MulAddWithParam(nn.Cell):
    def __init__(self):
        super(MulAddWithParam, self).__init__()
        self.mul_add = MulAdd()
        self.param = Parameter(Tensor(np.array([[3, 2]], np.float32)), 'param')

    def construct(self, x):
        return self.mul_add(self.param, x)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_refkey_bprop():
    grad_by_list = C.GradOperation(get_all=True, get_by_list=True)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network
            self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

        def construct(self, x):
            weights = self.weights
            grads = grad_by_list(self.network, weights)(x)
            return grads

    network = GradWrap(MulAddWithParam())
    input_data = Tensor(np.array([2, 2], np.float32))
    grads = network(input_data)
    assert (grads[0][0].asnumpy() == np.array([4, 4]).astype(np.float32)).all()
    assert (grads[1][0].asnumpy() == np.array([2, 2]).astype(np.float32)).all()


class MulAddWithWrongOutputNum(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        return (2 * dout,)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_mul_add_with_wrong_output_num():
    context.set_context(check_bprop=True)
    mul_add = MulAddWithWrongOutputNum()
    with pytest.raises(TypeError):
        grad_all(mul_add)(1, 2)


class MulAddWithWrongOutputType(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        return 2 * dout, 2


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_mul_add_with_wrong_output_type():
    context.set_context(check_bprop=True)
    mul_add = MulAddWithWrongOutputType()
    with pytest.raises(TypeError):
        grad_all(mul_add)(1, Tensor(np.ones([2, 2])))


class MulAddWithWrongOutputShape(nn.Cell):
    def __init__(self):
        super(MulAddWithWrongOutputShape, self).__init__()
        self.ones = Tensor(np.ones([2,]))

    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        return 2, self.ones


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_grad_mul_add_with_wrong_output_shape():
    context.set_context(check_bprop=True)
    mul_add = MulAddWithWrongOutputShape()
    with pytest.raises(TypeError):
        grad_all(mul_add)(1, Tensor(np.ones([2, 2])))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_forward_with_parameter():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x + x
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    out = GradNet(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_forward_with_parameter_in_sub_cell():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x + x
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    out = GradNet(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_forward_with_parameter_in_sub_cell_get_by_list():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs and Parameters when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x + x
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.params = ParameterTuple(net.trainable_params())
            self.grad_op = C.GradOperation(get_by_list=True, get_all=True)

        def construct(self, x, y):
            grad_f = self.grad_op(self.net, self.params)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    out = GradNet(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    expect_dz = np.array([0.0]).astype(np.float32)
    assert np.allclose(out[0][0].asnumpy(), expect_dx)
    assert np.allclose(out[0][1].asnumpy(), expect_dy)
    assert np.allclose(out[1][0].asnumpy(), expect_dz)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_forward_with_parameter():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x + x
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    out = GradNet(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_forward_with_parameter_in_sub_cell():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x + x
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            grad_f = grad_all(self.net)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    out = GradNet(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_forward_with_parameter_in_sub_cell_get_by_list():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs and Parameters when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            dx = x + x
            dy = y + y
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.params = ParameterTuple(net.trainable_params())
            self.grad_op = C.GradOperation(get_by_list=True, get_all=True)

        def construct(self, x, y):
            grad_f = self.grad_op(self.net, self.params)
            return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    out = GradNet(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    expect_dz = np.array([0.0]).astype(np.float32)
    assert np.allclose(out[0][0].asnumpy(), expect_dx)
    assert np.allclose(out[0][1].asnumpy(), expect_dy)
    assert np.allclose(out[1][0].asnumpy(), expect_dz)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dde_self_define_cell_output_not_use():
    """
    Feature: Custom cell bprop
    Description: Fprop output[1] only used by bprop, it should not erased by dde.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class SelfDefineCell(ms.nn.Cell):
        def construct(self, x):
            return x + 1, x + 2

        def bprop(self, x, out, dout):
            return (out[1],)

    class ForwardNet(ms.nn.Cell):
        def __init__(self):
            super(ForwardNet, self).__init__()
            self.self_defined_cell = SelfDefineCell()

        def construct(self, x):
            # keep out1 not used in fprop.
            out0, _ = self.self_defined_cell(x)
            return out0

    class TestNet(ms.nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()
            self.forward_net = ForwardNet()
            self.grad_op = ms.ops.GradOperation(get_all=True)

        def construct(self, x):
            grad_out = self.grad_op(self.forward_net)(x)
            return grad_out

    net = TestNet()
    x_input = ms.Tensor([1])
    out = net(x_input)
    assert out[0] == ms.Tensor([3])
