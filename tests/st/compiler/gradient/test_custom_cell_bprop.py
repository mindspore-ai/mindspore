# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import re

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import jit, nn
from mindspore import Parameter, ParameterTuple
from mindspore import context, mutable
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ops
from mindspore._extends import cell_attr_register
from mindspore.graph._parse import compile_config
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)
grad_all = C.GradOperation(get_all=True)


class MulAdd(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        # In this test case, The user defined bprop is wrong defined purposely to distinguish from ad result
        return 2 * dout, 2 * y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_grad_mul_add():
    mul_add = MulAdd()
    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    assert grad_all(mul_add)(x, y) == (2, 4)


class InlineMulADD(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul_add = MulAdd()
        self.param = 2

    def construct(self, x, y):
        return self.mul_add(x, y) + x + self.param * y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_grad_inline_mul_add():
    inline_mul_add = InlineMulADD()
    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    assert grad_all(inline_mul_add)(x, y) == (3, 6)


class WithParameter(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(1, 'param1')
        self.param2 = Parameter(2, 'param2')

    def construct(self, x, y):
        return self.param1 * self.param2 * x + y

    def bprop(self, x, y, out, dout):
        # In this test case, The user defined bprop is wrong defined purposely to distinguish from ad result
        return self.param1 * self.param2 * dout, 2 * y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_with_param():
    with_param = WithParameter()
    with pytest.raises(RuntimeError):
        grad_all(with_param)(mutable(1), 2)
        _pynative_executor.sync()


class WithNoBprop(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_with_no_bprop():
    with_no_bprop = WithNoBprop()
    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    assert grad_all(with_no_bprop)(x, y) == (2, 1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_grad_in_bprop_1():
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0], grads[1]

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_2()

        def construct(self, x, y):
            return self.f(x, y)

    grad_in_bprop = GradInBprop_3()
    grads = grad_all(grad_in_bprop)(Tensor(np.ones([2, 2]).astype(np.float32)),
                                    Tensor(np.ones([2, 2]).astype(np.float32)))
    assert (grads[0].asnumpy() == np.ones([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.zeros([2, 2]).astype(np.float32)).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_grad_in_bprop_with_jit():
    """
    Feature: Test grad jit with custom bprop.
    Description: When custom bprop has J, need to expand.
    Expectation: No exception.
    """
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0] + 10, grads[1] + 10

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_2()

        @jit
        def construct(self, x, y):
            return self.f(x, y)

    grad_in_bprop = GradInBprop_3()
    grads = grad_all(grad_in_bprop)(Tensor(np.ones([2, 2]).astype(np.float32)),
                                    Tensor(np.ones([2, 2]).astype(np.float32)))
    assert (grads[0].asnumpy() == (np.ones([2, 2]) + 10).astype(np.float32)).all()
    assert (grads[1].asnumpy() == (np.zeros([2, 2]) + 10).astype(np.float32)).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_grad_in_bprop_2():
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

        def bprop(self, x, y, out, dout):
            return x * y, y + x

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0], grads[1]

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_2()

        def construct(self, x, y):
            return self.f(x, y)

    grad_in_bprop = GradInBprop_3()
    grads = grad_all(grad_in_bprop)(Tensor(np.ones([2, 2]).astype(np.float32)),
                                    Tensor(np.ones([2, 2]).astype(np.float32)))
    assert (grads[0].asnumpy() == np.ones([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([[2, 2], [2, 2]]).astype(np.float32)).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_grad_in_bprop_3():
    class GradInBprop_1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            return self.relu(x)

    class GradInBprop_2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.f = GradInBprop_1()

        def construct(self, x, y):
            return self.f(x, y), grad_all(self.f)(x, y)

        def bprop(self, x, y, out, dout):
            grads = grad_all(self.f)(x, y)
            return out[1][0], grads[1]

    class GradInBprop_3(nn.Cell):
        def __init__(self):
            super().__init__()
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.75)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_bprop_twoinputbprop():
    """
    Feature: Custom bprop
    Description: Test custom bprop
    Expectation: No exception.
    """
    class GradOfAllInputs(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True)

        def construct(self, *inputs):
            grad_net = self.grad_op(self.net)
            return grad_net(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    net = TwoInputBprop()
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    input2 = Tensor(np.ones([2, 2]).astype(np.float32))
    grad_net = GradOfAllInputs(net)
    grad_net.set_train()
    grads = grad_net(input1, input2)
    assert len(grads) == 2
    assert (grads[0].asnumpy() == np.array([5, 5]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([8, 8]).astype(np.float32)).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
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
        super().__init__()
        self.mul_add = MulAdd()
        self.param = Parameter(Tensor(np.array([[3, 2]], np.float32)), 'param')

    def construct(self, x):
        return self.mul_add(self.param, x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_refkey_bprop():
    grad_by_list = C.GradOperation(get_all=True, get_by_list=True)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super().__init__()
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1.00)
def test_grad_mul_add_with_wrong_output_num():
    compile_config.CHECK_BPROP = 1
    mul_add = MulAddWithWrongOutputNum()
    with pytest.raises(ValueError):
        grad_all(mul_add)(mutable(1), 2)
        _pynative_executor.sync()
    compile_config.CHECK_BPROP = ''


class MulAddWithWrongOutputType(nn.Cell):
    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        return 2 * dout, 2


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_grad_mul_add_with_wrong_output_type():
    compile_config.CHECK_BPROP = 1
    mul_add = MulAddWithWrongOutputType()
    with pytest.raises(TypeError):
        grad_all(mul_add)(1, Tensor(np.ones([2, 2])))
        _pynative_executor.sync()
    compile_config.CHECK_BPROP = ''


class MulAddWithWrongOutputShape(nn.Cell):
    def __init__(self):
        super().__init__()
        self.ones = Tensor(np.ones([2,]))

    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        return 2, self.ones


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1.00)
def test_grad_mul_add_with_wrong_output_shape():
    compile_config.CHECK_BPROP = 1
    mul_add = MulAddWithWrongOutputShape()
    with pytest.raises(ValueError):
        grad_all(mul_add)(1, Tensor(np.ones([2, 2])))
        _pynative_executor.sync()
    compile_config.CHECK_BPROP = ''


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_with_parameter():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_with_parameter_in_sub_cell():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_with_parameter_in_sub_cell_get_by_list():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs and Parameters when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_forward_with_parameter():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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
    context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_forward_with_parameter_in_sub_cell():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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
    context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_forward_with_parameter_in_sub_cell_get_by_list():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of inputs and Parameters when the forward net using Parameter in the sub-cell.
    Expectation: Get the correct gradients.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net = Net1()

        def construct(self, x, y):
            return self.net(x, y)

    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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
    context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dde_self_define_cell_output_not_use():
    """
    Feature: Custom cell bprop
    Description: Fprop output[1] only used by bprop, it should not erased by dde.
    Expectation: Get the correct gradients.
    """

    class SelfDefineCell(ms.nn.Cell):
        def construct(self, x):
            return x + 1, x + 2

        def bprop(self, x, out, dout):
            return (out[1],)

    class ForwardNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.self_defined_cell = SelfDefineCell()

        def construct(self, x):
            # keep out1 not used in fprop.
            out0, _ = self.self_defined_cell(x)
            return out0

    class TestNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.forward_net = ForwardNet()
            self.grad_op = ops.GradOperation(get_all=True)

        def construct(self, x):
            grad_out = self.grad_op(self.forward_net)(x)
            return grad_out

    net = TestNet()
    x_input = ms.Tensor([1])
    out = net(x_input)
    assert out[0] == ms.Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bprop_defined_in_cell_attr_register():
    """
    Feature: Custom cell bprop
    Description: Get the gradients of input for the cell which has been added @cell_attr_register.
    Expectation: Get the correct gradients.
    """

    class Net(nn.Cell):
        @cell_attr_register
        def __init__(self):
            super().__init__()
            self.z = Parameter(Tensor(2, mstype.float32), name='z')

        def construct(self, x, y):
            x = x * self.z
            return x * y

        def bprop(self, x, y, out, dout):
            return y, x

    net = Net()
    x = Tensor(3, mstype.float32)
    y = Tensor(4, mstype.float32)
    output = ops.grad(net)(x, y)
    assert output == 4


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("bprop_return_len", [0, 1, 2, 3])
def test_bprop_return_length_simple(bprop_return_len):
    """
    Feature: Custom cell bprop
    Description: When the output of the custom bprop dose not match the parameter size of forward function,
                 a ValueError should be raised in a simple net.
    Expectation: ValueError is raised with correct message.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x * y

        def bprop(self, x, y, out, dout):
            return (1.0,) * bprop_return_len

    net = Net()
    x = Tensor(3, mstype.float32)
    y = Tensor(4, mstype.float32)

    if bprop_return_len >= 2:
        ops.grad(net)(x, y)
    else:
        with pytest.raises(RuntimeError) as err:
            ops.grad(net)(x, y)

        pattern = f"The output size of the 'bprop' must match the number of parameters in its corresponding primal " \
            f"function '.*' : {bprop_return_len} vs. 2."
        found = re.search(pattern, str(err.value))
        assert found is not None


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("bprop_return_len", [0, 1, 2, 3])
def test_bprop_return_length_with_side_effect_and_param_lift(bprop_return_len):
    """
    Feature: Custom cell bprop
    Description: When the output of the custom bprop dose not match the parameter size of forward function,
                 a ValueError should be raised in a net with side effect and parameter lifting.
    Expectation: ValueError is raised with correct message.
    """
    class Net(nn.Cell):
        def __init__(self) -> None:
            super().__init__()
            self.z = Parameter(Tensor(2, mstype.float32), name='z')
            self.tensor = Tensor(3, mstype.float32)

        def construct(self, x, y):
            print("This is a message that tests for I/O side effect.")
            ops.assign(self.z, Tensor(3, ms.float32))
            return x * y * self.z

        def bprop(self, x, y, out, dout):
            print("This is another message that tests for I/O side effect.")
            return (self.tensor,) * bprop_return_len

    net = Net()
    x = Tensor(3, mstype.float32)
    y = Tensor(4, mstype.float32)

    if bprop_return_len >= 2:
        ops.grad(net)(x, y)
    else:
        with pytest.raises(RuntimeError) as err:
            ops.grad(net)(x, y)

        pattern = f"The output size of the 'bprop' must match the number of parameters in its corresponding primal " \
            f"function '.*' : {bprop_return_len} vs. 2."
        found = re.search(pattern, str(err.value))
        assert found is not None
