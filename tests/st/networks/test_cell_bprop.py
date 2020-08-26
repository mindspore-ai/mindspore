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

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


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
