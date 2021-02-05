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
"""
@File  : test_parse.py
@Author:
@Date  : 2019-01-23 17:13
@Desc  :
"""
import logging
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.api import ms_function, _executor
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer
from mindspore.ops.functional import tensor_add
from ...ut_filter import non_graph_engine

# pylint: disable=W0613,W0612
# W0613: unused-argument

@pytest.fixture(name='enable_check_bprop')
def fixture_enable_check_bprop():
    context.set_context(check_bprop=True)
    yield
    context.set_context(check_bprop=False)


grad_all = C.GradOperation(get_all=True)


log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)
context.set_context(mode=context.GRAPH_MODE)

# Test case: use the parse obj interface use default parameter
class Net(nn.Cell):
    """ Net definition """

    def __init__(self, dim):
        super(Net, self).__init__()
        self.softmax1 = nn.Softmax(dim)
        self.softmax2 = nn.Softmax(dim + 1)

    def construct(self, input_data, input1=1+2+3+4):
        return self.softmax1(input_data)


@non_graph_engine
def test_parse_defalut_parameter_case2():
    """ test_parse_defalut_parameter_case2 """
    log.debug("begin test_parse_defalut_parameter_case2")
    net = Net(0)
    npd = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')
    log.debug("input value is: %r", npd)
    input_data = ms.Tensor(npd)
    input_data.set_dtype(ms.float32)

    log.debug("start run")
    output = net(input_data)

    value = output.asnumpy()
    log.debug("output value = %r", value)


# Test case: use the variable parameter for parse object
class Net1(nn.Cell):
    """ Net1 definition """

    def __init__(self):
        super(Net1, self).__init__()

    def construct(self, *args):
        x = args[0]
        return x


def test_var_parameter_case2():
    """ test_var_parameter_case2 """
    log.debug("begin test_var_parameter_case2")
    net = Net1()
    npd = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')
    log.debug("input value is: %r", npd)
    input_data = ms.Tensor(npd)
    input_data.set_dtype(ms.float32)

    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input1 = ms.Tensor(np1)
    np2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input2 = ms.Tensor(np2)

    _executor.compile(net, input_data, input1, input2)


# Test case: test the global flag
g_x = Tensor(np.ones([3, 3]).astype(np.float32))


@ms_function
def tensor_add_global(x):
    """ tensor_add_global """
    global g_x
    res = tensor_add(x, g_x)
    return res


@non_graph_engine
def test_global_flag():
    """ test_global_flag """
    log.debug("begin test_global_flag")
    x = Tensor(np.ones([3, 3]).astype(np.float32))
    res = tensor_add_global(x)
    log.debug("finished test_global_flag, ret = %r", res)


class NetWithNDarray(nn.Cell):
    """ NetWithNDarray definition """

    def __init__(self, dim):
        super(NetWithNDarray, self).__init__()
        self.softmax = nn.Softmax(dim)
        self.x = ms.Tensor(np.ones(shape=(1)).astype(np.float32))

    def construct(self, input_data):
        return self.softmax(input_data) * self.x


@non_graph_engine
def test_net_with_ndarray():
    """ test_net_with_ndarray """
    net = NetWithNDarray(0)
    input_data = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')

    net(ms.Tensor(input_data))


def test_bprop_with_wrong_output_num(enable_check_bprop):
    class BpropWithWrongOutputNum(PrimitiveWithInfer):
        @prim_attr_register
        def __init__(self):
            super(BpropWithWrongOutputNum, self).__init__('BpropWithWrongOutputNum')

        def __call__(self, x, y):
            return x

        def infer_shape(self, x_shape, yshape):
            return x_shape

        def infer_dtype(self, x_type, y_type):
            return x_type

    @bprop_getters.register(BpropWithWrongOutputNum)
    def get_bprop_with_wrong_output_num(self):
        """Generate bprop for BpropWithWrongOutputNum"""

        def bprop(x, y, out, dout):
            return (dout,)

        return bprop

    class BpropWithWrongOutputNumCell(nn.Cell):
        def __init__(self):
            super(BpropWithWrongOutputNumCell, self).__init__()

        def construct(self, x, y):
            return BpropWithWrongOutputNum()(x, y)

    with pytest.raises(ValueError):
        grad_all(BpropWithWrongOutputNumCell())(Tensor(np.array(1).astype(np.int32)),
                                                Tensor(np.array(2).astype(np.int32)))

def test_bprop_with_wrong_output_type(enable_check_bprop):
    class BpropWithWrongOutputType(PrimitiveWithInfer):
        @prim_attr_register
        def __init__(self):
            super(BpropWithWrongOutputType, self).__init__('BpropWithWrongOutputType')

        def __call__(self, x):
            return x

        def infer_shape(self, x_shape):
            return x_shape

        def infer_dtype(self, x_type):
            return x_type

    @bprop_getters.register(BpropWithWrongOutputType)
    def get_bprop_with_wrong_output_type(self):
        """Generate bprop for BpropWithWrongOutputType"""

        def bprop(x, out, dout):
            return (1,)

        return bprop

    class BpropWithWrongOutputTypeCell(nn.Cell):
        def __init__(self):
            super(BpropWithWrongOutputTypeCell, self).__init__()

        def construct(self, x):
            return BpropWithWrongOutputType()(x)

    with pytest.raises(TypeError):
        grad_all(BpropWithWrongOutputTypeCell())(Tensor(np.ones([64, 10]).astype(np.int32)))


def test_bprop_with_wrong_output_shape(enable_check_bprop):
    class BpropWithWrongOutputShape(PrimitiveWithInfer):
        @prim_attr_register
        def __init__(self):
            super(BpropWithWrongOutputShape, self).__init__('BpropWithWrongOutputShape')

        def __call__(self, x):
            return x

        def infer_shape(self, x_shape):
            return x_shape

        def infer_dtype(self, x_type):
            return x_type

    @bprop_getters.register(BpropWithWrongOutputShape)
    def get_bprop_with_wrong_output_shape(self):
        """Generate bprop for BpropWithWrongOutputShape"""
        ones = Tensor(np.ones([2,]).astype(np.int32))

        def bprop(x, out, dout):
            return (ones,)

        return bprop

    class BpropWithWrongOutputShapeCell(nn.Cell):
        def __init__(self):
            super(BpropWithWrongOutputShapeCell, self).__init__()

        def construct(self, x):
            return BpropWithWrongOutputShape()(x)

    with pytest.raises(ValueError):
        net = BpropWithWrongOutputShapeCell()
        net.set_grad()
        grad_all(net)(Tensor(np.ones([64, 10]).astype(np.int32)))

class AssignWhenInsertGrad(nn.Cell):
    """ NetWithNDarray definition """

    def __init__(self):
        super(AssignWhenInsertGrad, self).__init__()
        self.gather = P.Gather()
        self.damping = Tensor(np.array([0.03, 0.03]).astype(np.float32))
        self.cov_step = ms.Parameter(0, name="cov_step", requires_grad=False)
        self.freq = Tensor(278, ms.int32)
        self.getG = P.InsertGradientOf(self.save_gradient)

    def save_gradient(self, dout):
        self.cov_step = self.cov_step + self.freq
        return dout

    def construct(self, x):
        self.gather(self.damping, self.cov_step, 0)
        out = P.ReLU()(x)
        out = self.getG(out)
        return out

grad_all = C.GradOperation(get_all=True)

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net

    def construct(self, *inputs):
        out = self.net(*inputs)
        return out, grad_all(self.net)(*inputs)

def test_assign_in_insert_grad():
    context.set_context(mode=context.GRAPH_MODE)
    net = AssignWhenInsertGrad().to_float(ms.float16)
    input_data = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')
    net_back = GradNet(net)
    net_back(ms.Tensor(input_data))

class Assign(nn.Cell):
    """ NetWithNDarray definition """

    def __init__(self):
        super(Assign, self).__init__()
        self.cov_step = ms.Parameter(0.0, name="cov_step", requires_grad=False)

    def construct(self, x):
        self.cov_step = self.cov_step + x
        return self.cov_step


def test_assign(enable_check_bprop):
    context.set_context(mode=context.GRAPH_MODE)
    net = Assign()
    input_data = ms.Tensor(np.array(1).astype(np.int32))
    net_back = GradNet(net)
    net_back(input_data)

class AssignCheck(nn.Cell):
    """ NetWithNDarray definition """

    def __init__(self):
        super(AssignCheck, self).__init__()
        self.cov_step = ms.Parameter(0.0, name="cov_step", requires_grad=False)

    def construct(self, x):
        self.cov_step = x
        return self.cov_step


def test_assign_check_none():
    context.set_context(mode=context.GRAPH_MODE)
    net = AssignCheck()
    with pytest.raises(TypeError):
        net(None)
