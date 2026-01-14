# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""test auto monad"""
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=R1705
# pylint: disable=R1707
import os
import time
import tempfile
import scipy
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn, context, Tensor, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops.composite import GradOperation
from mindspore.train.summary.summary_record import SummaryRecord
from tests.summary_utils import SummaryReader
from tests.security_utils import security_off_wrap
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class AssignAddNet(nn.Cell):
    def __init__(self, para):
        super().__init__()
        self.para = Parameter(para, name="para")
        self.assign_add = P.AssignAdd()

    def construct(self, value):
        self.assign_add(self.para, value)
        return self.para


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_assign_add():
    """
    Feature: Auto monad feature.
    Description: Verify assign_add.
    Expectation: No exception.
    """
    x = Tensor(1, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    expect = Tensor(3, dtype=mstype.int32)
    net = AssignAddNet(x)
    out = net(y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


class AssignSubNet(nn.Cell):
    def __init__(self, para):
        super().__init__()
        self.para = Parameter(para, name="para")
        self.assign_sub = P.AssignSub()

    def construct(self, value):
        self.assign_sub(self.para, value)
        return self.para


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_assign_sub():
    """
    Feature: Auto monad feature.
    Description: Verify assign_sub.
    Expectation: No exception.
    """
    x = Tensor(3, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    expect = Tensor(1, dtype=mstype.int32)
    net = AssignSubNet(x)
    out = net(y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


class ScatterAddNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_add = P.ScatterAdd()

    def construct(self, indices, updates):
        self.scatter_add(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_add():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_add.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]), mstype.float32)
    expect = Tensor(np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]), mstype.float32)
    net = ScatterAddNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterSubNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_sub = P.ScatterSub()

    def construct(self, indices, updates):
        self.scatter_sub(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_sub():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_sub.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]), mstype.float32)
    expect = Tensor(np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]), mstype.float32)
    net = ScatterSubNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterMulNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_mul = P.ScatterMul()

    def construct(self, indices, updates):
        self.scatter_mul(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_scatter_mul():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_mul.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]), mstype.float32)
    expect = Tensor(np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]), mstype.float32)
    net = ScatterMulNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterDivNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_div = P.ScatterDiv()

    def construct(self, indices, updates):
        self.scatter_div(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_div():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_div.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]), mstype.float32)
    expect = Tensor(np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]), mstype.float32)
    net = ScatterDivNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterMaxNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_max = P.ScatterMax()

    def construct(self, indices, updates):
        self.scatter_max(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_max():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_max.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]) * 88, mstype.float32)
    expect = Tensor(np.array([[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]]), mstype.float32)
    net = ScatterMaxNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterMinNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_min = P.ScatterMin()

    def construct(self, indices, updates):
        self.scatter_min(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_min():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_min.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]), mstype.float32)
    expect = Tensor(np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), mstype.float32)
    net = ScatterMinNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterUpdateNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_update = P.ScatterUpdate()

    def construct(self, indices, updates):
        self.scatter_update(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_scatter_update():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_update.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([[[1.0, 2.2, 1.0], [2.0, 1.2, 1.0]], [[2.0, 2.2, 1.0], [3.0, 1.2, 1.0]]]), mstype.float32)
    expect = Tensor(np.array([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]]), mstype.float32)
    net = ScatterUpdateNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNdAddNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_nd_add = P.ScatterNdAdd()

    def construct(self, indices, updates):
        self.scatter_nd_add(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_nd_add():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_nd_add.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    expect = Tensor(np.array([1, 10, 9, 4, 12, 6, 7, 17]), mstype.float32)
    net = ScatterNdAddNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNdSubNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_nd_sub = P.ScatterNdSub()

    def construct(self, indices, updates):
        self.scatter_nd_sub(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_scatter_nd_sub():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_nd_sub.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    expect = Tensor(np.array([1, -6, -3, 4, -2, 6, 7, -1]), mstype.float32)
    net = ScatterNdSubNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNdUpdateNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_nd_update = P.ScatterNdUpdate()

    def construct(self, indices, updates):
        self.scatter_nd_update(self.input_x, indices, updates)
        return self.input_x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scatter_nd_update():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_nd_update.
    Expectation: No exception.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)
    expect = Tensor(np.array([[1., 0.3, 3.6], [0.4, 2.2, -3.2]]), mstype.float32)
    net = ScatterNdUpdateNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNonAliasingAddNet(nn.Cell):
    def __init__(self, input_x):
        super().__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_non_aliasing_add = P.ScatterNonAliasingAdd()

    def construct(self, indices, updates):
        out = self.scatter_non_aliasing_add(self.input_x, indices, updates)
        return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_scatter_non_aliasing_add():
    """
    Feature: Auto monad feature.
    Description: Verify scatter_non_aliasing_add.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    expect = Tensor(np.array([1.0, 10.0, 9.0, 4.0, 12.0, 6.0, 7.0, 17.0]), mstype.float32)
    net = ScatterNonAliasingAddNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class SummaryNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.scalar_summary = P.ScalarSummary()
        self.image_summary = P.ImageSummary()
        self.tensor_summary = P.TensorSummary()
        self.histogram_summary = P.HistogramSummary()

    def construct(self, image_tensor):
        self.image_summary("image", image_tensor)
        self.tensor_summary("tensor", image_tensor)
        self.histogram_summary("histogram", image_tensor)
        scalar = image_tensor[0][0][0][0]
        self.scalar_summary("scalar", scalar)
        return scalar


def train_summary_record(test_writer, steps):
    """Train and record summary."""
    net = SummaryNet()
    out_me_dict = {}
    for i in range(0, steps):
        image_tensor = Tensor(np.array([[[[i]]]]).astype(np.float32))
        out_put = net(image_tensor)
        time.sleep(0.5)
        test_writer.record(i)
        out_me_dict[i] = out_put.asnumpy()
    return out_me_dict


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.02)
@security_off_wrap
def test_summary():
    """
    Feature: Auto monad feature.
    Description: Verify summary operator.
    Expectation: No exception.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        steps = 2
        with SummaryRecord(tmp_dir) as test_writer:
            train_summary_record(test_writer, steps=steps)

            file_name = os.path.realpath(test_writer.log_dir)
        with SummaryReader(file_name) as summary_writer:
            for _ in range(steps):
                event = summary_writer.read_event()
                tags = set(value.tag for value in event.summary.value)
                assert tags == {'tensor', 'histogram', 'scalar', 'image'}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_igamma():
    """
    Feature: Auto monad feature.
    Description: Verify igamma operator.
    Expectation: No exception.
    """

    class IGammaTest(nn.Cell):
        def __init__(self):
            super().__init__()
            self.igamma = nn.IGamma()

        def construct(self, x, a):
            return self.igamma(a=a, x=x)

    x = 4.22
    a = 2.29
    net = IGammaTest()
    out = net(Tensor(x, mstype.float32), Tensor(a, mstype.float32))
    expect = scipy.special.gammainc(a, x)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-5, atol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_side_effect_scalarsummary_in_bprop():
    """
    Feature: Auto monad feature.
    Description: Verify ScalarSummary operator in bprop.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.summary = P.ScalarSummary()

        def construct(self, x, y):
            return x, y

        def bprop(self, x, y, out, dout):
            name_x = "xx"
            name_y = "yy"
            self.summary(name_x, x)
            self.summary(name_y, y)
            dx = x * 2
            dy = y * 3
            return dx, dy

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ms.ops.GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor([3], dtype=ms.int32)
    y = Tensor([4], dtype=ms.int32)
    net = Net()
    _ = net(x, y)
    out_grad = GradNet(Net())(x, y)
    assert out_grad[0].asnumpy() == 6
    assert out_grad[1].asnumpy() == 12


class _Grad(nn.Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        if self.wrt_params:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network, self.params)(*inputs)
            else:
                real_inputs = inputs[:self.real_inputs_count]
                sense_param_inputs = inputs[self.real_inputs_count:]
                return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        else:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network)(*inputs)
            else:
                real_inputs = inputs[:self.real_inputs_count]
                sense_param_inputs = inputs[self.real_inputs_count:]
                return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class SideEffectOneInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()
        self.mul = P.Mul()
        self.print1 = P.Print()

    def construct(self, x):
        return self.relu(x)

    def bprop(self, x, out, dout):
        x = self.relu(x)
        x = 5 * x
        self.print1("x1: ", x)
        x = self.mul(x, x)
        self.print1("x2: ", x)
        return 5 * x,

    def grad_mindspore_impl(self, params1, grad_ys):
        grad_net = GradOfAllInputs(self, sens_param=True)
        grad_net.set_train()
        grad_out = grad_net(params1, grad_ys)
        return grad_out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_side_effect_bprop_oneinputbprop():
    """
    Feature: Auto monad feature.
    Description: Verify print operator in bprop.
    Expectation: No exception.
    """
    net = SideEffectOneInputBprop()
    net.set_grad()
    grad_ys = Tensor(np.ones([2, 2]), ms.float32)
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    grads = net.grad_mindspore_impl(input1, grad_ys)
    assert (grads[0].asnumpy() == np.ones([2, 2]).astype(np.float32) * 125).all()
    assert len(grads) == 1
