# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
# pylint: disable=W0235
# pylint: disable=W0612
# pylint: disable=W0621
# pylint: disable=R1705
import torch
import pytest
import numpy as np
import mindspore as ms
import mindspore.ops.functional as F
from mindspore import Tensor, context, nn
from mindspore import dtype as mstype
from mindspore.common import dtype, ParameterTuple
from mindspore.ops import constexpr
from mindspore.ops.operations._sequence_ops import TensorToTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.composite import HyperMap, MultitypeFuncGraph, GradOperation
from mindspore.nn import Cell
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)

add = P.Add()
single_element_fg = C.MultitypeFuncGraph("single_element_fg")


@single_element_fg.register("Tensor")
def single_element_fg_for_tensor(x):
    return P.Square()(x)


double_elements_fg = C.MultitypeFuncGraph("double_elements_fg")


@double_elements_fg.register("Tensor", "Tuple")
def double_elements_fg_for_tensor_tuple(x, y):
    return P.Tile()(x, y)


@double_elements_fg.register("Tensor", "List")
def double_elements_fg_for_tensor_list(x, y):
    return x + y[0]


@double_elements_fg.register("Number", "Number")
def double_elements_fg_for_Number(x, y):
    return x + y


class HyperMapNet(nn.Cell):
    def __init__(self, fg):
        super(HyperMapNet, self).__init__()
        self.common_map = C.HyperMap()
        self.fg = fg

    def construct(self, nest_tensor_list):
        output = self.common_map(self.fg, *nest_tensor_list)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_single_element_hypermap_with_tensor_input():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with single tensor input can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    common_map = HyperMapNet(single_element_fg)
    output = common_map((x,))
    expect_output_1 = np.array([1.0, 4.0, 9.0])
    expect_output_2 = np.array([16.0, 25.0, 36.0])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_double_elements_hypermap_tensor_tuple_inputs():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with tensor and tuple inputs can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ((1, 2), (2, 1))
    common_map = HyperMapNet(double_elements_fg)
    output = common_map((x, y))
    expect_output_1 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    expect_output_2 = np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_double_elements_hypermap_tensor_list_inputs():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with tensor and list inputs can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ([1, 2], [2, 1])
    common_map = HyperMapNet(double_elements_fg)
    output = common_map((x, y))
    expect_output_1 = np.array([2.0, 3.0, 4.0])
    expect_output_2 = np.array([6.0, 7.0, 8.0])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.75)
def test_doubel_elements_hypermap_correct_mix_inputs():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with mix correct inputs (Tensor + Tuple and Tensor + List)
                 can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ((1, 2), [2, 1])
    common_map = HyperMapNet(double_elements_fg)
    output = common_map((x, y))
    expect_output_1 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    expect_output_2 = np.array([6.0, 7.0, 8.0])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_double_elements_hypermap_with_dynamic_element():
    """
    Feature: HyperMap
    Description: When the inputs to hypermap is inconsistent, error will be raised.
    Expectation: error.
    """

    class HyperNet(nn.Cell):
        def __init__(self, fg):
            super(HyperNet, self).__init__()
            self.common_map = C.HyperMap()
            self.fg = fg

        def construct(self, x, y):
            output = self.common_map(self.fg, (TensorToTuple()(
                x), TensorToTuple()(x)), (TensorToTuple()(y), TensorToTuple()(y)))
            return output

    x = Tensor(np.array([1, 2, 3]), mstype.float32)
    y = Tensor(np.array([4, 5, 6]), mstype.float32)
    common_map = HyperNet(double_elements_fg)
    dyn_input = Tensor(shape=(None,), dtype=x.dtype)
    common_map.set_inputs(x, dyn_input)
    res = common_map(x, y)
    assert res == ((5, 7, 9), (5, 7, 9))


class SinNet(Cell):
    def construct(self, x):
        return F.sin(x)


class CosNet(Cell):
    def construct(self, x):
        return F.cos(x)


class TanhNet(Cell):
    def construct(self, x):
        return F.tanh(x)


class SquareNet(Cell):
    def construct(self, x):
        return F.square(x)


def torch_layer(x, y):
    a = torch.cos(x)
    b = torch.tanh(y)
    return a, b


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_switch_layer():
    """
    Feature: HyperMap
    Description: Test the HyperMap with switch_layer.
    Expectation: success.
    """

    test = MultitypeFuncGraph("switch_layer")

    class SNet(Cell):
        def __init__(self, mtfg):
            super().__init__()
            self.hyper_map = HyperMap(mtfg)
            self.layer1 = (SinNet(), CosNet())
            self.layer2 = (TanhNet(), SquareNet())

        def construct(self, i, j, x, y):
            out = self.hyper_map((i, j), (x, y), (self.layer1, self.layer2))
            return out

    @test.register("Tensor", "Tensor", "Tuple")
    def _layer(x, y, layers):  # pylint: disable=W0612
        return layers[x](y)

    net = SNet(test)
    i = Tensor(1, dtype.int32)
    j = Tensor(0, dtype.int32)
    x = np.random.rand(3, 4).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    s1 = np.ones([3, 4]).astype(np.float32)
    s2 = np.ones([3, 4]).astype(np.float32)
    out = net(i, j, Tensor(x), Tensor(y))
    gradnet = F.grad(net, grad_position=(2, 3))
    grad = gradnet(i, j, Tensor(x), Tensor(y))
    tcx = torch.tensor(x, dtype=torch.float, requires_grad=True)
    tcy = torch.tensor(y, dtype=torch.float, requires_grad=True)
    tcs1 = torch.tensor(s1, dtype=torch.float)
    tcs2 = torch.tensor(s2, dtype=torch.float)
    tcout = torch_layer(tcx, tcy)
    tcout[0].backward(tcs1)
    tcout[1].backward(tcs2)
    tcgrad0 = tcx.grad
    tcgrad1 = tcy.grad

    assert np.allclose(tcout[0].detach().numpy(), out[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcout[1].detach().numpy(), out[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad0.detach().numpy(), grad[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad1.detach().numpy(), grad[1].asnumpy(), 0.0001, 0.0001)


class FuncNet(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self):
        out = self.hyper_map((self.excute_square, self.excute_add))
        return out

    def excute_square(self, x):
        return F.square(x)

    def excute_add(self, x):
        return F.add(x, x)


class FuncToc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mtfg):
        out = map(mtfg, (self.excute_square, self.excute_add))
        return out

    def excute_square(self, x):
        return x * x

    def excute_add(self, x):
        return x + x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_tuple_func():
    """
    Feature: HyperMap
    Description: Test HyperMap with function.
    Expectation: success.
    """
    f = MultitypeFuncGraph('test_function')

    @constexpr
    def atensor():
        return Tensor([1, 2], dtype.float32)

    @f.register("Function")
    def _func(func):
        return func(atensor())

    def torch_func(func):
        return func(torch.tensor([1, 2], dtype=torch.float))

    net = FuncNet(f)
    out = net()
    tcnet = FuncToc()
    a, b = list(tcnet(torch_func))
    assert np.allclose(out[0].asnumpy(), a, 0.0001, 0.0001)
    assert np.allclose(out[1].asnumpy(), b, 0.0001, 0.0001)


class SplitNet(Cell):
    def __init__(self, mtfg, s1, s2):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)
        self.s1 = s1
        self.s2 = s2

    def construct(self, x, y):
        out = self.hyper_map((x, y), (self.s1, self.s2))
        return out


class TcSplit(torch.nn.Module):
    def __init__(self, mtfg, s1, s2):
        super().__init__()
        self.mtfg = mtfg
        self.s1 = s1
        self.s2 = s2

    def forward(self, x, y):
        out = map(self.mtfg, (x, y), (self.s1, self.s2))
        return list(out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_op_split():
    """
    Feature: HyperMap
    Description: Test the HyperMap with split.
    Expectation: success.
    """
    s = MultitypeFuncGraph('split')

    @s.register("Tensor", "Number")
    def _split(t, n):
        return P.Split(0, n)(t)

    def torch_split(t, n):
        return torch.split(t, n)

    net = SplitNet(s, 4, 3)
    x = np.random.rand(8, 4).astype(np.float32)
    y = np.random.rand(6, 4).astype(np.float32)
    msa, msb = net(Tensor(x), Tensor(y))

    tcnet = TcSplit(torch_split, 2, 2)
    tcx = torch.tensor(x, dtype=torch.float)
    tcy = torch.tensor(y, dtype=torch.float)
    tca, tcb = tcnet(tcx, tcy)
    for t, m in zip(tca, msa):
        assert np.allclose(t.numpy(), m.asnumpy(), 0.0001, 0.0001)
    for t, m in zip(tcb, msb):
        assert np.allclose(t.numpy(), m.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_forward_return_dyn_container_and_grad():
    """
    Feature: HyperMap
    Description: Test the HyperMap with grad.
    Expectation: success.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, list_in):
            length = len(list_in)
            if length >= 2:
                ele1 = list_in[0]
                ele2 = list_in[length - 1]
                temp = self.add(ele1, ele2)
                return (ele1, ele2, temp)
            else:
                add = self.add(list_in[0], 1)
                return (list_in[0], add)

    input_1 = np.random.rand(2, 2).astype(np.float32)
    input_2 = np.random.rand(2, 2).astype(np.float32)
    input_x = ms.mutable((Tensor(input_1), Tensor(input_2)), dynamic_len=True)
    net = Net()
    out = net(input_x)
    assert np.allclose(out[0].asnumpy(), input_1, 0.0001, 0.0001)
    assert np.allclose(out[1].asnumpy(), input_2, 0.0001, 0.0001)
    assert np.allclose(out[2].asnumpy(), input_1 + input_2, 0.0001, 0.0001)

    grad_func = ms.ops.GradOperation(get_all=True)(Net())
    _ = grad_func(input_x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_multi_dynamic_input():
    """
    Feature: HyperMap
    Description: Test the HyperMap with multi dynamic input.
    Expectation: success.
    """

    def add(x, y):
        return x + y

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.hypermap = ms.ops.HyperMap()

        def construct(self, x, y):
            out = self.hypermap(add, x, y)
            return out

    input_1 = np.random.rand(2, 2, 3).astype(np.float32)
    input_2 = np.random.rand(2, 2, 3).astype(np.float32)
    input_x = ms.mutable((Tensor(input_1), Tensor(input_2)), dynamic_len=True)
    input_y = ms.mutable((Tensor(input_2), Tensor(input_1)), dynamic_len=True)
    net = Net()
    out = net(input_x, input_y)
    assert np.allclose(out[0].asnumpy(), input_1 + input_2, 0.0001, 0.0001)
    assert np.allclose(out[1].asnumpy(), input_1 + input_2, 0.0001, 0.0001)


@constexpr
def get_tuple():
    return tuple()


class OnesNet(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self):
        t = get_tuple()
        out = self.hyper_map((t,))
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_zero_input():
    """
    Feature: HyperMap
    Description: Test the HyperMap with ones.
    Expectation: success.
    """
    ones = MultitypeFuncGraph("ones")

    @ones.register()
    def _ones():
        return P.Ones()(2, 3)

    net = OnesNet(ones)
    out = net()
    assert out[0] == tuple()


class TileNet(Cell):
    def __init__(self, mtfg, m1, m2):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)
        self.m1 = m1
        self.m2 = m2

    def construct(self, x, y):
        out = self.hyper_map((x, y), (self.m1, self.m2))
        return out


class TcTile(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, m1, m2):
        def tc_tile(x, m):
            return x.repeat(m)

        return tuple(map(tc_tile, (x, y), (m1, m2)))


class _Grad(Cell):
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


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_two_input_tile():
    """
    Feature: HyperMap
    Description: Test the HyperMap with tile.
    Expectation: success.
    """
    t = MultitypeFuncGraph('tile')

    @t.register('Tensor', 'Tuple')
    def _tile(x, m):
        return P.Tile()(x, m)

    m1 = (3, 4)
    m2 = (2, 5)

    net = TileNet(t, m1, m2)
    x = np.random.rand(2, 4).astype(np.float32)
    y = np.random.rand(6, 3).astype(np.float32)
    sensenp1 = np.random.rand(6, 16).astype(np.float32)
    sensenp2 = np.random.rand(12, 15).astype(np.float32)
    msout = net(Tensor(x), Tensor(y))
    gradnet = GradOfAllInputs(net)
    msgrad = gradnet(Tensor(x), Tensor(y), (Tensor(sensenp1), Tensor(sensenp2)))
    tcx = torch.tensor(x, dtype=torch.float, requires_grad=True)
    tcy = torch.tensor(y, dtype=torch.float, requires_grad=True)
    tcsense1 = torch.tensor(sensenp1, dtype=torch.float)
    tcsense2 = torch.tensor(sensenp2, dtype=torch.float)
    torch_tile = TcTile()
    tcout = torch_tile(tcx, tcy, m1, m2)
    tcout[0].backward(tcsense1)
    tcout[1].backward(tcsense2)
    tcgrad1, tcgrad2 = tcx.grad, tcy.grad
    assert np.allclose(tcout[0].detach().numpy(), msout[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(tcout[1].detach().numpy(), msout[1].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(tcgrad1.detach().numpy(), msgrad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(tcgrad2.detach().numpy(), msgrad[1].asnumpy(), 0.00001, 0.00001)


class ReshapeNet(Cell):
    def __init__(self, mtfg, shape):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)
        self.shape = shape

    def construct(self, x):
        out = self.hyper_map((x,), (self.shape,))
        return out[0]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_two_input_reshape():
    """
    Feature: HyperMap
    Description: Test the HyperMap with reshape.
    Expectation: success.
    """
    r = MultitypeFuncGraph('reshape')

    @r.register('Tensor', 'Tuple')
    def _reshape(x, s):
        return P.Reshape()(x, s)

    net = ReshapeNet(r, (4, 5))
    x = np.random.rand(2, 10).astype(np.float32)
    sense = np.random.rand(4, 5).astype(np.float32)
    out = net(Tensor(x))
    gradnet = GradOfFirstInput(net)
    grad = gradnet(Tensor(x), Tensor(sense))
    assert np.allclose(x.reshape(4, 5), out.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(sense.reshape(2, 10), grad.asnumpy(), 0.0001, 0.0001)


class SliceNet(Cell):
    def __init__(self, mtfg, begin, size):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)
        self.begin = begin
        self.size = size

    def construct(self, x):
        out = self.hyper_map((x,), (self.begin,), (self.size,))
        return out[0]


def torch_slice(x):
    return x[2: 3, 3: 5, 3: 5]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_three_input_slice():
    """
    Feature: HyperMap
    Description: Test the HyperMap with reshape.
    Expectation: success.
    """
    s = MultitypeFuncGraph('slice')

    @s.register('Tensor', 'List', 'Tuple')
    def _slice(x, begin, size):
        return P.Slice()(x, begin, size)

    begins = [2, 3, 3]
    sizes = (1, 2, 2)
    x = np.random.rand(5, 6, 6).astype(np.float32)
    xs = np.random.rand(1, 2, 2).astype(np.float32)
    net = SliceNet(s, begins, sizes)
    out = net(Tensor(x))
    gradnet = GradOfFirstInput(net)
    grad = gradnet(Tensor(x), Tensor(xs))
    tcx = torch.tensor(x, dtype=torch.float, requires_grad=True)
    tcs = torch.tensor(xs, dtype=torch.float)
    tcout = torch_slice(tcx)
    tcout.backward(tcs)
    tcgrad = tcx.grad
    assert np.allclose(tcout.detach().numpy(), out.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad.detach().numpy(), grad.asnumpy(), 0.0001, 0.0001)


def torch_reshape_mul(x, y, z):
    a = x.reshape(3, 5)
    b = y * z
    return a, b


class ReshapeNet2(Cell):
    def __init__(self, mtfg, shape):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)
        self.shape = shape

    def construct(self, x, y, z):
        out = self.hyper_map((x, y), (self.shape, z))
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_reshape_mul():
    """
    Feature: HyperMap
    Description: Test the HyperMap with reshape.
    Expectation: success.
    """
    test = MultitypeFuncGraph("two_types")

    @test.register("Tensor", "Tuple")
    def _1(x, shape):
        return P.Reshape()(x, shape)

    @test.register("Tensor", "Tensor")
    def _2(x, y):
        return P.Mul()(x, y)

    net = ReshapeNet2(test, (3, 5))
    x = np.random.rand(5, 3).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    z = np.random.rand(3, 4).astype(np.float32)
    s1 = np.random.rand(3, 5).astype(np.float32)
    s2 = np.random.rand(3, 4).astype(np.float32)

    out = net(Tensor(x), Tensor(y), Tensor(z))
    gradnet = GradOfAllInputs(net)
    grad = gradnet(Tensor(x), Tensor(y), Tensor(z), (Tensor(s1), Tensor(s2)))
    tcx = torch.tensor(x, dtype=torch.float, requires_grad=True)
    tcy = torch.tensor(y, dtype=torch.float, requires_grad=True)
    tcz = torch.tensor(z, dtype=torch.float, requires_grad=True)
    tcs1 = torch.tensor(s1, dtype=torch.float)
    tcs2 = torch.tensor(s2, dtype=torch.float)
    tcout = torch_reshape_mul(tcx, tcy, tcz)
    tcout[0].backward(tcs1)
    tcout[1].backward(tcs2)
    tcgrad0 = tcx.grad
    tcgrad1 = tcy.grad
    tcgrad2 = tcz.grad
    assert np.allclose(tcout[0].detach().numpy(), out[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcout[1].detach().numpy(), out[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad0.detach().numpy(), grad[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad1.detach().numpy(), grad[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad2.detach().numpy(), grad[2].asnumpy(), 0.0001, 0.0001)


class AddnNet(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self, x, y):
        out = self.hyper_map(([x, x], y), (y, x))
        return out


def torch_add(x, y):
    a = [x, x]
    b = sum(a) + y
    c = y + x
    return b, c


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_addn_add():
    """
    Feature: HyperMap
    Description: Test the HyperMap with add.
    Expectation: success.
    """
    test = MultitypeFuncGraph("add")

    @test.register("List", "Tensor")
    def _1(a, x):
        s = P.AddN()(a)
        return s + x

    @test.register("Tensor", "Tensor")
    def _2(a, b):
        return a + b

    net = AddnNet(test)
    x = np.random.rand(3, 4).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    s1 = np.random.rand(3, 4).astype(np.float32)
    s2 = np.random.rand(3, 4).astype(np.float32)
    out = net(Tensor(x), Tensor(y))
    gradnet = GradOfAllInputs(net, sens_param=True)
    grad = gradnet(Tensor(x), Tensor(y), (Tensor(s1), Tensor(s2)))
    tcx = torch.tensor(x, dtype=torch.float, requires_grad=True)
    tcy = torch.tensor(y, dtype=torch.float, requires_grad=True)
    tcs1 = torch.tensor(s1, dtype=torch.float)
    tcs2 = torch.tensor(s2, dtype=torch.float)
    tcout = torch_add(tcx, tcy)
    tcout[0].backward(tcs1)
    tcout[1].backward(tcs2)
    tcgrad0, tcgrad1 = tcx.grad, tcy.grad
    assert np.allclose(tcout[0].detach().numpy(), out[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcout[1].detach().numpy(), out[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad0.detach().numpy(), grad[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad1.detach().numpy(), grad[1].asnumpy(), 0.0001, 0.0001)


class ThreeNet(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self, x, y):
        l1 = [x, x]
        t2 = (x, x)
        l3 = [x, y]
        l4 = [y, x]
        l5 = [y, y]
        t6 = (y, y)
        out = self.hyper_map((x, x, y), (l1, t2, l3), (l4, l5, t6))
        return out


def torch_sum(x, y):
    a = x * (x + x) * (y + x)
    b = x * (x + x) + (y + y)
    c = y - (x + y) * (y + y)
    return a, b, c


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_register_three_types():
    """
    Feature: HyperMap
    Description: Test the HyperMap with addn.
    Expectation: success.
    """
    test = MultitypeFuncGraph("Three")

    @test.register("Tensor", "List", "List")
    def _1(a, b, c):
        return a * P.AddN()(b) * P.AddN()(c)

    @test.register("Tensor", "Tuple", "List")
    def _2(a, b, c):
        return a * P.AddN()(b) + P.AddN()(c)

    @test.register("Tensor", "List", "Tuple")
    def _3(a, b, c):
        return a - P.AddN()(b) * P.AddN()(c)

    net = ThreeNet(test)
    x = np.random.rand(3, 4).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    s1 = np.random.rand(3, 4).astype(np.float32)
    s2 = np.random.rand(3, 4).astype(np.float32)
    s3 = np.random.rand(3, 4).astype(np.float32)
    out = net(Tensor(x), Tensor(y))
    gradnet = GradOfAllInputs(net)
    grad = gradnet(Tensor(x), Tensor(y), (Tensor(s1), Tensor(s2), Tensor(s3)))
    tcx = torch.tensor(x, dtype=torch.float, requires_grad=True)
    tcy = torch.tensor(y, dtype=torch.float, requires_grad=True)
    tcs1 = torch.tensor(s1, dtype=torch.float)
    tcs2 = torch.tensor(s2, dtype=torch.float)
    tcs3 = torch.tensor(s3, dtype=torch.float)
    tcout = torch_sum(tcx, tcy)
    tcout[0].backward(tcs1)
    tcout[1].backward(tcs2)
    tcout[2].backward(tcs3)
    tcgrad0 = tcx.grad
    tcgrad1 = tcy.grad
    assert np.allclose(tcout[0].detach().numpy(), out[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcout[1].detach().numpy(), out[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcout[2].detach().numpy(), out[2].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad0.detach().numpy(), grad[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(tcgrad1.detach().numpy(), grad[1].asnumpy(), 0.0001, 0.0001)


class SumNet(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self, x):
        out = self.hyper_map((x, x, x), (x, x, x))
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_one_input_tuple_sum():
    """
    Feature: HyperMap
    Description: Test the HyperMap with sum.
    Expectation: success.
    """
    s = MultitypeFuncGraph("sum")

    @s.register("Tuple")
    def _sum(t):
        return P.AddN()(t)

    net = SumNet(s)
    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    with pytest.raises(TypeError):
        net(x)


class ConcatNet(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self, x):
        out = self.hyper_map([[x, x], [x, x]])
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_one_input_list_concat():
    """
    Feature: HyperMap
    Description: Test the HyperMap with concat.
    Expectation: success.
    """
    c = MultitypeFuncGraph('concat')

    @c.register('List')
    def _concat(t):
        return P.Concat()(t)

    x = Tensor([1, 2, 3])
    net = ConcatNet(c)
    with pytest.raises(TypeError):
        net(x)


class SumNet2(Cell):
    def __init__(self, mtfg):
        super().__init__()
        self.hyper_map = HyperMap(mtfg)

    def construct(self, x, y):
        t1 = (x, x, y)
        t2 = (y, y, x)
        t3 = (x, y)
        out = self.hyper_map((y, x), (t1, t2), t3)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hypermap_register_diff_input_num():
    """
    Feature: HyperMap
    Description: Test the HyperMap with sum.
    Expectation: success.
    """
    test = MultitypeFuncGraph("sum")

    @test.register("Tensor", "Tuple", "Tuple")
    def _1(x, y, z):
        return x + sum(y) + sum(z)

    @test.register("Tensor", "Tuple")
    def _2(x, y):
        return x + sum(y)

    net = SumNet2(test)
    x = Tensor(np.random.rand(3, 4).astype(np.float32))
    y = Tensor(np.random.rand(3, 4).astype(np.float32))

    with pytest.raises(ValueError):
        net(x, y)
