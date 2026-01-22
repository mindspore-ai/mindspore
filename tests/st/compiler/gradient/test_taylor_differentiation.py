# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""test taylor differentiation in graph mode"""
import pytest
import numpy as np
from mindspore import nn
from mindspore import context
from mindspore import ops
from mindspore import Tensor, jit, Parameter
from mindspore.ops.functional import jet, derivative, grad
from mindspore.common import dtype
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark

context.set_context(jit_level='O0')


class MultipleInputSingleOutputNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        self.exp = ops.Exp()

    def construct(self, x, y):
        out1 = self.sin(x)
        out2 = self.cos(y)
        out3 = out1 * out2 + out1 / out2
        out = self.exp(out3)
        return out


class MultipleInputMultipleOutputNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x, y):
        out1 = self.sin(x)
        out2 = self.cos(y)
        return out1, out2


class SingleInputSingleOutputNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        self.exp = ops.Exp()

    def construct(self, x):
        out1 = self.sin(x)
        out2 = self.cos(out1)
        out3 = self.exp(out2)
        out = out1 + out2 - out3
        return out


def function_graph(x):
    y = ops.exp(x)
    z = ops.tan(y)
    return z


class SingleInputSingleOutputWithScalarNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.log = ops.Log()

    def construct(self, x):
        out1 = self.log(x)
        out = ops.add(ops.div(1, out1), 2)
        return ops.mul(out, 3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_single_input_single_output_graph_mode(mode):
    """
    Features: Function jet
    Description: Test jet with single input in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    primals = Tensor([1., 1.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    net = SingleInputSingleOutputNet()
    expected_primals = np.array([-0.43931, -0.43931]).astype(np.float32)
    expected_series = np.array([[0.92187, 0.92187], [-1.56750, -1.56750], [-0.74808, -0.74808]]).astype(np.float32)
    out_primals, out_series = jet(net, primals, series)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_single_input_single_output_with_scalar_graph_mode(mode):
    """
    Features: Function jet
    Description: Test jet with single input with scalar in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    primals = Tensor([2., 2.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    net = SingleInputSingleOutputWithScalarNet()
    out_primals, out_series = jet(net, primals, series)
    expected_primals = np.array([10.328085, 10.328085]).astype(np.float32)
    expected_series = np.array([[-3.1220534, -3.1220534], [6.0652323, 6.0652323],
                                [-18.06463, -18.06463]]).astype(np.float32)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_single_input_single_output_graph_mode(mode):
    """
    Features: Function derivative
    Description: Test derivative with single input in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    primals = Tensor([1., 1.])
    order = 3
    net = SingleInputSingleOutputNet()
    expected_primals = np.array([-0.43931, -0.43931]).astype(np.float32)
    expected_series = np.array([-0.74808, -0.74808]).astype(np.float32)
    out_primals, out_series = derivative(net, primals, order)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_multiple_input_single_output_graph_mode(mode):
    """
    Features: Function jet
    Description: Test jet with multiple inputs in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    primals = (Tensor([1., 1.]), Tensor([1., 1.]))
    series = (Tensor([[1., 1.], [0., 0.], [0., 0.]]), Tensor([[1., 1.], [0., 0.], [0., 0.]]))
    net = MultipleInputSingleOutputNet()
    expected_primals = np.array([7.47868, 7.47868]).astype(np.float32)
    expected_series = np.array([[22.50614, 22.50614], [133.92517, 133.92517], [1237.959, 1237.959]]).astype(np.float32)
    out_primals, out_series = jet(net, primals, series)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


class AddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    def construct(self, x, y):
        return self.add(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('input_dtype', [dtype.float64, dtype.int64, dtype.int32, dtype.int16])
def test_jet_multiple_input_single_output_graph_mode_dtype(mode, input_dtype):
    """
    Features: Function jet
    Description: Test jet with different input types in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = jet(self.net, (x, x), (y, y))
            return out, prime


    context.set_context(mode=mode)
    net = AddNet()
    x = Tensor([1, 1], dtype = input_dtype)
    y = Tensor([[1, 1], [0, 0]], dtype = input_dtype)
    ms_net = Net(net)
    if input_dtype == dtype.int16:
        with pytest.raises(TypeError):
            ms_net(x, y)
            _pynative_executor.sync()
    else:
        ms_out = ms_net(x, y)
        assert np.allclose(ms_out[0].asnumpy(), [2, 2], atol=1.e-4)
        assert np.allclose(ms_out[1].asnumpy(), [[2, 2], [0, 0]], atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_multiple_input_single_output_graph_mode(mode):
    """
    Features: Function derivative
    Description: Test derivative with multiple inputs in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    primals = (Tensor([1., 1.]), Tensor([1., 1.]))
    order = 3
    net = MultipleInputSingleOutputNet()
    expected_primals = np.array([7.47868, 7.47868]).astype(np.float32)
    expected_series = np.array([1237.959, 1237.959]).astype(np.float32)
    out_primals, out_series = derivative(net, primals, order)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('input_dtype', [dtype.float64, dtype.int64, dtype.int32])
def test_derivative_multiple_input_single_output_graph_mode_dtype(mode, input_dtype):
    """
    Features: Function derivative
    Description: Test derivative with different input types in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = derivative(self.net, (x, x), y)
            return out, prime


    context.set_context(mode=mode)
    net = AddNet()
    x = Tensor([1, 1], dtype = input_dtype)
    y = 2
    ms_net = Net(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [2, 2], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), [0, 0], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_invalid_order_0(mode):
    """
    Features: Function derivative
    Description: Test derivative with invalid order.
    Expectation: ValueError.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = derivative(self.net, (x, x), y)
            return out, prime


    context.set_context(mode=mode)
    net = AddNet()
    x = Tensor([1, 1], dtype.int32)
    y = 0
    ms_net = Net(net)
    with pytest.raises(ValueError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_invalid_order_float(mode):
    """
    Features: Function derivative
    Description: Test derivative with invalid order.
    Expectation: TypeError.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = derivative(self.net, (x, x), y)
            return out, prime


    context.set_context(mode=mode)
    net = AddNet()
    x = Tensor([1, 1], dtype.int32)
    y = 1.5
    ms_net = Net(net)
    with pytest.raises(TypeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_invalid_order_int16(mode):
    """
    Features: Function derivative
    Description: Test derivative with invalid order.
    Expectation: TypeError.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = derivative(self.net, (x, x), y)
            return out, prime


    context.set_context(mode=mode)
    net = AddNet()
    x = Tensor([1, 1], dtype.int16)
    y = 2
    ms_net = Net(net)
    with pytest.raises(TypeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_invalid_input_type(mode):
    """
    Features: Function derivative
    Description: Test derivative with invalid input type.
    Expectation: TypeError.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = derivative(self.net, (x, x), y)
            return out, prime


    context.set_context(mode=mode)
    net = AddNet()
    x = Tensor([1, 1], dtype.int32)
    y = Tensor(2, dtype.float32)
    ms_net = Net(net)
    with pytest.raises(TypeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_invalid_input_func(mode):
    """
    Features: Function derivative
    Description: Test derivative with invalid input func.
    Expectation: RuntimeError.
    """
    context.set_context(mode=mode)
    x = Tensor([1, 1], dtype.float32)
    y = 2
    with pytest.raises(RuntimeError):
        derivative(ops.Add(), (x, x), y)
        _pynative_executor.sync()


class SinNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sin = ops.Sin()

    def construct(self, x):
        out = self.sin(x)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_derivative_grad(mode):
    """
    Features: Function derivative
    Description: Test derivative with multiple inputs in graph mode.
    Expectation: No exception.
    """
    class Grad(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, a, b):
            def get_der(x, y):
                return derivative(self.net, x, y)

            grad_net = ops.grad(get_der)
            grad_result = grad_net(a, b)
            return grad_result

    context.set_context(mode=mode)
    net = SinNet()
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = Grad(net)
    dgrad = ms_net(x, y)
    assert np.allclose(dgrad.asnumpy(), np.array([0, 0]), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_derivative_derivative(mode):
    """
    Features: Function derivative
    Description: Test derivative multiple times.
    Expectation: RuntimeError.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            _, self.derivative_net = derivative(net, x, y)

        def construct(self, x, y):
            out, prime = derivative(self.derivative_net, x, y)
            return out, prime

    context.set_context(mode=mode)
    net = SinNet()
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = Net(net)
    with pytest.raises(RuntimeError):
        ms_net(x, y)
        _pynative_executor.sync()


class DerivativeNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x, y):
        out, prime = derivative(self.net, x, y)
        return out, prime

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_fn_func(mode):
    """
    Features: Function derivative
    Description: Test derivative with a function as input.
    Expectation: No exception.
    """
    def exp_sin(x):
        return ops.Sin()(ops.Exp()(x))

    context.set_context(mode=mode)
    net = exp_sin
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [0.41078135, 0.41078135], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), [-5.513626, -5.513626], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_derivative_fn_jit_func():
    """
    Features: Function derivative
    Description: Test derivative with a jit function as input.
    Expectation: No exception.
    """
    @jit
    def cos_exp(x):
        return ops.Cos()(ops.Exp()(x))

    net = cos_exp
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [-0.91173387, -0.91173387], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), [5.620221, 5.620221], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_primitive(mode):
    """
    Features: Function derivative
    Description: Test derivative with a primitive function as input.
    Expectation: RuntimeError.
    """
    context.set_context(mode=mode)
    net = ops.Sin()
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    with pytest.raises(RuntimeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_lambda(mode):
    """
    Features: Function derivative
    Description: Test derivative with a lambda function as input.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    net = lambda x: x * x * x # pylint: disable=unnecessary-lambda-assignment
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [0.99999905, 0.99999905], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), [5.9999943, 5.9999943], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_parameter(mode):
    """
    Features: Function derivative
    Description: Test derivative with a net with parameters as input.
    Expectation: RuntimeError.
    """
    class ParameterNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor([1, 2], dtype.float32), name="p")

        def construct(self, x):
            return self.param * x

    context.set_context(mode=mode)
    net = ParameterNet()
    x = Tensor([1, 1], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    with pytest.raises(RuntimeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_if(mode):
    """
    Features: Function derivative
    Description: Test derivative with a net with if as input.
    Expectation: ValueError.
    """
    class IfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sin = ops.Sin()
            self.cos = ops.Cos()

        def construct(self, x):
            if x > 1:
                out = self.sin(x)
            else:
                out = self.cos(x)
            return out

    context.set_context(mode=mode)
    net = IfNet()
    x = Tensor([2], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    with pytest.raises(ValueError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_dyn(mode):
    """
    Features: Function derivative
    Description: Test derivative with dynamic input.
    Expectation: ValueError.
    """
    context.set_context(mode=mode)
    net = SinNet()
    dyn = Tensor(shape=[None], dtype=dtype.float32)
    net.set_inputs(dyn)
    x = Tensor([2], dtype.float32)
    y = 2
    ms_net = DerivativeNet(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [0.9092974], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), [-0.9092965], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_construct_graph_mode(mode):
    """
    Features: Function jet
    Description: Test jet in construct with multiple inputs in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            res_primals, res_series = jet(self.net, x, y)
            return res_primals, res_series

    primals = Tensor([2., 2.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    net = SingleInputSingleOutputWithScalarNet()
    hod_net = Net(net)
    expected_primals = np.array([10.328085, 10.328085]).astype(np.float32)
    expected_series = np.array([[-3.1220534, -3.1220534], [6.0652323, 6.0652323],
                                [-18.06463, -18.06463]]).astype(np.float32)
    out_primals, out_series = hod_net(primals, series)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_derivative_construct_graph_mode(mode):
    """
    Features: Function derivative
    Description: Test derivative in construct with multiple inputs in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    class Net(nn.Cell):
        def __init__(self, net, order):
            super().__init__()
            self.net = net
            self.order = order

        def construct(self, x, y):
            res_primals, res_series = derivative(self.net, (x, y), self.order)
            return res_primals, res_series

    primals_x = Tensor([1., 1.])
    primals_y = Tensor([1., 1.])
    net = MultipleInputMultipleOutputNet()
    hod_net = Net(net, order=3)
    expected_primals_x = np.array([0.841470957, 0.841470957]).astype(np.float32)
    expected_primals_y = np.array([0.540302277, 0.540302277]).astype(np.float32)
    expected_series_x = np.array([-0.540302277, -0.540302277]).astype(np.float32)
    expected_series_y = np.array([0.841470957, 0.841470957]).astype(np.float32)
    out_primals, out_series = hod_net(primals_x, primals_y)
    assert np.allclose(out_primals[0].asnumpy(), expected_primals_x, atol=1.e-4)
    assert np.allclose(out_primals[1].asnumpy(), expected_primals_y, atol=1.e-4)
    assert np.allclose(out_series[0].asnumpy(), expected_series_x, atol=1.e-4)
    assert np.allclose(out_series[1].asnumpy(), expected_series_y, atol=1.e-4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_function_graph_mode(mode):
    """
    Features: Function jet
    Description: Test function in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    primals = Tensor([1., 1.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    out_primals, out_series = jet(function_graph, primals, series)
    expected_primals = np.array([-0.450549, -0.450549]).astype(np.float32)
    expected_series = np.array([[3.270079, 3.270079], [-4.739784, -4.739784],
                                [56.995613, 56.995613]]).astype(np.float32)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_add_2_inputs_float32_not_cell_or_function(mode):
    """
    Features: Function jet
    Description: Test jet, with fn not cell or function, dtype float32
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    with pytest.raises(RuntimeError):
        jet(ops.Add(), (x, x), (y, y))
        _pynative_executor.sync()



@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_double_alpha(mode):
    """
    Features: Function jet
    Description: Test jet with double alpha
    Expectation: No exception.
    """
    class Net3(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sin = ops.Sin()
            self.cos = ops.Cos()

        def construct(self, x, y):
            s = self.sin(x)
            c = self.cos(y)
            return s * c, c * c

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = jet(self.net, (x, x), (y, y))
            return out, prime

    context.set_context(mode=mode)
    net = Net3()
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    ms_out, ms_prime = ms_net(x, y)
    assert np.allclose((ms_out[0].asnumpy(), ms_out[1].asnumpy()),
                                ([0.45464844, 0.45464844], [0.29192644, 0.29192644]), 0.0001,
                                0.0001)
    assert np.allclose((ms_prime[0].asnumpy(), ms_prime[1].asnumpy()),
                                ([[-0.4161465, -0.4161465], [-1.818592, -1.818592]
                                  ], [[-0.9092965, -0.9092965], [0.8322927, 0.8322927]]), 0.0001,
                                0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_jet_grad(mode):
    """
    Features: Function jet
    Description: Test jet jet grad
    Expectation: No exception.
    """
    class Grad(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, a, b):
            def get_jet(x, y):
                return jet(self.net, x, y)

            grad_net = grad(get_jet)
            grad_result = grad_net(a, b)
            return grad_result

    context.set_context(mode=mode)
    net = SinNet()
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Grad(net)
    jet_grad = ms_net(x, y)
    assert np.allclose(jet_grad.asnumpy(),\
        np.array([-8.41470957e-01, -8.41470957e-01]), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_jet_grad_jet_0_graph_mode():
    """
    Features: Function jet
    Description: Test jet grad_jet_0_graph_mode
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = SinNet()
    x = Tensor([1,], dtype.float32)
    y = Tensor([[1,], [0,]], dtype.float32)
    ms_out = jet(grad(net), x, y)
    assert np.allclose(ms_out[0].asnumpy(), (0.540302, ), 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), ([-0.8414702], [-0.5403015]), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_jet_jet(mode):
    """
    Features: Function jet
    Description: Test jet jet jet
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            _, self.seljet_net = jet(net, x, y)

        def construct(self, x, y):
            out, prime = jet(self.seljet_net, x, y)
            return out, prime

    context.set_context(mode=mode)
    net = SinNet()
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    with pytest.raises(RuntimeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_fn_ms_function(mode):
    """
    Features: Function jet
    Description: Test jet with jit
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = jet(self.net, x, y)
            return out, prime

    @jit
    def cos_exp(x):
        return ops.Cos()(ops.Exp()(x))

    context.set_context(mode=mode)
    net = cos_exp
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [-0.91173387, -0.91173387], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(),
                       [[-1.1166183, -1.1166183], [5.620221, 5.620221]], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_primitive(mode):
    """
    Features: Function jet
    Description: Test jet with primitive
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = jet(self.net, x, y)
            return out, prime

    context.set_context(mode=mode)
    net = ops.Sin()
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    with pytest.raises(RuntimeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_lambda(mode):
    """
    Features: Function jet
    Description: Test jet with lambda
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = jet(self.net, x, y)
            return out, prime

    context.set_context(mode=mode)
    net = lambda x: x * x * x     #pylint: disable=unnecessary-lambda-assignment
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [0.99999905, 0.99999905], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(),
                                [[2.9999971, 2.9999971], [5.9999943, 5.9999943]], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_parameter(mode):
    """
    Features: Function jet
    Description: Test jet with parameter
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out, prime = jet(self.net, x, y)
            return out, prime

    class Net6(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor([1, 2], dtype.float32), name="p")

        def construct(self, x):
            return self.param * x

    context.set_context(mode=mode)
    net = Net6()
    x = Tensor([1, 1], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    with pytest.raises(RuntimeError):
        ms_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_dyn(mode):
    """
    Features: Function jet
    Description: Test jet with dynamic shape
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out_1, out_2 = jet(self.net, x, y)
            return out_1, out_2

    context.set_context(mode=mode)
    net = SinNet()
    dyn = Tensor(shape=[None], dtype=dtype.float32)
    net.set_inputs(dyn)
    x = Tensor([2], dtype.float32)
    y = Tensor([[1], [0]], dtype.float32)
    ms_net = Net(net)
    ms_out = ms_net(x, y)
    assert np.allclose(ms_out[0].asnumpy(), [0.9092974], 0.0001, 0.0001)
    assert np.allclose(ms_out[1].asnumpy(), [[-0.41614667], [-0.9092965]], 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_jet_dyn_series_shape_not_same_as_primals(mode):
    """
    Features: Function jet
    Description: Test jet with dynamic shape not same as primals
    Expectation: Raise exception.
    """
    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x, y):
            out_1, out_2 = jet(self.net, x, y)
            return out_1, out_2

    context.set_context(mode=mode)
    net = SinNet()
    dyn = Tensor(shape=[None], dtype=dtype.float32)
    net.set_inputs(dyn)
    x = Tensor([2], dtype.float32)
    y = Tensor([[1, 1], [0, 0]], dtype.float32)
    ms_net = Net(net)
    with pytest.raises(ValueError):
        ms_net(x, y)
        _pynative_executor.sync()
