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
"""test taylor differentiation in graph mode"""
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import ops
from mindspore import Tensor
from mindspore.ops.functional import jet, derivative

context.set_context(mode=context.GRAPH_MODE)


class MultipleInputSingleOutputNet(nn.Cell):
    def __init__(self):
        super(MultipleInputSingleOutputNet, self).__init__()
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
        super(MultipleInputMultipleOutputNet, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x, y):
        out1 = self.sin(x)
        out2 = self.cos(y)
        return out1, out2


class SingleInputSingleOutputNet(nn.Cell):
    def __init__(self):
        super(SingleInputSingleOutputNet, self).__init__()
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
        super(SingleInputSingleOutputWithScalarNet, self).__init__()
        self.log = ops.Log()

    def construct(self, x):
        out1 = self.log(x)
        out = 1 / out1 + 2
        return out * 3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jet_single_input_single_output_graph_mode():
    """
    Features: Function jet
    Description: Test jet with single input in graph mode.
    Expectation: No exception.
    """
    primals = Tensor([1., 1.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    net = SingleInputSingleOutputNet()
    expected_primals = np.array([-0.43931, -0.43931]).astype(np.float32)
    expected_series = np.array([[0.92187, 0.92187], [-1.56750, -1.56750], [-0.74808, -0.74808]]).astype(np.float32)
    out_primals, out_series = jet(net, primals, series)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jet_single_input_single_output_with_scalar_graph_mode():
    """
    Features: Function jet
    Description: Test jet with single input with scalar in graph mode.
    Expectation: No exception.
    """
    primals = Tensor([2., 2.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    net = SingleInputSingleOutputWithScalarNet()
    out_primals, out_series = jet(net, primals, series)
    expected_primals = np.array([10.328085, 10.328085]).astype(np.float32)
    expected_series = np.array([[-3.1220534, -3.1220534], [6.0652323, 6.0652323],
                                [-18.06463, -18.06463]]).astype(np.float32)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_derivative_single_input_single_output_graph_mode():
    """
    Features: Function derivative
    Description: Test derivative with single input in graph mode.
    Expectation: No exception.
    """
    primals = Tensor([1., 1.])
    order = 3
    net = SingleInputSingleOutputNet()
    expected_primals = np.array([-0.43931, -0.43931]).astype(np.float32)
    expected_series = np.array([-0.74808, -0.74808]).astype(np.float32)
    out_primals, out_series = derivative(net, primals, order)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jet_multiple_input_single_output_graph_mode():
    """
    Features: Function jet
    Description: Test jet with multiple inputs in graph mode.
    Expectation: No exception.
    """
    primals = (Tensor([1., 1.]), Tensor([1., 1.]))
    series = (Tensor([[1., 1.], [0., 0.], [0., 0.]]), Tensor([[1., 1.], [0., 0.], [0., 0.]]))
    net = MultipleInputSingleOutputNet()
    expected_primals = np.array([7.47868, 7.47868]).astype(np.float32)
    expected_series = np.array([[22.50614, 22.50614], [133.92517, 133.92517], [1237.959, 1237.959]]).astype(np.float32)
    out_primals, out_series = jet(net, primals, series)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_derivative_multiple_input_single_output_graph_mode():
    """
    Features: Function derivative
    Description: Test derivative with multiple inputs in graph mode.
    Expectation: No exception.
    """
    primals = (Tensor([1., 1.]), Tensor([1., 1.]))
    order = 3
    net = MultipleInputSingleOutputNet()
    expected_primals = np.array([7.47868, 7.47868]).astype(np.float32)
    expected_series = np.array([1237.959, 1237.959]).astype(np.float32)
    out_primals, out_series = derivative(net, primals, order)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jet_construct_graph_mode():
    """
    Features: Function jet
    Description: Test jet in construct with multiple inputs in graph mode.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_derivative_construct_graph_mode():
    """
    Features: Function derivative
    Description: Test derivative in construct with multiple inputs in graph mode.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, net, order):
            super(Net, self).__init__()
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


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jet_function_graph_mode():
    """
    Features: Function jet
    Description: Test function in graph mode.
    Expectation: No exception.
    """
    primals = Tensor([1., 1.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    out_primals, out_series = jet(function_graph, primals, series)
    expected_primals = np.array([-0.450549, -0.450549]).astype(np.float32)
    expected_series = np.array([[3.270079, 3.270079], [-4.739784, -4.739784],
                                [56.995613, 56.995613]]).astype(np.float32)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
