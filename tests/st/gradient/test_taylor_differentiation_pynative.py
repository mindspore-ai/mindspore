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
"""test taylor differentiation in pynative mode"""
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.ops.functional import jet, derivative

context.set_context(mode=context.PYNATIVE_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def __init__(self):
        super(SingleInputSingleOutputNet, self).__init__()
        self.exp = P.Exp()
        self.cos = P.Cos()
        self.sin = P.Sin()

    def construct(self, x):
        out1 = self.sin(x)
        out2 = self.cos(out1)
        out3 = self.exp(out2)
        out = out1 + out2 - out3
        return out


class MultipleInputSingleOutputNet(nn.Cell):
    def __init__(self):
        super(MultipleInputSingleOutputNet, self).__init__()
        self.exp = P.Exp()
        self.cos = P.Cos()
        self.sin = P.Sin()

    def construct(self, x, y):
        out1 = self.sin(x)
        out2 = self.cos(y)
        out3 = out1 * out2 + out1 / out2
        out = self.exp(out3)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jet_multiple_input_single_output_pynative_mode():
    """
    Features: Function jet
    Description: Test jet with multiple inputs in pynative mode.
    Expectation: No exception.
    """
    series = (Tensor([[1., 1.], [0., 0.], [0., 0.]]), Tensor([[1., 1.], [0., 0.], [0., 0.]]))
    primals = (Tensor([1., 1.]), Tensor([1., 1.]))
    net = MultipleInputSingleOutputNet()
    expected_primals = np.array([7.47868, 7.47868]).astype(np.float32)
    expected_series = np.array([[22.50614, 22.50614], [133.92517, 133.92517], [1237.959, 1237.959]]).astype(np.float32)
    out_primals, out_series = jet(net, primals, series)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_derivative_multiple_input_single_output_pynative_mode():
    """
    Features: Function derivative
    Description: Test derivative with multiple inputs in pynative mode.
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
def test_jet_single_input_single_output_pynative_mode():
    """
    Features: Function jet
    Description: Test jet with single input in pynative mode.
    Expectation: No exception.
    """
    primals = Tensor([1., 1.])
    series = Tensor([[1., 1.], [0., 0.], [0., 0.]])
    net = SingleInputSingleOutputNet()
    expected_primals = np.array([-0.43931, -0.43931]).astype(np.float32)
    expected_series = np.array([[0.92187, 0.92187], [-1.56750, -1.56750], [-0.74808, -0.74808]]).astype(np.float32)
    out_primals, out_series = jet(net, primals, series)
    assert np.allclose(out_primals.asnumpy(), expected_primals, atol=1.e-4)
    assert np.allclose(out_series.asnumpy(), expected_series, atol=1.e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_derivative_single_input_single_output_pynative_mode():
    """
    Features: Function derivative
    Description: Test derivative with single input in pynative mode.
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
