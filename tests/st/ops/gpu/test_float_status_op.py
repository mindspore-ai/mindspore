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

import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetFloatStatus(nn.Cell):
    def __init__(self):
        super(NetFloatStatus, self).__init__()
        self.status = P.FloatStatus()

    def construct(self, x):
        return self.status(x)


class NetIsNan(nn.Cell):
    def __init__(self):
        super(NetIsNan, self).__init__()
        self.isnan = P.IsNan()

    def construct(self, x):
        return self.isnan(x)


class NetIsInf(nn.Cell):
    def __init__(self):
        super(NetIsInf, self).__init__()
        self.isinf = P.IsInf()

    def construct(self, x):
        return self.isinf(x)


class NetIsFinite(nn.Cell):
    def __init__(self):
        super(NetIsFinite, self).__init__()
        self.isfinite = P.IsFinite()

    def construct(self, x):
        return self.isfinite(x)


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
x1 = np.array([[1.2, 2, np.nan, 88]]).astype(np.float32)
x2 = np.array([[np.inf, 1, 88.0, 0]]).astype(np.float32)
x3 = np.array([[1, 2], [3, 4], [5.0, 88.0]]).astype(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_status(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for FloatStatus
    Expectation: the result match to expectation
    """
    ms_status = NetFloatStatus()
    output1 = ms_status(Tensor(x1.astype(dtype)))
    expect1 = 1
    assert output1.asnumpy()[0] == expect1

    output2 = ms_status(Tensor(x2.astype(dtype)))
    expect2 = 1
    assert output2.asnumpy()[0] == expect2

    output3 = ms_status(Tensor(x3.astype(dtype)))
    expect3 = 0
    assert output3.asnumpy()[0] == expect3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_nan(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for IsNan
    Expectation: the result match to expectation
    """
    ms_isnan = NetIsNan()
    output1 = ms_isnan(Tensor(x1.astype(dtype)))
    expect1 = [[False, False, True, False]]
    assert (output1.asnumpy() == expect1).all()

    output2 = ms_isnan(Tensor(x2.astype(dtype)))
    expect2 = [[False, False, False, False]]
    assert (output2.asnumpy() == expect2).all()

    output3 = ms_isnan(Tensor(x3.astype(dtype)))
    expect3 = [[False, False], [False, False], [False, False]]
    assert (output3.asnumpy() == expect3).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_inf(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for IsInf
    Expectation: the result match to expectation
    """
    ms_isinf = NetIsInf()
    output1 = ms_isinf(Tensor(x1.astype(dtype)))
    expect1 = [[False, False, False, False]]
    assert (output1.asnumpy() == expect1).all()

    output2 = ms_isinf(Tensor(x2.astype(dtype)))
    expect2 = [[True, False, False, False]]
    assert (output2.asnumpy() == expect2).all()

    output3 = ms_isinf(Tensor(x3.astype(dtype)))
    expect3 = [[False, False], [False, False], [False, False]]
    assert (output3.asnumpy() == expect3).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_finite(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Netfinite
    Expectation: the result match to expectation
    """
    ms_isfinite = NetIsFinite()
    output1 = ms_isfinite(Tensor(x1.astype(dtype)))
    expect1 = [[True, True, False, True]]
    assert (output1.asnumpy() == expect1).all()

    output2 = ms_isfinite(Tensor(x2.astype(dtype)))
    expect2 = [[False, True, True, True]]
    assert (output2.asnumpy() == expect2).all()

    output3 = ms_isfinite(Tensor(x3.astype(dtype)))
    expect3 = [[True, True], [True, True], [True, True]]
    assert (output3.asnumpy() == expect3).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_float_status_gpu_dynamic_shape():
    """
    Feature: test FloatStatus op on GPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetFloatStatus()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, ms.float32))
    except_shape = (1,)
    assert output.asnumpy().shape == except_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_nan_gpu_dynamic_shape():
    """
    Feature: test FloatStatus op on GPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIsNan()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, ms.float32))
    except_shape = (1, 32, 9, 9)
    assert output.asnumpy().shape == except_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_inf_gpu_dynamic_shape():
    """
    Feature: test FloatStatus op on GPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIsInf()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, ms.float32))
    except_shape = (1, 32, 9, 9)
    assert output.asnumpy().shape == except_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_finite_gpu_dynamic_shape():
    """
    Feature: test FloatStatus op on GPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetIsFinite()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, ms.float32))
    except_shape = (1, 32, 9, 9)
    assert output.asnumpy().shape == except_shape
