# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.ops import operations as P


def cum_prod(nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x0 = np.random.rand(2, 3, 4, 4).astype(nptype)
    axis0 = 3

    x1 = np.random.rand(2, 3, 4, 4).astype(nptype)
    axis1 = 3

    x2 = np.random.rand(2, 3, 1, 4).astype(nptype)
    axis2 = 2

    x3 = np.random.rand(2, 3, 1, 4).astype(nptype)
    axis3 = 2

    x4 = np.random.rand(2, 3, 4, 4).astype(nptype)
    axis4 = 1

    x5 = np.random.rand(2, 3).astype(nptype)
    axis5 = 1

    x6 = np.random.rand(1, 1, 1, 1).astype(nptype)
    axis6 = 0

    class CumProd(nn.Cell):
        def __init__(self, nptype):
            super(CumProd, self).__init__()

            self.x0 = Tensor(x0)
            self.axis0 = axis0

            self.x1 = Tensor(x1)
            self.axis1 = axis1

            self.x2 = Tensor(x2)
            self.axis2 = axis2

            self.x3 = Tensor(x3)
            self.axis3 = axis3

            self.x4 = Tensor(x4)
            self.axis4 = axis4

            self.x5 = Tensor(x5)
            self.axis5 = axis5

            self.x6 = Tensor(x6)
            self.axis6 = axis6

        @ms_function
        def construct(self):
            return (P.CumProd()(self.x0, self.axis0),
                    P.CumProd()(self.x1, self.axis1),
                    P.CumProd()(self.x2, self.axis2),
                    P.CumProd()(self.x3, self.axis3),
                    P.CumProd()(self.x4, self.axis4),
                    P.CumProd()(self.x5, self.axis5),
                    P.CumProd()(self.x6, self.axis6))

    cumprod = CumProd(nptype)
    output = cumprod()

    expect0 = np.cumprod(x0, axis=axis0)
    diff0 = abs(output[0].asnumpy() - expect0)
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0].shape == expect0.shape

    expect1 = np.cumprod(x1, axis=axis1)
    diff1 = abs(output[1].asnumpy() - expect1)
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1].shape == expect1.shape

    expect2 = np.cumprod(x2, axis=axis2)
    diff2 = abs(output[2].asnumpy() - expect2)
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2].shape == expect2.shape

    expect3 = np.cumprod(x3, axis=axis3)
    diff3 = abs(output[3].asnumpy() - expect3)
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output[3].shape == expect3.shape

    expect4 = np.cumprod(x4, axis=axis4)
    diff4 = abs(output[4].asnumpy() - expect4)
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output[4].shape == expect4.shape

    expect5 = np.cumprod(x5, axis=axis5)
    diff5 = abs(output[5].asnumpy() - expect5)
    error5 = np.ones(shape=expect5.shape) * 1.0e-5
    assert np.all(diff5 < error5)
    assert output[5].shape == expect5.shape

    expect6 = np.cumprod(x6, axis=axis6)
    diff6 = abs(output[6].asnumpy() - expect6)
    error6 = np.ones(shape=expect6.shape) * 1.0e-5
    assert np.all(diff6 < error6)
    assert output[6].shape == expect6.shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cum_prod_uint8():
    cum_prod(np.uint8)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cum_prod_int8():
    cum_prod(np.int8)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cum_prod_int32():
    cum_prod(np.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cum_prod_float16():
    cum_prod(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cum_prod_float32():
    cum_prod(np.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = P.CumProd()

    def construct(self, x):
        return self.op(x, 0)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cumprod_dshape():
    """
    Feature: Test cumprod dynamic shape.
    Description: Test cumprod dynamic shape.
    Expectation: Success.
    """
    net = Net()
    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn)
    input_x = Tensor(np.random.random(([3, 10])), dtype=ms.float32)
    output = net(input_x)
    expect_shape = (3, 10)
    assert output.asnumpy().shape == expect_shape
