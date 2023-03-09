# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import function as F


class NetLog(nn.Cell):
    def __init__(self):
        super(NetLog, self).__init__()
        self.log = P.Log()

    def construct(self, x):
        return self.log(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_log(data_type, mode):
    """
    Feature: Log
    Description: test cases for Log
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(1, 2, (2, 3, 4, 4)).astype(data_type)
    x1_np = np.random.uniform(1, 2, 1).astype(data_type)
    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)
    expect0 = np.log(x0_np)
    expect1 = np.log(x1_np)

    context.set_context(mode=mode, device_target="GPU")
    log = NetLog()
    output0 = log(x0)
    output1 = log(x1)

    assert output0.shape == expect0.shape
    assert output1.shape == expect1.shape

    np.allclose(output0.asnumpy(), expect0)
    np.allclose(output1.asnumpy(), expect1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_func(data_type):
    """
    Feature: Log
    Description: test cases for Log
    Expectation: the result match to numpy
    """
    x = np.random.randn(2, 3, 3, 4).astype(data_type)
    y_expect = np.log(x)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    tensor = Tensor(x)
    out = F.log(tensor)

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    tensor = Tensor(x)
    out = F.log(tensor)

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_tensor(data_type):
    """
    Feature: Log
    Description: test cases for Log
    Expectation: the result match to numpy
    """
    x = np.random.randn(2, 3, 3, 4).astype(data_type)
    y_expect = np.log(x)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    tensor = Tensor(x)
    out = tensor.log()

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    tensor = Tensor(x)
    out = tensor.log()

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)
