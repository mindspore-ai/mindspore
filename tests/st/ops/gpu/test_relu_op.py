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
from mindspore.ops.operations import _inner_ops as inner


class NetRelu(nn.Cell):
    def __init__(self):
        super(NetRelu, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


class NetReluDynamic(nn.Cell):
    def __init__(self):
        super(NetReluDynamic, self).__init__()
        self.conv = inner.GpuConvertToDynamicShape()
        self.relu = P.ReLU()

    def construct(self, x):
        x_conv = self.conv(x)
        return self.relu(x_conv)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_float32():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    expect = np.array([[[[0, 1, 10,],
                         [1, 0, 1,],
                         [10, 1, 0.]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_int8():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.int8))
    expect = np.array([[[[0, 1, 10,],
                         [1, 0, 1,],
                         [10, 1, 0.]]]]).astype(np.int8)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_int32():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.int32))
    expect = np.array([[[[0, 1, 10,],
                         [1, 0, 1,],
                         [10, 1, 0.]]]]).astype(np.int32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_int64():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.int64))
    expect = np.array([[[[0, 1, 10,],
                         [1, 0, 1,],
                         [10, 1, 0.]]]]).astype(np.int64)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    print(output.asnumpy(), expect)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu = NetRelu()
    output = relu(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_int64_dynamic_shape():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.int64))
    expect = np.array([[[[0, 1, 10,],
                         [1, 0, 1,],
                         [10, 1, 0.]]]]).astype(np.int64)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu_dynamic = NetReluDynamic()
    output = relu_dynamic(x)
    assert (output.asnumpy() == expect).all()
