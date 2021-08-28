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


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.expand_dims = P.ExpandDims()

    def construct(self, tensor):
        return self.expand_dims(tensor, -1)


class NetDynamic(nn.Cell):
    def __init__(self):
        super(NetDynamic, self).__init__()
        self.conv = inner.GpuConvertToDynamicShape()
        self.expand_dims = P.ExpandDims()

    def construct(self, x):
        x_conv = self.conv(x)
        return self.expand_dims(x_conv, -1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_bool():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.bool)
    net = NetDynamic()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int8():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int8)
    net = NetDynamic()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_uint8():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.uint8)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int16)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int64)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.float16)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.float32)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float64():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.float64)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))
