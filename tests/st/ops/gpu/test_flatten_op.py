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


class NetFlatten(nn.Cell):
    def __init__(self):
        super(NetFlatten, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x):
        return self.flatten(x)


class NetAllFlatten(nn.Cell):
    def __init__(self):
        super(NetAllFlatten, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x):
        loop_count = 4
        while loop_count > 0:
            x = self.flatten(x)
            loop_count = loop_count - 1
        return x


class NetFirstFlatten(nn.Cell):
    def __init__(self):
        super(NetFirstFlatten, self).__init__()
        self.flatten = P.Flatten()
        self.relu = P.ReLU()

    def construct(self, x):
        loop_count = 4
        while loop_count > 0:
            x = self.flatten(x)
            loop_count = loop_count - 1
        x = self.relu(x)
        return x


class NetLastFlatten(nn.Cell):
    def __init__(self):
        super(NetLastFlatten, self).__init__()
        self.flatten = P.Flatten()
        self.relu = P.ReLU()

    def construct(self, x):
        loop_count = 4
        x = self.relu(x)
        while loop_count > 0:
            x = self.flatten(x)
            loop_count = loop_count - 1
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_all_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetAllFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetAllFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_first_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[0, 0.3, 3.6], [0.4, 0.5, 0]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetFirstFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetFirstFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_last_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[0, 0.3, 3.6], [0.4, 0.5, 0]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetLastFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetLastFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()
    