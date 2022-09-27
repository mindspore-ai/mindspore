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

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class L2LossNet(nn.Cell):
    def __init__(self):
        super(L2LossNet, self).__init__()
        self.l2_loss = P.L2Loss()

    def construct(self, x):
        return self.l2_loss(x)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp32_22():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-4
    x = Tensor(np.array([[1., 2.], [3., 4.]]), ms.float32)
    expect = np.array(15, np.float32)
    output = P.L2Loss()(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp16_22():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-4
    x = Tensor(np.array([[1., 2.], [3., 4.]]), ms.float16)
    expect = np.array(15, np.float16)
    output = P.L2Loss()(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp32_14():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-4
    x = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    expect = np.array(15, np.float32)
    output = P.L2Loss()(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp16_14():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-4
    x = Tensor(np.array([1., 2., 3., 4.]), ms.float16)
    expect = np.array(15, np.float16)
    output = P.L2Loss()(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

def test_gather_graph_fp32_14():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    error = 1e-4
    x = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    expect = np.array(15, np.float32)
    l2_loss = L2LossNet()
    output = l2_loss(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

def test_gather_graph_fp16_14():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    error = 1e-4
    x = Tensor(np.array([1., 2., 3., 4.]), ms.float16)
    expect = np.array(15, np.float16)
    l2_loss = L2LossNet()
    output = l2_loss(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
