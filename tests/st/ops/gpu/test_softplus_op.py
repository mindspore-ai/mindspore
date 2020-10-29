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
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class SoftplusNet(nn.Cell):
    def __init__(self):
        super(SoftplusNet, self).__init__()
        self.softplus = P.Softplus()

    def construct(self, x):
        return self.softplus(x)


def SoftplusCompute(x):
    return np.log(1 + np.exp(x))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softplus_1d():
    x_np = np.random.random((50,)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softplus_2d():
    x_np = np.random.random((50, 40)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softplus_4d():
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softplus_neg():
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32) * -1
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softplus_4d_fp16():
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float16)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy(), rtol=5e-3)
