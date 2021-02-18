# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

class SqueezeNet(nn.Cell):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.squeeze = P.Squeeze()

    def construct(self, tensor):
        return self.squeeze(tensor)


def squeeze(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(nptype)
    net = SqueezeNet()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == x.squeeze())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_bool():
    squeeze(np.bool)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_uint8():
    squeeze(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_uint16():
    squeeze(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_uint32():
    squeeze(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_int8():
    squeeze(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_int16():
    squeeze(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_int32():
    squeeze(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_int64():
    squeeze(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_float16():
    squeeze(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_float32():
    squeeze(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_float64():
    squeeze(np.float64)
