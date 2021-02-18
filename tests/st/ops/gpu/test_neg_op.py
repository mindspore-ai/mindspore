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


class NetNeg(nn.Cell):
    def __init__(self):
        super(NetNeg, self).__init__()
        self.neg = P.Neg()

    def construct(self, x):
        return self.neg(x)


def neg(nptype):
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    x1_np = np.random.uniform(-2, 2, 1).astype(nptype)
    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)
    expect0 = np.negative(x0_np)
    expect1 = np.negative(x1_np)
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    error1 = np.ones(shape=expect1.shape) * 1.0e-5

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    neg_net = NetNeg()
    output0 = neg_net(x0)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
    output1 = neg_net(x1)
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    neg_net = NetNeg()
    output0 = neg_net(x0)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
    output1 = neg_net(x1)
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_neg_float16():
    neg(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_neg_float32():
    neg(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_neg_float64():
    neg(np.float64)
