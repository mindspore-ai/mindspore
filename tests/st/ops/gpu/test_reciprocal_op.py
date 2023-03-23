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


class NetReciprocal(nn.Cell):
    def __init__(self):
        super(NetReciprocal, self).__init__()
        self.reciprocal = P.Reciprocal()

    def construct(self, x):
        return self.reciprocal(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.bool_, np.int8, np.int16, np.int32, np.int64,
                                       np.uint8, np.uint16, np.uint32, np.uint64,
                                       np.float16, np.float32, np.float64,
                                       np.complex64, np.complex128])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reciprocal(data_type, mode):
    """
    Feature: ALL To ALL
    Description: test cases for Reciprocal
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(data_type)
    x1_np = np.random.uniform(-2, 2, 1).astype(data_type)
    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)
    expect0 = np.reciprocal(x0_np)
    error0 = np.ones(shape=expect0.shape) * 1.0e-3
    expect1 = np.reciprocal(x1_np)
    error1 = np.ones(shape=expect1.shape) * 1.0e-3

    context.set_context(mode=mode, device_target="GPU")
    reciprocal = NetReciprocal()
    output0 = reciprocal(x0)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
    output1 = reciprocal(x1)
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape
