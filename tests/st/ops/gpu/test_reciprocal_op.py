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
from tests.mark_utils import arg_mark

import numpy as np
import torch
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.bool_, np.int8, np.int16, np.int32, np.int64,
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
    expect0 = torch.reciprocal(torch.tensor(x0_np))
    error0 = np.ones(shape=expect0.shape) * 1.0e-3
    expect1 = torch.reciprocal(torch.tensor(x1_np))
    error1 = np.ones(shape=expect1.shape) * 1.0e-3

    context.set_context(mode=mode, device_target="GPU")
    reciprocal = NetReciprocal()
    output0 = reciprocal(x0)
    output0_np = output0.asnumpy()
    output0_np = np.where(np.isinf(output0_np), 0.0, output0_np)
    expect0_np = expect0.numpy()
    expect0_np = np.where(np.isinf(expect0_np), 0.0, expect0_np)
    diff0 = output0_np - expect0_np
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
    output1 = reciprocal(x1)
    output1_np = output1.asnumpy()
    output1_np = np.where(np.isinf(output1_np), 0.0, output1_np)
    expect1_np = expect1.numpy()
    expect1_np = np.where(np.isinf(expect1_np), 0.0, expect1_np)
    diff1 = output1_np - expect1_np
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape
