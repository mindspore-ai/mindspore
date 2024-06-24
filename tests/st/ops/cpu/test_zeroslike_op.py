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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetZerosLike(nn.Cell):
    def __init__(self):
        super(NetZerosLike, self).__init__()
        self.zeros_like = P.ZerosLike()

    def construct(self, x):
        return self.zeros_like(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.float64])
def test_ZerosLike(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for ZerosLike
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(dtype)
    x1_np = np.random.uniform(-2, 2, 1).astype(dtype)

    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    zeros_like = NetZerosLike()
    output0 = zeros_like(x0)
    expect0 = np.zeros_like(x0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = zeros_like(x1)
    expect1 = np.zeros_like(x1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape
