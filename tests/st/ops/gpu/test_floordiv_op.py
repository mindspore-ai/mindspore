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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

class NetFloorDiv(nn.Cell):
    def __init__(self):
        super(NetFloorDiv, self).__init__()
        self.floordiv = P.FloorDiv()

    def construct(self, x, y):
        return self.floordiv(x, y)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_floor_div():
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.float32)
    x2_np = np.random.randint(1, 5, (2, 1, 1, 4, 9)).astype(np.float32)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4, 9)).astype(np.float32)
    x3_np = np.random.randint(1, 5, 1).astype(np.float32)
    y3_np = np.random.randint(1, 5, 1).astype(np.float32)
    x4_np = np.array(768).astype(np.float32)
    y4_np = np.array(3072.5).astype(np.float32)
    x5_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    y5_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    x6_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.int32)
    y6_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.int32)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)
    x5 = Tensor(x5_np)
    y5 = Tensor(y5_np)
    x6 = Tensor(x6_np)
    y6 = Tensor(y6_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    floor_div = NetFloorDiv()
    output0 = floor_div(x0, y0)
    expect0 = np.floor_divide(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = floor_div(x1, y1)
    expect1 = np.floor_divide(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = floor_div(x2, y2)
    expect2 = np.floor_divide(x2_np, y2_np)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    output3 = floor_div(x3, y3)
    expect3 = np.floor_divide(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = floor_div(x4, y4)
    expect4 = np.floor_divide(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    output5 = floor_div(x5, y5)
    expect5 = np.floor_divide(x5_np, y5_np)
    diff5 = output5.asnumpy() - expect5
    error5 = np.ones(shape=expect5.shape) * 1.0e-5
    assert np.all(diff5 < error5)
    assert output5.shape == expect5.shape

    output6 = floor_div(x6, y6)
    expect6 = np.floor_divide(x6_np, y6_np)
    diff6 = output6.asnumpy() - expect6
    error6 = np.ones(shape=expect6.shape) * 1.0e-5
    assert np.all(diff6 < error6)
    assert output6.shape == expect6.shape
