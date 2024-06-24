# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import ops as P
from mindspore import Tensor, nn
from mindspore.common.initializer import One

import numpy as np
import pytest


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ss = P.StridedSlice()
        self.neg = P.Neg()

    def construct(self, x, y, z):
        x = self.ss(x, y, z, (1, 1))
        x = self.neg(x)
        return x


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_feed_input_dynamic():
    """
    Feature: Test StridedSlice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the output is dynamic caused by tensor input with begin/end.
    Expectation: Assert the result is equal the numpy result.
    """
    net = Net()
    x = Tensor(np.ones((4, 5)), ms.float32)
    y = Tensor((1, 1), ms.int64)
    z = Tensor((4, 5), ms.int64)

    dyn_x = Tensor(shape=[4, None], dtype=ms.float32)
    dyn_y = Tensor(shape=[2], dtype=ms.int64, init=One())
    dyn_z = Tensor(shape=[2], dtype=ms.int64, init=One())

    net.set_inputs(dyn_x, dyn_y, dyn_z)

    expect = np.negative(np.ones([3, 4]))
    out = net(x, y, z)
    tol = 1e-6
    assert (np.abs(out.asnumpy() - expect) < tol).all()


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_feed_output_dynamic():
    """
    Feature: Test StridedSlice for dynamic shape in feed mode.
    Description: The input shape is static but the output is dynamic caused by tensor input with begin/end.
    Expectation: Assert the result is equal the numpy result.
    """
    net = Net()
    x = Tensor(np.ones((4, 5)), ms.float32)
    y = Tensor((1, 1), ms.int64)
    z = Tensor((4, 5), ms.int64)

    expect = np.negative(np.ones([3, 4]))
    out = net(x, y, z)
    tol = 1e-6
    assert (np.abs(out.asnumpy() - expect) < tol).all()
