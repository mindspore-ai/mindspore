# Copyright 2023 Huawei Technologies Co., Ltd
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

""" Test conv2d. """

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, nn, Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, w):
        return ops.conv2d(x, w, None, (1, 1), 'pad', (1, 1), (1, 1), 1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_conv2d_forward_fp16():
    """
    Feature: conv2d.
    Description: test op conv2d with fp16 input.
    Expectation: expect correct result.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    x_np = np.random.randn(2, 4, 5, 5).astype(np.float16)
    w_np = np.random.randn(2, 4, 3, 3).astype(np.float16)

    x = Tensor(x_np)
    w = Tensor(w_np)

    net = Net()

    ms.set_context(device_target='CPU')
    out_c = net(x, w)

    ms.set_context(device_target='Ascend')
    out_d = net(x, w)

    error = 1.0e-3
    assert np.allclose(out_c.asnumpy(), out_d.asnumpy(), error, error)
