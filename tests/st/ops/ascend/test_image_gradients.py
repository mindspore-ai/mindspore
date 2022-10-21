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

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.image_gradients = nn.ImageGradients()

    @jit
    def construct(self, x):
        return self.image_gradients(x)


def test_image_gradients():
    image = Tensor(np.array([[[[1, 2], [3, 4]]]]), dtype=mstype.int32)
    expected_dy = np.array([[[[2, 2], [0, 0]]]]).astype(np.int32)
    expected_dx = np.array([[[[1, 0], [1, 0]]]]).astype(np.int32)
    net = Net()
    dy, dx = net(image)
    assert not np.any(dx.asnumpy() - expected_dx)
    assert not np.any(dy.asnumpy() - expected_dy)


def test_image_gradients_multi_channel_depth():
    # 4 x 2 x 2 x 2
    dtype = mstype.int32
    image = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             [[[3, 5], [7, 9]], [[11, 13], [15, 17]]],
                             [[[5, 10], [15, 20]], [[25, 30], [35, 40]]],
                             [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]]), dtype=dtype)
    expected_dy = Tensor(np.array([[[[2, 2], [0, 0]], [[2, 2], [0, 0]]],
                                   [[[4, 4], [0, 0]], [[4, 4], [0, 0]]],
                                   [[[10, 10], [0, 0]], [[10, 10], [0, 0]]],
                                   [[[20, 20], [0, 0]], [[20, 20], [0, 0]]]]), dtype=dtype)
    expected_dx = Tensor(np.array([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]],
                                   [[[2, 0], [2, 0]], [[2, 0], [2, 0]]],
                                   [[[5, 0], [5, 0]], [[5, 0], [5, 0]]],
                                   [[[10, 0], [10, 0]], [[10, 0], [10, 0]]]]), dtype=dtype)
    net = Net()
    dy, dx = net(image)

    assert not np.any(dx.asnumpy() - expected_dx.asnumpy())
    assert not np.any(dy.asnumpy() - expected_dy.asnumpy())
