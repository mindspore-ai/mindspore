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
""" test image gradients """
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _executor
from mindspore.common.api import ms_function

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.image_gradients = nn.ImageGradients()

    @ms_function
    def construct(self, x):
        return self.image_gradients(x)


def test_compile():
    # input shape 1 x 1 x 2 x 2
    image = Tensor(np.array([[[[1, 2], [3, 4]]]]), dtype=mstype.int32)
    net = Net()
    _executor.compile(net, image)


def test_compile_multi_channel():
    # input shape 4 x 2 x 2 x 2
    dtype = mstype.int32
    image = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             [[[3, 5], [7, 9]], [[11, 13], [15, 17]]],
                             [[[5, 10], [15, 20]], [[25, 30], [35, 40]]],
                             [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]]), dtype=dtype)
    net = Net()
    _executor.compile(net, image)


def test_invalid_5d_input():
    dtype = mstype.float32
    image = Tensor(np.random.random([4, 1, 16, 16, 1]), dtype=dtype)
    net = Net()
    with pytest.raises(ValueError):
        _executor.compile(net, image)
