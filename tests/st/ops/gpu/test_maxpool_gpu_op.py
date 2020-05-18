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


class Net_Pool(nn.Cell):
    def __init__(self):
        super(Net_Pool, self).__init__()
        self.maxpool_fun = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="VALID")

    def construct(self, x):
        return self.maxpool_fun(x)


class Net_Pool2(nn.Cell):
    def __init__(self):
        super(Net_Pool2, self).__init__()
        self.maxpool_fun = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="SAME")

    def construct(self, x):
        return self.maxpool_fun(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool2d():
    x = Tensor(np.array([[[
        [0, 1, 2, 3, -4, -5],
        [6, 7, 8, 9, -10, -11],
        [12, 13, 14, -15, -16, -17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float32))
    expect_result = (np.array([[[
        [7, 9, -4],
        [19, 21, 23],
        [31, 33, 35]
    ]]]))
    expect_result2 = (np.array([[[
        [14, 14, -4],
        [26, 28, 29],
        [32, 34, 35]
    ]]]))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    maxpool2d = Net_Pool()
    maxpool2d2 = Net_Pool2()
    output2 = maxpool2d2(x)
    output = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    maxpool2d = Net_Pool()
    maxpool2d2 = Net_Pool2()
    output2 = maxpool2d2(x)
    output = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()
