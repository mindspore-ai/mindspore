# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function

context.set_context(device_target='GPU')


class NetOneHot(nn.Cell):
    def __init__(self):
        super(NetOneHot, self).__init__()
        self.on_value = 2.0
        self.off_value = 3.0

        self.depth_1 = 6
        self.one_hot_1 = nn.OneHot(-1, self.depth_1, self.on_value, self.off_value)

        self.depth_2 = 4
        self.one_hot_2 = nn.OneHot(0, self.depth_1, self.on_value, self.off_value)
        self.one_hot_3 = nn.OneHot(0, self.depth_2, self.on_value, self.off_value)
        self.one_hot_4 = nn.OneHot(1, self.depth_1, self.on_value, self.off_value)

    @ms_function
    def construct(self, indices1, indices2, indices3, indices4):
        return (self.one_hot_1(indices1), self.one_hot_2(indices2),
                self.one_hot_3(indices3), self.one_hot_4(indices4))


def one_hot(nptype):
    one_hot_net = NetOneHot()
    indices1 = Tensor(np.array([[0, 1], [4, 5], [2, 6]]).astype(nptype))
    indices2 = Tensor(np.array([1, 2, 3]).astype(nptype))
    indices3 = Tensor(np.array([[0, 1], [1, 0]]).astype(nptype))
    indices4 = Tensor(np.array([[0, 1], [4, 5], [2, 6]]).astype(nptype))
    output = one_hot_net(indices1, indices2, indices3, indices4)
    expect_0 = np.array([
        [[2., 3., 3., 3., 3., 3.], [3., 2., 3., 3., 3., 3.]],
        [[3., 3., 3., 3., 2., 3.], [3., 3., 3., 3., 3., 2.]],
        [[3., 3., 2., 3., 3., 3.], [3., 3., 3., 3., 3., 3.]]
    ]).astype(np.float32)
    expect_1 = np.array([
        [3., 3., 3.],
        [2., 3., 3.],
        [3., 2., 3.],
        [3., 3., 2.],
        [3., 3., 3.],
        [3., 3., 3.]
    ]).astype(np.float32)
    expect_2 = np.array([
        [[2., 3.], [3., 2.]], [[3., 2.], [2., 3.]], [[3., 3.], [3., 3.]],
        [[3., 3.], [3., 3.]]
    ]).astype(np.float32)
    expect_3 = np.array([
        [[2., 3.], [3., 2.], [3., 3.], [3., 3.], [3., 3.], [3., 3.]],
        [[3., 3.], [3., 3.], [3., 3.], [3., 3.], [2., 3.], [3., 2.]],
        [[3., 3.], [3., 3.], [2., 3.], [3., 3.], [3., 3.], [3., 3.]]
    ]).astype(np.float32)
    assert (output[0].asnumpy() == expect_0).all()
    assert (output[1].asnumpy() == expect_1).all()
    assert (output[2].asnumpy() == expect_2).all()
    assert (output[3].asnumpy() == expect_3).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_one_hot_int32():
    one_hot(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_one_hot_int64():
    one_hot(np.int64)
