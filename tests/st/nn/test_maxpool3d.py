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

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def __init__(self, kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        super(Net, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding,
                                 dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def construct(self, x):
        out = self.pool(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_maxpool3d_normal(mode):
    """
    Feature: MaxPool3d
    Description: Verify the result of MaxPool3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.random.randint(0, 10, [5, 3, 4, 6, 7]), dtype=ms.float32)
    pool1 = Net(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=3, return_indices=False)
    output1 = pool1(x)

    pool2 = Net(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=3, return_indices=True)
    output2 = pool2(x)

    assert output1.shape == (5, 3, 3, 5, 6)
    assert output2[0].shape == (5, 3, 3, 5, 6)
    assert output2[1].shape == (5, 3, 3, 5, 6)
