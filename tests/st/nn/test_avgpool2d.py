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
import mindspore.ops as ops


class Net(nn.Cell):
    def __init__(self, kernel_size=1, stride=1, pad_mode="valid", padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None, data_format='NCHW'):
        super(Net, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding,
                                 ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                 divisor_override=divisor_override, data_format=data_format)

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
def test_avgpool2d_normal(mode):
    """
    Feature: AvgPool2d
    Description: Verify the result of AvgPool2d
    Expectation: success
    """
    ms.set_context(mode=mode)
    x1 = ms.Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), ms.float32)
    pool1 = Net(kernel_size=3, stride=1)
    output1 = pool1(x1)

    x2 = ops.randn(6, 6, 8, 8)
    pool2 = Net(kernel_size=4, stride=1, pad_mode='pad', padding=2, divisor_override=5)
    output2 = pool2(x2)

    assert output1.shape == (1, 2, 2, 2)
    assert output2.shape == (6, 6, 9, 9)
