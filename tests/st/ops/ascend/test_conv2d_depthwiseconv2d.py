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
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import Normal
from mindspore import Tensor


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_conv2d_depthwiseconv2d_str():
    net = nn.Conv2d(128, 128, (2, 3), stride=4, pad_mode='valid', padding=0, group=128, weight_init='normal')
    input_data = Tensor(np.ones([3, 128, 127, 114]), dtype=mstype.float32)
    output = net(input_data)
    assert output.shape == (3, 128, 32, 28)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_conv2d_depthwiseconv2d_initializer():
    net = nn.Conv2d(128, 128, (2, 3), stride=4, pad_mode='valid', padding=0, group=128, weight_init=Normal())
    input_data = Tensor(np.ones([3, 128, 127, 114]), dtype=mstype.float32)
    output = net(input_data)
    assert output.shape == (3, 128, 32, 28)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_conv2d_depthwiseconv2d_tensor():
    weight_init = Tensor(np.random.randn(128, 1, 2, 3).astype(np.float32))
    net = nn.Conv2d(128, 128, (2, 3), stride=4, pad_mode='valid', padding=0, group=128, weight_init=weight_init)
    input_data = Tensor(np.ones([3, 128, 127, 114]), dtype=mstype.float32)
    output = net(input_data)
    assert output.shape == (3, 128, 32, 28)
