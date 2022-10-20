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
    def __init__(self, groups):
        super(Net, self).__init__()
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def construct(self, x):
        return self.channel_shuffle(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_channel_shuffle_normal(mode):
    """
    Feature: ChannelShuffle
    Description: Verify the result of ChannelShuffle
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net(2)
    x = ms.Tensor(np.arange(16).reshape((1, 4, 2, 2)), dtype=ms.int32)
    out = net(x)
    expect_out = np.array([[[[0, 1], [2, 3]], [[8, 9], [10, 11]],
                            [[4, 5], [6, 7]], [[12, 13], [14, 15]]]]).astype(np.int32)
    assert np.allclose(out.asnumpy(), expect_out)
