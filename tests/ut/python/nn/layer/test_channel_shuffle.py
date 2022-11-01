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
"""
test channel_shuffle api
"""

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor


class ChannelShuffleNet(nn.Cell):
    """ChannelShuffle"""
    def __init__(self, groups):
        super(ChannelShuffleNet, self).__init__()
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def construct(self, x):
        return self.channel_shuffle(x)


def test_compile_channel_shuffle():
    """
    Feature: Test ChannelShuffleNet
    Description: Test the functionality of ChannelShuffle
    Expectation: Success
    """
    net = ChannelShuffleNet(2)
    x = ms.Tensor(np.arange(16).astype(np.int32).reshape(1, 4, 2, 2))
    _cell_graph_executor.compile(net, x)
