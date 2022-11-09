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

import pytest
from mindspore import nn
from mindspore.rewrite import SymbolTree


class SampleNet(nn.Cell):
    """SampleNet."""

    def __init__(self):
        """__init__"""
        super(SampleNet, self).__init__()
        self.avg_pool = nn.AvgPool2d(5)
        self.conv2d_0 = nn.Conv2d(64, 128, 1)
        self.conv2d_1 = nn.Conv2d(64, 128, 1)
        self.is_training = True

    def construct(self, x):
        """construct."""
        x = self.avg_pool(x)
        conv0 = self.conv2d_0(x)
        conv1 = self.conv2d_1(x)
        if self.is_training:
            return conv0, conv1
        return conv0


def test_log():
    """
    Feature: Test rewrite error log.
    Description: Rewrite sample network and expect output 'return (conv0, conv1)'.
    Expectation: Success.
    """
    net = SampleNet()
    with pytest.raises(RuntimeError) as err:
        _ = SymbolTree.create(net)
    assert "return (conv0, conv1)" in err.value.args[0]
