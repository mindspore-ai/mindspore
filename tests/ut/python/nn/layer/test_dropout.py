# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
test pooling api
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor


class Dropout1dNet(nn.Cell):
    def __init__(self, p):
        super(Dropout1dNet, self).__init__()
        self.dropout1 = nn.Dropout1d(p)

    def construct(self, x):
        return self.dropout1(x)


def test_dropout1_normal():
    """
    Feature: dropout1d
    Description: Verify the result of Dropout1d
    Expectation: success
    """
    x = Tensor(np.random.randn(4, 3).astype(np.float32))
    net = Dropout1dNet(p=0.5)
    _cell_graph_executor.compile(net, x)
