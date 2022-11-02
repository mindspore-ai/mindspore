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
test pooling api
"""
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import _cell_graph_executor


class Roll(nn.Cell):
    def construct(self, x):
        return ops.roll(x, shifts=2, dims=0)


def test_compile_roll():
    """
    Feature: Test Roll
    Description: Test the functionality of roll
    Expectation: Success
    """
    net = Roll()
    x = ms.Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32))
    _cell_graph_executor.compile(net, x)
