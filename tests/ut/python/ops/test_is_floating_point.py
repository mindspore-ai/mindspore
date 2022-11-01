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
test is floating point api
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import _cell_graph_executor


class Roll(nn.Cell):
    def construct(self, x):
        return ops.is_floating_point(x)


def test_compile_is_floating_point():
    """
    Feature: Test is floating point
    Description: Test the functionality of is floating point
    Expectation: Success
    """
    net = Roll()
    x = ms.Tensor([1, 2, 3], ms.float32)
    _cell_graph_executor.compile(net, x)
