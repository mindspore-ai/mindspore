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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import _cell_graph_executor


class Net(nn.Cell):
    def construct(self, x):
        output = ops.rsqrt(x)
        return output


def test_rsqrt_normal():
    """
    Feature: Test rsqrt
    Description: Test the functionality of rsqrt
    Expectation: Success
    """
    net = Net()
    x = ms.Tensor([-0.0370, 0.2970, 1.5420, -0.9105])
    _cell_graph_executor.compile(net, x)
