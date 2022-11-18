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
test arange api
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import _cell_graph_executor


class Net(nn.Cell):
    def construct(self, start=0, end=None, step=1, dtype=None):
        return ops.arange(start, end, step, dtype=dtype)


def test_arange_normal():
    """
    Feature: arange
    Description: Test the functionality of arange
    Expectation: success
    """
    net = Net()
    _cell_graph_executor.compile(net, 1, 6)
