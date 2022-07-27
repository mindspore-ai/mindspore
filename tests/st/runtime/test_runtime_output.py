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

import pytest
from mindspore import context, nn


class NetValueNodeWithDepend(nn.Cell):
    def construct(self, input_x):
        print(input_x[1][1])
        output = input_x[1]
        output[1] = 0
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_value_node_with_depend():
    """
    Feature: Runtime special output.
    Description: Test the output is the depend with value node, that the value can't be converted the tensor.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = [[1, 2, 3, 4], [5, 6, 7, 8]]
    net = NetValueNodeWithDepend()
    output = net(x)
    assert output == (5, 0, 7, 8)
