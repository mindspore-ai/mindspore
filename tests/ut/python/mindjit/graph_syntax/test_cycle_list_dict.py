# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import context


def test_cycle_container_structure():
    """
    Feature: Graph do not support container with cycle
    Description: Test container with cycle.
    Expectation: Runtime Error.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    a = [1, 2]
    a += [a]
    with pytest.raises(RuntimeError) as error_info:
        net(a)
    assert "Detect recursion when converting python object." in str(error_info.value)


def test_cycle_container_structure_2():
    """
    Feature: Graph do not support container with cycle
    Description: Test container with cycle.
    Expectation: Runtime Error.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    a = {"1": 1}
    a["2"] = a
    with pytest.raises(RuntimeError) as error_info:
        net(a)
    assert "Detect recursion when converting python object." in str(error_info.value)


def test_cycle_container_structure_3():
    """
    Feature: Graph do not support container with cycle
    Description: Test container with cycle.
    Expectation: Runtime Error.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    a = [1, 2, 3]
    b = [4, 5, 6]
    a[0] = b
    b[0] = a
    with pytest.raises(RuntimeError) as error_info:
        net(a)
    assert "Detect recursion when converting python object." in str(error_info.value)
    with pytest.raises(RuntimeError) as error_info:
        net(b)
    assert "Detect recursion when converting python object." in str(error_info.value)
