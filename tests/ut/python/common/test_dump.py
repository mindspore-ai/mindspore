# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test dump."""
import pytest

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import set_dump


def test_set_dump_on_cell():
    """
    Feature: Python API set_dump.
    Description: Use set_dump API on Cell instance.
    Expectation: Success.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(5, 6, 5, pad_mode='valid')
            self.relu1 = nn.ReLU()

        def construct(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            return x

    net = MyNet()
    set_dump(net.conv1)


def test_set_dump_on_primitive():
    """
    Feature: Python API set_dump.
    Description: Use set_dump API on Primitive instance.
    Expectation: Success.
    """
    op = ops.Add()
    set_dump(op)


def test_input_type_check():
    """
    Feature: Python API set_dump.
    Description: Use set_dump API on unsupported instance.
    Expectation: Throw ValueError exception.
    """
    with pytest.raises(ValueError):
        set_dump(1)
