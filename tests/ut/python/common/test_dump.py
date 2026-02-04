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
import warnings

import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.tools import set_dump


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
    set_dump(net.relu1)

    assert net.relu1.get_flags()["dump"] is True


def test_set_dump_on_primitive():
    """
    Feature: Python API set_dump.
    Description: Use set_dump API on Primitive instance.
    Expectation: Success.
    """
    op = ops.Add()
    set_dump(op)
    assert op.attrs["dump"] == "true"


def test_input_type_check():
    """
    Feature: Python API set_dump.
    Description: Use set_dump API on unsupported instance.
    Expectation: Throw ValueError exception.
    """
    with pytest.raises(ValueError):
        set_dump(1)


@pytest.mark.skip(reason="Warning can only be triggered once, please execute "
                         "this test case manually.")
def test_set_dump_warning():
    """
    Feature: Python API set_dump.
    Description: Test the warning about device target and mode.
    Expectation: Trigger warning message.
    """
    context.set_context(device_target="CPU")
    context.set_context(mode=context.PYNATIVE_MODE)
    op = ops.Add()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        set_dump(op)
        assert "Only Ascend device target is supported" in str(w[-2].message)
        assert "Only GRAPH_MODE is supported" in str(w[-1].message)


def test_set_dump_on_cell_with_false():
    """
    Feature: Python API set_dump on cell with False.
    Description: Use set_dump API on Cell instance.
    Expectation: Success.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.relu1 = nn.ReLU()

        def construct(self, x):
            x = self.relu1(x)
            return x

    net = MyNet()
    set_dump(net.relu1)
    assert net.relu1.get_flags()["dump"] is True

    set_dump(net, False)
    assert net.relu1.get_flags()["dump"] is False


def test_set_dump_on_primitive_with_false():
    """
    Feature: Python API set_dump on primitive with False.
    Description: Use set_dump API on Cell instance.
    Expectation: Success.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.relu1 = ops.ReLU()

        def construct(self, x):
            x = self.relu1(x)
            return x

    net = MyNet()
    set_dump(net.relu1)
    assert net.relu1.attrs.get("dump") == "true"

    set_dump(net, False)
    assert net.relu1.attrs.get("dump") == "false"
