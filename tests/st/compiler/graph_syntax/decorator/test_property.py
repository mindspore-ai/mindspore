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
""" test property decorator in graph mode. """
import pytest
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter, jit
from tests.mark_utils import arg_mark


ms.set_context(mode=ms.GRAPH_MODE)


class PropertyNet(nn.Cell):
    def __init__(self, p):
        super().__init__()
        p = Tensor(p, ms.float32)
        self.p = Parameter(p, name='param_p', requires_grad=False)
        self.identity = ops.Identity()

    @property
    def get_parameter(self):
        return self.identity(self.p)

    def set_parameter(self, p):
        p = Tensor(p, ms.float32)
        return ops.assign(self.p, p)

    def construct(self, x):
        return self.get_parameter * x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_property():
    """
    Feature: Support property decorator in graph mode.
    Description: Support property decorator in graph mode.
    Expectation: No exception.
    """

    b = PropertyNet(10)
    res1 = b(1)
    assert res1 == 10
    b.set_parameter(20)
    res2 = b(1)
    assert res2 == 20


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_property_in_graph_jit():
    """
    Feature: Support property decorator in graph mode.
    Description: Support property decorator in graph mode.
    Expectation: No exception.
    """
    @jit
    def test_property():
        b = PropertyNet(10)
        b.set_parameter(20)
        return b.get_parameter + 2

    out = test_property()
    assert out == 22


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_property_in_outer_net():
    """
    Feature: Support property decorator in graph mode.
    Description: Support property decorator in graph mode.
    Expectation: No exception.
    """
    class OuterNet(nn.Cell):
        def construct(self, x):
            b = PropertyNet(10)
            b.set_parameter(2)
            return b.get_parameter + x

    net = OuterNet()
    out = net(1)
    assert out == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_property_in_construct_init():
    """
    Feature: Support property decorator in graph mode.
    Description: Support property decorator in graph mode.
    Expectation: No exception.
    """
    class OuterNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.b = PropertyNet(10)

        def construct(self, x):
            self.b.set_parameter(2)
            return self.b.get_parameter + x

    net = OuterNet()
    out = net(1)
    assert out == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_property_outer():
    """
    Feature: Support property decorator in graph mode.
    Description: Support property decorator in graph mode.
    Expectation: No exception.
    """
    class OuterNet(nn.Cell):
        def construct(self, x):
            x.set_parameter(2)
            return x.get_parameter + 1

    x = PropertyNet(10)
    net = OuterNet()
    out = net(x)
    assert out == 3
