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
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, mint, context, Tensor, Parameter
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_parameter_ref_key():
    """
    Feature: Test parameter.
    Description: Parameters with different ref_key.
    Expectation: No exception.
    """
    def Normalize(in_channels, num_groups=32):
        return mint.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels).to_float(ms.float32)

    class Block(nn.Cell):
        def __init__(self, in_channels=512, out_channels=512):
            super().__init__()
            self.norm1 = Normalize(in_channels)
            self.norm2 = Normalize(out_channels)
            self.norm1.weight = ms.Parameter(ms.ops.ones_like(self.norm1.weight))
            self.norm2.weight = ms.Parameter(ms.ops.ones_like(self.norm2.weight) * 2)

        def construct(self, h):
            w1 = self.norm1.weight.value()
            w2 = self.norm2.weight.value()
            h1 = self.norm1(h)
            h2 = self.norm2(h1)
            return h1, h2, w1, w2

    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.random.randn(1, 512, 32, 32), ms.float32)
    net = Block()
    [_, _, w1, w2] = net(x)
    assert np.allclose(w1 * 2, w2)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_cell_parameter_as_input1():
    """
    Feature: Test parameter.
    Description: Test cell parameter as input
    Expectation: No exception.
    """
    class ParamNet(nn.Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param = Parameter(Tensor(2, ms.float32), name="myname")

        @ms.jit
        def func(self, same_param):
            return same_param * self.param

        def construct(self, x):
            return self.func(self.param) * x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ParamNet()
    x = ms.Tensor(3, ms.float32)
    net(x)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_cell_parameter_as_input2():
    """
    Feature: Test parameter.
    Description: Test cell parameters as input
    Expectation: No exception.
    """
    class ParamNet(nn.Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param1 = Parameter(Tensor(2, ms.float32), name="myname1")
            self.param2 = Parameter(Tensor(3, ms.float32), name="myname2")

        @ms.jit
        def func(self, same_param1, same_param2):
            return same_param1 * same_param2 * self.param1 * self.param2

        def construct(self, x):
            return self.func(self.param1, self.param2) * x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ParamNet()
    x = ms.Tensor(3, ms.float32)
    net(x)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_cell_parameter_as_input3():
    """
    Feature: Test parameter.
    Description: Test cell parameter and another parameter with same name as inputs
    Expectation: Get the expected exception report.
    """
    class ParamNet(nn.Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param = Parameter(Tensor(2, ms.float32), name="myname")

        @ms.jit
        def func(self, same_param, same_name_param):
            return self.param * same_param * same_name_param

        def construct(self, same_name_param):
            return self.func(self.param, same_name_param)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ParamNet()
    same_name_param = ms.Parameter(ms.Tensor(1), name="myname")
    with pytest.raises(ValueError, match="its name 'myname' already exists."):
        net(same_name_param)
