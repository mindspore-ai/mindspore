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

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_hyper_param():
    """
    Feature: Resolve parameter.
    Description: The name of parameter in construct is the same with the name of parameter of class init.
    Expectation: self.a is different from a in construct.
    """
    class HyperParamNet(Cell):
        def __init__(self):
            super(HyperParamNet, self).__init__()
            self.a = Parameter(Tensor(1, ms.float32), name="a")
            self.b = Parameter(Tensor(5, ms.float32), name="param_b")
            self.c = Parameter(Tensor(9, ms.float32), name="param_c")

        def func_inner(self, c):
            return self.a + self.b + c

        def construct(self, a, b):
            self.a = a
            self.b = b
            return self.func_inner(self.c)

    x = Tensor(11, ms.float32)
    y = Tensor(19, ms.float32)
    net = HyperParamNet()
    output = net(x, y)
    output_expect = Tensor(39, ms.float32)
    assert output == output_expect


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_hyper_param_with_control_sink():
    """
    Feature: Resolve parameter.
    Description: Parameters whose name are the same between different graphs do not affect each other.
    Expectation: self.a is different from a in construct.
    """
    class HyperParamNet(Cell):
        def __init__(self):
            super(HyperParamNet, self).__init__()
            self.a = Parameter(Tensor(1, ms.float32), name="a")
            self.b = Parameter(Tensor(5, ms.float32), name="b")
            self.c = Parameter(Tensor(9, ms.float32), name="c")

        def func_inner(self, c):
            return self.a + self.b + c

        def func_inner_2(self, a, c):
            return a - self.b + c

        def construct(self, a, b):
            self.b = b
            if a > self.b:
                return self.func_inner_2(a, self.c)
            return self.func_inner(self.c)

    x = Tensor(11, ms.float32)
    y = Tensor(19, ms.float32)
    net = HyperParamNet()
    output = net(x, y)
    output_expect = Tensor(29, ms.float32)
    assert output == output_expect
