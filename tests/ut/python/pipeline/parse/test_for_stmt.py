# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_for_stmt """
from dataclasses import dataclass
import numpy as np

from mindspore import Tensor, Model, context
from mindspore.nn import Cell
from mindspore.nn import ReLU
from ...ut_filter import non_graph_engine


@dataclass
class Access:
    a: int
    b: int

    def max(self):
        if self.a > self.b:
            return self.a
        return self.b


class access2_net(Cell):
    """ access2_net definition """

    def __init__(self, number, loop_count=1):
        super().__init__()
        self.number = number
        self.loop_count = loop_count
        self.relu = ReLU()

    def construct(self, x):
        a = self.loop_count
        b = self.number
        z = Access(a, b)
        for _ in (a, z):
            x = self.relu(x)
        return x


def function_access_base(number):
    """ function_access_base """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    if number == 2:
        net = access2_net(number)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    model.predict(input_me)


@non_graph_engine
def test_access_0040():
    """ test_access_0040 """
    function_access_base(2)
