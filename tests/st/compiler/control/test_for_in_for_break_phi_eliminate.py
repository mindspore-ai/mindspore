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
from tests.st.compiler.control.cases_register import case_register

from mindspore.nn import Cell
from mindspore.common import Tensor, dtype
import mindspore.ops.operations as P
from mindspore import Parameter
import numpy as np


class IfInFor(Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(np.ones((1,), dtype=np.int32)), name="w1")
        self.shape = P.Shape()

    def construct(self, x, y):
        shape = self.shape(y)
        for _ in range(1):
            if shape[2] % 2 == 0:
                for m in range(0):
                    m -= 1
                    if m > 10:
                        m /= 5
                    x = x + m * self.param
                    if m < 0:
                        break
        return x


@case_register.level1
@case_register.target_ascend
@case_register.target_gpu
def test_for_in_for_break_phi_node_eliminate():
    """
    Feature: Phi node eliminate.
    Description: For loop created some redundant  phi node, such as phi_range, which will cause some
        problems in infer process.
    Expectation: Compiling success.
    """

    x = Tensor([2])
    y = Tensor(np.ones((2, 2, 2)), dtype.int32)
    net = IfInFor()
    out = net(x, y)
    assert out == Tensor([2])
