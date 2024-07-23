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

from mindspore.nn import Cell
from mindspore.common import Tensor, dtype, Parameter
import mindspore.ops.functional as F
import numpy as np
from tests.st.compiler.control.cases_register import case_register


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([2], dtype.float32), name='weight')
        self.b = Parameter(Tensor([(- 2)], dtype.float32), name='bias')

    def construct(self, x, y):
        if self.w != 3:
            self.b = 3 + x
            if self.w == self.b:
                self.w = self.b + y
                return self.b
            if x <= self.w:
                return x
        elif self.b >= y:
            for _ in range(2):
                x = 2 + y
                if x != 3:
                    pass
        elif y <= 3:
            while x == self.w:
                return y
        for f in range(2):
            if x != self.w:
                x = 3 + x
                x = f / y
                self.w = y + x
            elif self.w >= 1:
                return x
            elif self.w > y:
                return self.w
            elif self.w != 5:
                return self.b
            else:
                return self.w
            if y == self.b:
                continue
        return x + y


@case_register.level0
@case_register.target_gpu
def test_watch_core_dump():
    """
    Feature: Control flow.
    Description: This complex control flow testcase has been cause core dump before, so add this case to watch graph
        compiling.
    Expectation: No core dump happened.
    """
    x = np.array([4], np.float32)
    y = np.array([5], np.float32)
    net1 = Net()
    grad_net = F.grad(net1, grad_position=(0, 1))
    grad_net(Tensor(x), Tensor(y))
