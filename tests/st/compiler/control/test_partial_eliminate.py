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
""" test_partial_eliminate """
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype, Parameter
import mindspore.ops.functional as F
import numpy as np
from tests.st.compiler.control.cases_register import case_register


class Net(Cell):

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([(- 1)], dtype.float32), name='w')
        self.b = Parameter(Tensor([(- 1)], dtype.float32), name='b')

    def construct(self, x, y):
        for s in range(2):
            if s > y:
                self.b = x + y
            elif self.b <= 2:
                return self.w
            else:
                return y
        return x + y


@case_register.level0
@case_register.target_gpu
def test_switch_partial_eliminate():
    """
    Feature: control flow
    Description: Eliminate partial node of pass: SwitchPartialEliminater
    Expectation: No exception.
    """
    x = np.array([3], np.float32)
    y = np.array([1], np.float32)
    net1 = Net()
    grad_net = F.grad(net1, grad_position=(0, 1))
    grad_net(Tensor(x), Tensor(y))
