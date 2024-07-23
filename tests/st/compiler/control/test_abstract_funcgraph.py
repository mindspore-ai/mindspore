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
from mindspore.common import Tensor, dtype
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import numpy as np
from tests.st.compiler.control.cases_register import case_register


@case_register.level0
@case_register.target_gpu
def test_watch_get_func_graphs_from_abstract():
    """
    Feature: Get func_graph from abstract.
    Description: Watching the function of getting func graph from abstract.
    Expectation: Output correct.
    """

    class Net(Cell):

        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x, y):
            for t in range(2):
                if y != x:
                    if x > 4:
                        x = y / x
                        y = 1 - x
                        y = y - y
                    elif x > 2:
                        y = x - 1
                    else:
                        y = 3 - y
                    y = t * x
                elif x != 3:
                    x = x - x
                if x == y:
                    continue
            return self.op(x, y)

    x = np.array([4], np.float32)
    y = np.array([1], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(Tensor(x), Tensor(y))
    assert fgrad[0] == Tensor([2], dtype.float32)
    assert fgrad[1] == Tensor([0], dtype.float32)
