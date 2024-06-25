# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
""" test high order control flow """
from tests.st.compiler.control.cases_register import case_register
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import context
import numpy as np

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
def test_high_control_while():
    """
    Feature: High-order differential function.
    Description: Infer of the high-order differential function.
    Expectation: Null.
    """

    class Net(Cell):
        def construct(self, x):
            while x < 10:
                x = (x * 2)
            return x

    net = Net()
    x = Tensor(1, dtype.float32)
    grad_net = F.grad(net)
    order_grad_net = F.grad(grad_net)
    order_grad = order_grad_net(x)
    assert order_grad == 0.0


@case_register.level0
@case_register.target_gpu
def test_high_control_for_while():
    """
    Feature: High-order differential function.
    Description: Infer of the complex high-order differential function.
    Expectation: Null.
    """

    class Net(Cell):
        def construct(self, x):
            for _ in [2]:
                for _ in [2]:
                    while x > 1:
                        x = (x / 3)
                        x = (x / 2)
                for _ in [2]:
                    x = (x / 1)
                    x = (x + 1)
            for _ in [3]:
                for _ in [4]:
                    x = (x / 1)
                    x = (x + 3)
                for _ in [5]:
                    x = (x / 3)
                    x = (x / 2)
            return x

    net = Net()
    x = Tensor(4, dtype.float32)
    grad_net = F.grad(net)
    grad_grad_net = F.grad(grad_net)
    result = grad_grad_net(x)
    assert result == 0.0


@case_register.level1
@case_register.target_gpu
def test_high_control_for_complex():
    """
    Feature: High-order differential function.
    Description: Infer of the complex high-order differential function.
    Expectation: Null.
    """

    class Net(Cell):

        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x, y):
            for _ in range(2):
                if x == y:
                    pass
                elif x == 5:
                    pass
                elif x <= 2:
                    while x <= 4:
                        if x <= 2:
                            break
                elif x <= 1:
                    pass
                elif y >= x:
                    pass
                elif x >= y:
                    pass
                elif x > y:
                    pass
                else:
                    pass
                while x < y:
                    if x == 0:
                        pass
                    elif x < 4:
                        pass
                    elif x < 0:
                        pass
                    else:
                        pass
                    if x >= y:
                        pass
                if y != x:
                    pass
                elif x >= 0:
                    pass
                else:
                    for _ in range(2):
                        if y >= x:
                            pass
                if x == 3:
                    continue
            return self.op(x, y)

    x = np.array([4], np.float32)
    y = np.array([4], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    sgrad_net = F.grad(grad_net)
    sgrad = sgrad_net(Tensor(x), Tensor(y))
    assert sgrad == 0.0


@case_register.level0
@case_register.target_gpu
def test_partial_graph_not_use():
    """
    Feature: High-order grad.
    Description: The graph of partial called by noting, but inferred by normalize. Now we let backend ignore the
    redundant graph, but we except front compiler make this partial graph as DeadNode.
    Expectation: No exception raised.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.w = ms.Parameter(Tensor([2], dtype.float32), name='weight')
            self.b = ms.Parameter(Tensor([0], dtype.float32), name='bias')

        def construct(self, x, y):
            if x > y:
                pass
            elif x == y:
                x = y / 2
            elif y != self.b:
                for _ in range(2):
                    y = y * 3
                    self.w = y * 3
                    x = self.w + y
                    if self.b <= y:
                        break
            return x + y

    x = np.array([4], np.float32)
    y = np.array([0], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    sgrad_net = F.grad(grad_net)
    sgrad = sgrad_net(Tensor(x), Tensor(y))
    print('second grad: ', sgrad)
