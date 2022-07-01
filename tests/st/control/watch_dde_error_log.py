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
from mindspore.common import Tensor
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import numpy as np


def test_switch_simplify_avoid_dead_node():
    """
    Feature: Switch simplify pass.
    Description: If switch simplify pass can't simplify constant tensor condition,
                 dead node will exist in backend.
    Expectation: output correct.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x, y):
            if y != x:
                x = y - 3
            elif x == 4:
                for r in range(2):
                    x = 1 / y
                    if x > 2:
                        y = y + 3
                        y = y - y
                        y = y * x
                    elif y >= x:
                        x = x * x
                    elif x > y:
                        x = y - r
                    else:
                        y = 2 + x
                    for _ in range(2):
                        x = x * y
                        x = x - 3
                        y = y + 2
                        if x > 3:
                            break
                    if x > 2:
                        break
            elif x == y:
                if y <= x:
                    y = x / 2
                    x = 3 + y
                    x = x * 2
                elif x == 2:
                    x = y * y
                elif x < y:
                    y = 2 * y
                elif x != 2:
                    y = x * y
            while x != 5:
                break
            return self.op(x, y)

    x = np.array([4], np.float32)
    y = np.array([4], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(Tensor(x), Tensor(y))
    sgrad_net = F.grad(grad_net)
    sgrad = sgrad_net(Tensor(x), Tensor(y))
    assert np.allclose(out.asnumpy(), np.array([-19.75], np.float32))
    assert np.allclose(fgrad[0].asnumpy(), np.array([0.], np.float32))
    assert np.allclose(fgrad[1].asnumpy(), np.array([-2.03125], np.float32))
    assert np.allclose(sgrad.asnumpy(), np.array([0.], np.float32))


if __name__ == "__main__":
    test_switch_simplify_avoid_dead_node()
