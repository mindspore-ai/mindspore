# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
test compile cache when two branch must return the same shape.
"""
import numpy as np
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
import mindspore.ops.functional as F


@jit(backend="ms_backend")
def grad_func(input_net, grad_input_x, grad_input_y):
    return F.grad(input_net, grad_position=(0, 1))(grad_input_x, grad_input_y)


class IfByIfNet(Cell):
    """IfByIfNet
    Args:
        None

    Inputs:
        x (Tensor): Scalar tensor of integer type, controls branch selection.
        y (Tensor): Scalar tensor of numeric type, participates in arithmetic operations.

    Returns:
        Tensor, element-wise sum of updated `y` and original `x`.

    Examples:
        >>> net = IfByIfNet()
        >>> x = Tensor(6, mstype.int32)
        >>> y = Tensor(2, mstype.int32)
        >>> output = net(x, y)
    """
    def __init__(self):
        super().__init__()
        self.a = 1

    def construct(self, x, y):
        for k in range(1):
            if x != 1:
                for _ in range(1):
                    y = k * x
                    y = self.a + y
                    if x > 5:
                        break
            if x == 5:
                for _ in range(1):
                    y = self.a - y
                    if x == y:
                        continue
        return x + y


if __name__ == "__main__":
    context.set_context(jit_config={"jit_level": "O1"})
    input_x = np.array([-1], np.float32)
    input_y = np.array([2], np.float32)
    net = IfByIfNet()
    forward_res = net(Tensor(input_x), Tensor(input_y))
    backend_res = grad_func(net, Tensor(input_x), Tensor(input_y))
    print("RUNTIME_COMPILE", forward_res[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", forward_res[0].asnumpy().shape, "RUNTIME_CACHE")
