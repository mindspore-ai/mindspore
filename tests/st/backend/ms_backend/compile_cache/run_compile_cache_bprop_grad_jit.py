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
test compile cache with if bprop net
"""
from mindspore.common import Tensor
from mindspore import context, nn
from mindspore.ops.composite import GradOperation
from mindspore.common import dtype as mstype
import mindspore as ms

@ms.jit
class IfBpropNet(nn.Cell):
    """IfBpropNet
    Args:
        None
    Inputs:
        x (Tensor): Input tensor of integer type.
        y (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> net = IfBpropNet()
        >>> x = Tensor(5, mstype.int32)
        >>> y = Tensor(3, mstype.int32)
        >>> output = net(x, y)
    """
    def construct(self, x, y):
        x = x * 3
        if y > 2:
            z = 3 * y + x
        else:
            z = 2 * x + y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout, out, dout


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor(1, mstype.float32)
    b = Tensor(2, mstype.float32)
    grad_all = GradOperation(get_all=True)
    res = grad_all(IfBpropNet())(a, b)
    print("RUNTIME_COMPILE", res, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", res[0].asnumpy().shape, "RUNTIME_CACHE")
