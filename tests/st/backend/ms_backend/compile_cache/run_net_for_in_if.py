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
test compile cache with control flow.
"""
import numpy as np
import mindspore as ms
from mindspore import jit, nn, Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C

grad_all = C.GradOperation(get_all=True)

class ForInIfNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()
        self.add = P.Add()
        param_a = np.full((1,), 5, dtype=np.float32)
        self.param_a = Parameter(Tensor(param_a), name='a')
        param_b = np.full((1,), 4, dtype=np.float32)
        self.param_b = Parameter(Tensor(param_b), name='b')

    @jit
    def construct(self, x):
        y = x + self.param_b
        if self.param_a > self.param_b:
            x = self.mul(x, 2)
            for i in range(-1, 5):
                x = self.add(i, x)
                self.param_b += 1
        elif y > x:
            y = self.param_a * y
        else:
            x = self.param_b * x
        return x, y

@ms.jit
class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        return grad_all(self.net)(*inputs)


if __name__ == "__main__":
    input_tensor = Tensor([10], mstype.float32)
    for_in_if_net = ForInIfNet()
    grad_net = GradNet(for_in_if_net)
    forward_net = ForInIfNet()
    forward_res = forward_net(input_tensor)
    backward_res = grad_net(input_tensor)

    assert forward_res == (Tensor([29], mstype.float32), Tensor([14], mstype.float32))
    assert backward_res == (Tensor([3], mstype.float32),)
    print("RUNTIME_COMPILE", forward_res[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", forward_res[0].asnumpy().shape, "RUNTIME_CACHE")
