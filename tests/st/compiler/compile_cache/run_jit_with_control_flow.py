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

from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.graph.api import jit
from tests.st.pynative.utils import GradOfAllInputs


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()
        self.add = P.TensorAdd()

    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        out = z
        for _ in range(5):
            if 2 * x < y:
                if 3 * x < y:
                    out = self.add(out, out)
                    x = x + 1
                out = self.relu(out)
            if x + 6 == y:
                break
        out = self.relu(out)
        return out


input_x = Tensor(2.0)
input_y = Tensor(10.0)
input_z = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
context.set_context(mode=context.PYNATIVE_MODE)
net0 = Net()
out_ms_fun = net0(input_x, input_y, input_z)
grad_net0 = GradOfAllInputs(net0)
grad_net0.set_train()
input_grad0 = grad_net0(input_x, input_y, input_z, out_ms_fun)
print("AAA", input_grad0, "BBB")
print("AAA", input_grad0[2].asnumpy().shape, "BBB")
