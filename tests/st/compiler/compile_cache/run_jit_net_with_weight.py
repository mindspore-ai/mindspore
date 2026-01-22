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

from mindspore import Tensor, context, nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell
from mindspore.graph.api import jit
from tests.st.pynative.utils import GradOfAllParams


class Net0(Cell):
    def __init__(self, para):
        super().__init__()
        self.para = Parameter(para, name="weight")
        self.relu = nn.ReLU()
        self.square = P.Square()

    @jit(backend="ms_backend")
    def construct(self):
        return self.square(self.relu(self.para))

inputs = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
context.set_context(mode=context.PYNATIVE_MODE)
net0 = Net0(inputs)
out_ms_fun = net0()
grad_net0 = GradOfAllParams(net0)
grad_net0.set_train()
input_grad0 = grad_net0(out_ms_fun)
print("AAA", input_grad0[0], "BBB")
print("AAA", input_grad0[0].asnumpy().shape, "BBB")
