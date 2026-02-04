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

from mindspore.common import Tensor
from mindspore import context, jit, nn
from mindspore.ops.composite import GradOperation
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y):
        z = x * y
        z = z * y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout, out, dout


context.set_context(mode=context.PYNATIVE_MODE)
grad_all = GradOperation(get_all=True)
res = grad_all(Net())(Tensor(1, mstype.float32), Tensor(2, mstype.float32))
print("AAA", res, "BBB")
print("AAA", res[0].asnumpy().shape, "BBB")
