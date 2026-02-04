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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.runtime as rt
from mindspore import Tensor, context, jit
from mindspore.common import Parameter
from mindspore import dtype as mstype

context.set_context(
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    },
    max_call_depth=600000
)

g_block_num = 20
steps = 20
input_len = 10

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x):
        x = self.add(x, self.param)
        for _ in range(5):
            x = self.add(x, 0.1)
            x = self.add(x, 0.2)
        x = self.mul(x, 2)
        x = self.add(x, 0.5)
        return x

class SeqNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = Net()

    @jit
    def construct(self, x):
        output = self.net(x)
        return output

def run_single_graph_save():
    rt.set_kernel_launch_capture(True)
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    base_shape = (2, 3)

    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    for i in range(1, 20):
        current_param = net.net.param.data.asnumpy()
        new_param = current_param + 0.1 * (i + 1)
        net.net.param.set_data(Tensor(new_param, mstype.float32))
        input_data1 = Tensor(np.full(base_shape, i).astype(np.float32))
        net(input_data1)

if __name__ == "__main__":
    run_single_graph_save()
