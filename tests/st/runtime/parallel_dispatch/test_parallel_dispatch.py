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

import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
from tests.mark_utils import arg_mark


g_block_num = 50
steps = 50

ascend_home_path = os.getenv('ASCEND_HOME_PATH')
if not ascend_home_path:
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"}, max_call_depth=6000)

class Net(nn.Cell):
    """
    Construct a single-input network structure.
    """
    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.reshape(x, (1, -1))

        for _ in range(g_block_num):
            x = self.add(x, 1)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (3, -1))
            x = self.mul(x, 0.251)
            x = self.add(x, 1)

            x = self.mul(x, 0.501)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.mul(x, 2)
            x = self.add(x, 1)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (6, -1))
            x = self.mul(x, 0.051)

        x = self.reshape(x, (2, -1))
        return x


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_host_bound_for_parallel_dispatch():
    """
    Feature: Graph parallel dispatch kernel.
    Description: Test host bound case for parallel dispatch kernels, which includes aclnn and inference
                 internal kernels.
    Expectation: The program execute and exit normally.
    """
    ms.runtime.set_kernel_launch_group()
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32)).pin_memory()
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)

    net = Net()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    for _ in range(steps):
        output = net(input_data)
        output.asnumpy()

    exp_val = -0.06835
    exp_array = np.array([[exp_val, exp_val, exp_val], [exp_val, exp_val, exp_val]])
    assert np.allclose(output.asnumpy(), exp_array, 0.0001, 0.0001)
