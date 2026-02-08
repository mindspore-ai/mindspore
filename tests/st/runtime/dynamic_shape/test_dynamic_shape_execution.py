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

import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, jit
from mindspore import dtype as mstype


g_block_num = 2
steps = 5

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

    @jit(backend="ms_backend")
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


def test_dynamic_shape_execution_with_conf_thread_num():
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)

    net = Net()
    net.set_inputs(dyn_input_data)

    # warm up
    output = net(input_data)
    print(output)

    for _ in range(steps):
        output = net(input_data)
        output.asnumpy()

    exp_val = -0.06835
    exp_array = np.array([[exp_val, exp_val, exp_val], [exp_val, exp_val, exp_val]])
    assert np.allclose(output.asnumpy(), exp_array, 0.01, 0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_dynamic_shape_execution_with_conf_thread_num")
    parser.add_argument("--thread_num", type=int, default=5, help="thread number")
    args_opt = parser.parse_args()
    ms.runtime.dispatch_threads_num(args_opt.thread_num)
    test_dynamic_shape_execution_with_conf_thread_num()
