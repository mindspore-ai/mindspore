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
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, jit
from mindspore import dtype as mstype
from mindspore.communication.management import init


steps = 3


class Net(nn.Cell):
    """
    Construct a single-input network structure including AllReduce, AllGather.
    """
    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.reshape = P.Reshape()
        self.all_reduce = P.AllReduce()
        self.all_gather = P.AllGather()

    @jit(backend="ms_backend")
    def construct(self, x):
        x = self.reshape(x, (1, -1))
        x = self.add(x, 1)
        x = self.all_reduce(x)
        x = self.sub(x, 1.1)
        x = self.reshape(x, (3, -1))
        x = self.mul(x, 0.251)
        x = self.all_gather(x)

        x = self.mul(x, 0.501)
        x = self.sub(x, 1.1)
        x = self.all_reduce(x)
        x = self.reshape(x, (2, -1))
        x = self.all_reduce(x)
        x = self.sub(x, 1.1)
        x = self.reshape(x, (2, -1))
        return x


def online_execution_order_check():
    """
    Run network including AllReduce, AllGather with execution order check.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32)).pin_memory()
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)

    net = Net()
    net.set_inputs(dyn_input_data)

    # warm up
    output = net(input_data)
    print(output)

    for _ in range(steps):
        output = net(input_data)
        output.asnumpy()

    exp_val = -12.865154
    assert np.all(output.asnumpy() == exp_val)


if __name__ == "__main__":
    try:
        os.environ["MS_DEV_RUNTIME_CONF"] = "execution_order_check_iteration:0"
        init()
        online_execution_order_check()
    finally:
        os.unsetenv("MS_DEV_RUNTIME_CONF")
