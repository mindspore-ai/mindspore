# Copyright 2021 Huawei Technologies Co., Ltd
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
import sys
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.ops import operations as P


class NetWithControlFlow(nn.Cell):
    def __init__(self):
        super(NetWithControlFlow, self).__init__()
        self.mul = P.Mul()
        self.add = P.Add()
        param_a = np.full((1,), 5, dtype=np.float32)
        self.param_a = Parameter(Tensor(param_a), name='a')
        param_b = np.full((1,), 4, dtype=np.float32)
        self.param_b = Parameter(Tensor(param_b), name='b')

    def construct(self, x):
        if self.param_a > self.param_b:
            x = self.mul(x, 2)
            for _ in range(0, 5):
                x = self.add(x, x)
                self.param_b += 1
        return x


def run_net_with_control_flow():
    x = Tensor([10], mstype.int32)
    net = NetWithControlFlow()
    output = net(x)
    print("{", output, "}")
    print("{", output.asnumpy().shape, "}")


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, enable_compile_cache=True, compile_cache_path=sys.argv[1])
    run_net_with_control_flow()
    context.set_context(enable_compile_cache=False)
