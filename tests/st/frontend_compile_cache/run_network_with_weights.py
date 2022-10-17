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


class NetWithWeights(nn.Cell):
    def __init__(self):
        super(NetWithWeights, self).__init__()
        self.matmul = P.MatMul()
        self.a = Parameter(Tensor(np.array([2.0], np.float32)), name='a')
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        y = y * self.a
        out = self.matmul(x, y)
        return out


def run_simple_net():
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
    net = NetWithWeights()
    output = net(x, y)
    print("{", output, "}")
    print("{", output.asnumpy().shape, "}")


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, enable_compile_cache=True, compile_cache_path=sys.argv[1])
    run_simple_net()
    context.set_context(enable_compile_cache=False)
