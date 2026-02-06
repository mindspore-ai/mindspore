# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore import context
from mindspore import Tensor, nn, jit
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
import numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

@jit(backend="ms_backend")
def grad_func(net, x, y, z):
    grad = mindspore.grad(net)
    grad(x, y, z)


def test_ge_nested_tuple():
    """
    Description: Test GE nested tuple.
    Description: Support nested tuple in GE.
    Expectation: Run without errors.
    """
    context.set_context(jit_config={"jit_level": "O0"})

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    if 3 * x < y:
                        out = self.add(out, out)
                        x += 1
                    out = self.relu(out)
                if x + 6 == y:
                    break
            out = self.relu(out)
            return out

    net = Net()
    x = Tensor(2, mindspore.float32)
    y = Tensor(10, mindspore.float32)
    z = Tensor(np.random.rand(4, 4, 4), dtype=mstype.float32)
    net(x, y, z)
    grad_func(net, x, y, z)
