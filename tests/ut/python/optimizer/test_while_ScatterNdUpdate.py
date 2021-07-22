# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import context, nn, Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self, data):
        super(Net, self).__init__()
        self.start = Tensor(0, dtype=mstype.int32)
        self.end = Tensor(2, dtype=mstype.int32)
        self.max_output = Parameter(data, "output_x")
        self.upd = P.ScatterNdUpdate()
        self.zero = Tensor(np.ones([1], dtype=np.int32))

    def construct(self, inputs):
        idx = self.start
        end = self.end
        while idx < end:
            xi = inputs[idx, :, :]
            self.upd(self.max_output, idx + self.zero, xi)
            idx = idx + 1
        return self.max_output + 0


def test_x():
    x = Tensor(np.arange(10 * 2 * 3).reshape(10, 2, 3).astype(np.float32))
    net = Net(x)
    net(x)
