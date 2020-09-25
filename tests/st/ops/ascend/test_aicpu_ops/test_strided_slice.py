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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, begin, end, strides):
        super(Net, self).__init__()
        self.strided_slice = P.StridedSlice()
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, input):
        return self.strided_slice(input, self.begin, self.end, self.strides)


input_x = np.array([[[0, 1, 2], [3, 4, 5]],
                    [[6, 7, 8], [9, 10, 11]],
                    [[12, 13, 14], [15, 16, 17]]
                   ]).astype(np.float32)
begin = (1, 0, 0)
end = (2, 2, 3)
strides = (1, 1, 2)


def test_net():
    net = Net(begin, end, strides)
    tinput = Tensor(input_x)
    output = net(tinput)
    print(output.asnumpy())
    assert np.all([[[6, 8], [9, 11]]] == output.asnumpy())
