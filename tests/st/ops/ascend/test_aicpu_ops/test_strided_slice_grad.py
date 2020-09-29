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
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, shape_x, begin, end, strides):
        super(Net, self).__init__()
        self.strided_slice_grad = G.StridedSliceGrad()
        self.shape_x = shape_x
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, dy):
        return self.strided_slice_grad(dy, self.shape_x, self.begin, self.end, self.strides)


dy = np.array([[[6, 8], [9, 11]]]).astype(np.float32)
shape_x = (3, 2, 3)
begin = (1, 0, 0)
end = (2, 2, 3)
strides = (1, 1, 2)


def test_net():
    net = Net(shape_x, begin, end, strides)
    tdy = Tensor(dy)
    output = net(tdy)
    print(output.asnumpy())
    assert np.all([[[0, 0, 0], [0, 0, 0]],
                   [[6, 0, 8], [9, 0, 11]],
                   [[0, 0, 0], [0, 0, 0]]
                   ] == output.asnumpy())
