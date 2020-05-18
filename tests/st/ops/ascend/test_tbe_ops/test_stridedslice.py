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
import pytest

import mindspore.context as context
import mindspore.ops.operations as P
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.train.model import Model

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(Cell):
    def __init__(self, begin, end, stride):
        super(Net, self).__init__()
        self.stridedslice = P.StridedSlice()
        self.begin = begin
        self.end = end
        self.stride = stride

    def construct(self, input):
        x = self.stridedslice(input, self.begin, self.end, self.stride)
        return x


def me_stridedslice(input1, begin, end, stride):
    input_me = Tensor(input1)
    net = Net(begin, end, stride)
    net.set_train()
    model = Model(net)
    output = model.predict(input_me)
    print(output.asnumpy())


def test_stridedslice_input_2d():
    input = np.random.randn(5, 5).astype(np.int32)
    begin = (0, 0)
    end = (2, 2)
    stride = (1, 1)

    me_stridedslice(input, begin, end, stride)


def test_stridedslice_input_3d():
    input = np.random.randn(5, 5, 5).astype(np.float32)
    begin = (0, 0, 0)
    end = (3, 3, 3)
    stride = (1, 1, 1)
    me_stridedslice(input, begin, end, stride)
