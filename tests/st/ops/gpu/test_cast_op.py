# Copyright 2019 Huawei Technologies Co., Ltd
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

import pytest
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
import mindspore.context as context
import numpy as np


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.Cast = P.Cast()

    def construct(self, x0, type0, x1, type1):
        output = (self.Cast(x0, type0),
                  self.Cast(x1, type1))
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float32))
    t0 = mstype.float16
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.float16))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net()
    output = net(x0, t0, x1, t1)
    type0 = output[0].asnumpy().dtype
    assert (type0 == 'float16')
    type1 = output[1].asnumpy().dtype
    assert (type1 == 'float32')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast1():
    x0 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.int32))
    t0 = mstype.float32
    x1 = Tensor(np.arange(24).reshape((4, 3, 2)).astype(np.bool))
    t1 = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net()
    output = net(x0, t0, x1, t1)
    type0 = output[0].asnumpy().dtype
    assert (type0 == 'float32')
    type1 = output[1].asnumpy().dtype
    assert (type1 == 'float32')
