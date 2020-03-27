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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context

class AssignAdd(nn.Cell):
    def __init__( self):
        super(AssignAdd, self).__init__()
        self.add = P.AssignAdd()

    def construct(self, x, y):
        res = self.add(x, y)
        return res

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_assign_add():
    expect1 = np.array([[[[ 0,  2,  4.],
                         [ 6,  8, 10.],
                         [12, 14, 16.]],
                        [[18, 20, 22.],
                         [24, 26, 28.],
                         [30, 32, 34.]],
                        [[36, 38, 40.],
                         [42, 44, 46.],
                         [48, 50, 52.]]]])
    expect2 = np.array([[[[ 0, 3, 6],
                          [ 9, 12, 15],
                          [18, 21, 24]],
                         [[27, 30, 33],
                          [36, 39, 42],
                          [45, 48, 51]],
                         [[54, 57, 60],
                          [63, 66, 69],
                          [72, 75, 78]]]])
    x = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    add = AssignAdd()
    output1 = add(x, y)
    assert (output1.asnumpy() == expect1).all()
    output2 = add(output1, y)
    assert (output2.asnumpy() == expect2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    add = AssignAdd()
    output1 = add(x, y)
    assert (output1.asnumpy() == expect1).all()
    output2 = add(output1, y)
    assert (output2.asnumpy() == expect2).all()
