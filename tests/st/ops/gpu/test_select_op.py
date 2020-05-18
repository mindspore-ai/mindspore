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

import pytest
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.select = P.Select()

    def construct(self, cond, x, y):
        return self.select(cond, x, y)


cond = np.array([[True, False], [True, False]]).astype(np.bool)
x = np.array([[1.2, 1], [1, 0]]).astype(np.float32)
y = np.array([[1, 2], [3, 4.0]]).astype(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_select():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    select = Net()
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    expect = [[1.2, 2], [1, 4.0]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)
