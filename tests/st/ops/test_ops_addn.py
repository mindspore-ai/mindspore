# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import pytest
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y, z):
        return self.addn((x, y, z))

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_addn(mode):
    """
    Feature: Test functional AddN operator in all Backend.
    Description: Operator AddN in CPU/GPU/ASCEND.
    Expectation: Assert result compare with tensorflow.
    """
    x = Tensor([1.57, 2.64, 9.34], ms.float32)
    y = Tensor([-0.29, 3.73, 8.36], ms.float32)
    z = Tensor([-0.59, 4.56, 6.79], ms.float32)
    addn = Net()
    output = addn(x, y, z)
    expect_result = np.array([0.69, 10.93, 24.49]).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), expect_result, rtol=0.004, atol=0.004)
