# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P


class NetFloatStatusAddN(Cell):
    def __init__(self):
        super(NetFloatStatusAddN, self).__init__()
        self.status = P.FloatStatus()
        self.addn = P.AddN()
        self.square = P.Square()

    def construct(self, x, y, z):
        res0 = self.square(x)
        res1 = self.status(res0)
        res2 = self.status(y)
        res3 = self.status(z)
        res4 = self.addn((res1, res2, res3))
        return self.square(res4)


def run_floatstatus_addn():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_z = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    net = NetFloatStatusAddN()
    result = net(Tensor(input_x), Tensor(input_y), Tensor(input_z))
    res = np.allclose(0, result.asnumpy(), rtol=1.e-4, atol=1.e-7)
    assert res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_floatstatus_addn():
    """
    Feature: graph kernel testcase for floatstatus addn fusion
    Description: random input when using graph_kernel in graph mode
    Expectation: the result is 0
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    run_floatstatus_addn()
