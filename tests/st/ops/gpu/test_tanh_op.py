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
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class TanhNet(nn.Cell):
    def __init__(self):
        super(TanhNet, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return self.tanh(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(name="get_all", get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_Tanh():
    x_np = np.array(
        [[0.28522366, 0.38033979, 1.54657853, -0.98530175, -0.54365635, 0.12652203, -1.33449938, -0.27737698],
         [2.06282293, 0.84635078, 0.16628414, -0.91823183, -0.72023044, -0.09147043, -0.04166984, -1.5664763],
         [-0.17157249, 0.44260951, -0.6683391, 1.13142613, 1.5536937, -0.32799768, -0.20016545, 0.06773927]],
        dtype=np.float32)
    dy_np = np.array(
        [[0.44969849, -0.187879, -0.64300827, 1.36638774, 0.89930276, -0.23835229, -0.67771854, -1.88984999],
         [2.00418801, 2.33336475, 0.00241747, 1.31558685, 0.06768817, -2.23008804, -0.26818366, -1.26873401],
         [1.83694105, 0.5339005, 0.51117424, 0.49202378, -0.83297819, -0.71001219, 0.18913512, 0.65580389]],
        dtype=np.float32)

    x_ms = Tensor(x_np)
    dy_ms = Tensor(dy_np)

    net = TanhNet()
    grad = Grad(net)
    output = grad(x_ms, dy_ms)

    expect = [[0.41501077, -0.16312202, -0.10675912, 0.58678646, 0.67828224, -0.23457714, -0.1643468, -1.75159405],
              [0.12541081, 1.2251587, 0.00235184, 0.62396731, 0.04191568, -2.21153283, -0.26771853, -0.20311764],
              [1.78391056, 0.44159236, 0.33690308, 0.16800483, -0.13651318, -0.63878956, 0.18175511, 0.65280384]]

    assert np.allclose(output[0].asnumpy(), expect)
