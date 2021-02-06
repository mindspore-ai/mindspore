# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class GeluNet(nn.Cell):
    def __init__(self):
        super(GeluNet, self).__init__()
        self.gelu = P.GeLU()

    def construct(self, x):
        return self.gelu(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gelugrad():
    x_ms = Tensor(np.array([0.58401114, 0.68800163, 0.9760397, 0.14702141, 0.46563736, 0.9607501,
                            0.14567593, 0.12261796, 0.37054458, 0.46421242]).astype(np.float32))
    dy_ms = Tensor(np.array([0.5559598, 0.96994054, 0.24770357, 0.34646875, 0.2984393, 0.03287048,
                             0.55681044, 0.966908, 0.06015943, 0.6099489]).astype(np.float32))

    net = GeluNet()
    grad = Grad(net)

    output = grad(x_ms, dy_ms)
    expect = [0.50963277, 0.9414753, 0.2667653, 0.21358444, 0.25243032, 0.0352667,
              0.34266686, 0.57757664, 0.04707306, 0.51536125]
    assert np.allclose(output[0].asnumpy(), expect)
