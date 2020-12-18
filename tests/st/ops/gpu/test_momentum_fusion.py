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
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore import Tensor
from mindspore.ops import operations as P

class MomentumFusionNet(nn.Cell):
    def __init__(self, var, accum):
        super(MomentumFusionNet, self).__init__()
        self.op = P.ApplyMomentum()
        self.add = P.AddN()
        self.mul = P.Mul()
        self.var = Parameter(var, name="variable")
        self.accum = Parameter(accum, name="accumulate")
        self.lr = 0.1
        self.weight_decay = 0.002
        self.moment = 0.98

    def construct(self, grad):
        wd = self.mul(self.var, self.weight_decay)
        g = self.add((wd, grad))
        return self.op(self.var, self.accum, self.lr, g, self.moment)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_momentum_fusion():
    np.random.seed(42)
    var = Tensor(np.random.randn(10, 20).astype(np.float32))
    accum = Tensor(np.random.randn(10, 20).astype(np.float32))
    grad = Tensor(np.random.randn(10, 20).astype(np.float32))

    context.set_context(device_target='GPU', mode=context.GRAPH_MODE)
    net1 = MomentumFusionNet(var, accum)
    _ = net1(grad)

    context.set_context(device_target='GPU', mode=context.PYNATIVE_MODE)
    net2 = MomentumFusionNet(var, accum)
    _ = net2(grad)

    assert np.allclose(net1.var.data.asnumpy(), net2.var.data.asnumpy(), atol=1e-5)
    assert np.allclose(net1.accum.data.asnumpy(), net2.accum.data.asnumpy(), atol=1e-5)
