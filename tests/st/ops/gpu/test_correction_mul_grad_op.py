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
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op_w = Q.CorrectionMulGrad()

    @jit
    def construct(self, dy, x, batch_std, running_std):
        dx, d_batch_std = self.op_w(dy, x, batch_std, running_std)
        return dx, d_batch_std


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_correction_mul_grad():
    net = Net()
    co, ci, h, w = 64, 1, 32, 32
    dout = np.random.uniform(-0.1, 0.1, size=[co, ci, h, w]).astype('float32')
    x = np.random.uniform(1, 1, size=[co, ci, h, w]).astype('float32')
    batch_std = np.random.uniform(1, 10, size=[co]).astype('float32')
    running_std = np.random.uniform(1, 10, size=[co]).astype('float32')
    output = net(Tensor(dout), Tensor(x), Tensor(batch_std), Tensor(running_std))
    expect = [0, 0]
    expect[0] = (dout * np.reshape(batch_std / running_std, (co, 1, 1, 1)))
    expect[1] = (np.sum(dout * x, (1, 2, 3)) / running_std)
    for i, _ in enumerate(output):
        assert np.allclose(output[i].asnumpy(), expect[i], rtol=1.e-5, atol=1.e-5)
