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
from tests.mark_utils import arg_mark

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
        self.op = Q.CorrectionMul()

    @jit
    def construct(self, x, batch_var, moving_var):
        return self.op(x, batch_var, moving_var)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_correction_mul():
    net = Net()
    co = 64
    x = np.random.uniform(-1, 1, size=[co, 64, 32, 32]).astype('float32')
    bv = np.random.uniform(1, 2, size=[co]).astype('float32')
    mv = np.random.uniform(1, 2, size=[co]).astype('float32')
    output = net(Tensor(x), Tensor(bv), Tensor(mv))
    expect = x * np.reshape(bv, (co, 1, 1, 1)) / np.reshape(mv, (co, 1, 1, 1))
    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(diff > error * -1)
    assert output.shape == expect.shape
