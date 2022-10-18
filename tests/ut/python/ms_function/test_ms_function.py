# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.nn import Momentum
from mindspore import context, Tensor
from mindspore.common.api import jit

grad_all = C.GradOperation(get_all=True)


class CellBprop(nn.Cell):
    def construct(self, x, y):
        return 2 * x * x + y * y

    @jit
    def bprop(self, x, y, out, dout):
        return dout, 2 * y


def test_cell_bprop_grad():
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    input_y = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net = CellBprop()
    with pytest.raises(RuntimeError):
        grad_all(net)(input_x, input_y)


class ConvNet(nn.Cell):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")

    def construct(self, x):
        out = self.conv(x)
        return out


class MomentumWithMsFunc(nn.Cell):
    def __init__(self, net):
        super(MomentumWithMsFunc, self).__init__()
        self.net = net
        self.optimizer = Momentum(filter(lambda x: x.requires_grad, self.net.get_parameters()), 0.1, 0.9)

    @jit
    def construct(self, grads):
        ret = self.optimizer(grads)
        return ret


def test_ms_func_decorate_forward():
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.random.randn(1, 1, 2, 2).astype(np.float32))
    net = ConvNet()
    grad_out = grad_all(net)(input_x)
    opt = MomentumWithMsFunc(net)
    opt(grad_out)
