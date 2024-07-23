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
from mindspore import context
import mindspore.nn as nn
from mindspore.nn import ReLU
from mindspore.nn import Cell
from mindspore.nn import Momentum
from mindspore.common.tensor import Tensor
from mindspore.common.api import jit
from mindspore.ops import composite as C
from tests.mark_utils import arg_mark


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_staging_together():
    """
    Feature: Pynative stage
    Description: Stage call
    Expectation: No exception.
    """
    class NetPynative(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()
        def construct(self, x):
            return self.relu(x)

    class NetStaging(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()
        @jit
        def construct(self, x):
            return self.relu(x)

    input1 = np.random.randn(2, 2).astype(np.float32)

    net1 = NetPynative()
    out_me_pynative = net1(Tensor(input1)).asnumpy()

    net2 = NetStaging()
    out_me_staging = net2(Tensor(input1)).asnumpy()

    assert np.allclose(out_me_pynative, out_me_staging, 0.001, 0.001)


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


grad_by_list = C.GradOperation(get_by_list=True)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_ms_func_decorate_forward():
    """
    Feature: Auto diff for jit.
    Description: Check the result for GradOperation.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.random.randn(1, 1, 2, 2).astype(np.float32))
    net = ConvNet()
    grad_out = grad_by_list(net, net.trainable_params())(input_x)
    opt = MomentumWithMsFunc(net)
    opt(grad_out)
