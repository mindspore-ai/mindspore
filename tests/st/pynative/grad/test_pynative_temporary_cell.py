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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as P
from mindspore.nn.optim import Momentum
from mindspore.common import ParameterTuple
from tests.mark_utils import arg_mark


class GradofParams(nn.Cell):
    def __init__(self, net, sens=False):
        super().__init__()
        self.grad = P.GradOperation(get_all=False, get_by_list=True, sens_param=sens)
        self.net = net
        self.params = ParameterTuple(self.net.trainable_params())

    def construct(self, *x):
        out = self.grad(self.net, self.params)(*x)
        return out


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_temporary_cell_variables():
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.add(x, x)
            return x

    class TempCellNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')

        def construct(self, x):
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = self.add(x, x)
            return x

    input_data = Tensor(np.random.randn(1, 1, 224, 224).astype(np.float32))
    # The first net run
    net = Net()
    backnet = GradofParams(net)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    grad_first = backnet(input_data)
    optimizer(grad_first)
    grad_second = backnet(input_data)
    # The second net run
    compare_net = TempCellNet()
    compare_backnet = GradofParams(compare_net)
    compare_optimizer = Momentum(filter(lambda x: x.requires_grad, compare_net.get_parameters()), 0.1, 0.9)
    compare_grad_first = compare_backnet(input_data)
    compare_optimizer(compare_grad_first)
    compare_grad_second = compare_backnet(input_data)
    # compare result
    assert np.allclose(grad_first[0].asnumpy(), compare_grad_first[0].asnumpy(), 0.01, 0.01)
    assert np.allclose(grad_second[0].asnumpy(), compare_grad_second[0].asnumpy(), 0.01, 0.01)
