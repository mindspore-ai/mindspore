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

"""LeNet test."""

import numpy as np

from lenet import LeNet5
import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor

context.set_context(mode=context.GRAPH_MODE)


grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)

batch_size = 1
channel = 1
height = 32
weight = 32
num_class = 10


class LeNetGrad(nn.Cell):
    """Backward of LeNet"""

    def __init__(self, network):
        super(LeNetGrad, self).__init__()
        self.grad_op = grad_all_with_sens
        self.network = network

    def construct(self, x, sens):
        grad_op = self.grad_op(self.network)(x, sens)

        return grad_op


def test_compile():
    """Compile forward graph"""
    net = LeNet(num_class=num_class)
    np.random.seed(7)
    inp = Tensor(np.array(np.random.randn(batch_size,
                                          channel,
                                          height,
                                          weight) * 3, np.float32))

    _cell_graph_executor.compile(net, inp)


def test_compile_grad():
    """Compile forward and backward graph"""
    net = LeNet5(num_class=num_class)
    inp = Tensor(np.array(np.random.randn(batch_size,
                                          channel,
                                          height,
                                          weight) * 3, np.float32))
    sens = Tensor(np.ones([batch_size, num_class]).astype(np.float32))
    grad_op = LeNetGrad(net)

    _cell_graph_executor.compile(grad_op, inp, sens)
