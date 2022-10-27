# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5).add_flags_recursive(fp32=True)
    return bn


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

    # model_parallel test


def test_auto_parallel_bn_with_prelu():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.bn = bn_with_initialize(16)
            self.prelu = nn.PReLU(16)

        def construct(self, x):
            out = self.bn(x)
            out = self.prelu(out)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")

    x = Tensor(np.random.rand(16, 16, 32, 64), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    net.set_train()
    _cell_graph_executor.compile(net, x)
