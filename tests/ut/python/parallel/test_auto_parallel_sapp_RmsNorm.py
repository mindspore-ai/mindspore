# Copyright 2024 Huawei Technologies Co., Ltd
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

import re
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore import context, Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.nn import TrainOneStepCell, Momentum

class RmsNormNet(nn.Cell):
    def __init__(self, weight):
        super().__init__()
        self.matmul = P.MatMul()
        self.rms_norm = P.RmsNorm()
        self.normalized_shape = weight.shape[1:]
        self.gamma = Parameter(initializer('ones', self.normalized_shape), name="gamma")

    def construct(self, x, y):
        out = self.matmul(x, y)
        out, _ = self.rms_norm(out, self.gamma)
        return out

def test_auto_parallel_sapp_RmsNorm():
    """
    Feature: test RmsNorm in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    x = Tensor(np.ones([8, 320]), dtype=ms.float32)
    y = Tensor(np.ones([320, 409600]), dtype=ms.float32)
    w = Tensor(np.ones([64, 409600]), dtype=ms.float32)

    net = RmsNormNet(w)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x, y, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(train_net)
    for (k, v) in strategies.items():
        if re.search('RmsNorm-op0', k) is not None:
            assert v == [[8, 1], [1]]
