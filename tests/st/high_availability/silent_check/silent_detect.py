# Copyright 2025 Huawei Technologies Co., Ltd
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
"""silent detect"""

import os
import sys
import time
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, Parameter, context, ops
from mindspore.communication import init, get_rank
from mindspore.nn import Momentum, TrainOneStepCell

context.set_context(mode=context.GRAPH_MODE)
init()
ms.set_seed(1)
np.random.seed(1)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(1, 8)
        self.fc2 = nn.Dense(8, 8)
        self.relu = ops.ReLU()
        self.eod_mask = ops.auto_generate.GenerateEodMaskV2()
        self.cur_step = Parameter(Tensor(-1, ms.int64), requires_grad=False)
        rank_id = get_rank()
        if rank_id == 2:
            self.flip_mode = 'bitflip_designed'
        else:
            self.flip_mode = 'multiply'
        print(f"process: {os.getpid()}, rank: {rank_id}, flip_mode: {self.flip_mode}")

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        ele_pos = Tensor(0, ms.int64)
        seed = Tensor(0, ms.int64)
        offset = Tensor(0, ms.int64)
        start = 0
        steps = [5]
        error_mode = 'cycle'
        multiply_factor = 1.0
        bit_pos = 0
        flip_probability = 0.0
        # GenerateEodMaskV2()(input=<Tensor>, ele_pos=<Tensor>, cur_step=<Tensor>, seed=<Tensor>
        #   , offset=<Tensor>, start=<int>, steps=<int, list of int, tuple of int>, error_mode=<string>
        #   , flip_mode=<string>, multiply_factor=<float>, bit_pos=<int>, flip_probability=<float>)
        self.cur_step = self.cur_step + 1
        x = self.eod_mask(x, ele_pos, self.cur_step, seed, offset, start, steps, error_mode, self.flip_mode,
                          multiply_factor, bit_pos, flip_probability)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    parallel_mode = "semi_auto_parallel"
    if len(sys.argv) > 1:
        parallel_mode = sys.argv[1]
    print(f"parallel_mode: {parallel_mode}")

    context.set_auto_parallel_context(device_num=8, parallel_mode=parallel_mode)
    net = Net()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net = TrainOneStepCell(net, optimizer)
    net.set_train()
    epochs = 200 if parallel_mode in ['auto_parallel', 'semi_auto_parallel'] else 1
    for i in range(epochs):
        inputs = Tensor(np.random.rand(8, 1).astype(np.float32))
        net(inputs)
        time.sleep(1)
