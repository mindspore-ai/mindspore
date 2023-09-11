# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, Conv2d, Flatten, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(
            self,
            conv_in_channel,
            conv_out_channel,
            reducemin_keep_dims=False,
            reducemin_axis=-1
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels=conv_in_channel,
            out_channels=conv_out_channel,
            kernel_size=1,
            stride=1,
            pad_mode="valid",
            has_bias=True,
            weight_init="ones",
            bias_init="ones",
        )
        self.reduce_min = P.ReduceMin(keep_dims=reducemin_keep_dims)
        self.flat = Flatten()
        self.reducemin_axis = reducemin_axis

    def construct(self, inputs):
        x = self.conv(inputs)
        x = self.reduce_min(x, self.reducemin_axis)
        x = self.flat(x)
        x = x * 1e-6
        return x


_x = Tensor(np.ones([32, 4, 8, 8]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_auto_parallel_reducemin_rec():
    """
    Feature: test reducemin net of auto parallel
    Description: using recursive algorithm
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="recursive_programming")
    net = Net(4, 64)
    compile_net(net)
