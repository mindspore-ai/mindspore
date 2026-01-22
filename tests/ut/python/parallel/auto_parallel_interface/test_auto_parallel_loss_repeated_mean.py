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
# limitations under the License

import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_prim import Convolution
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from parallel.utils.utils import ParallelValidator
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode


class Net(Cell):
    def __init__(self, conv2d_weight_size, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1, pad=0,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = _get_cache_prim(Convolution)().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.conv2d_weight = Parameter(initializer("ones", conv2d_weight_size, ms.float32), "w1")
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.group = group

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight, None, self.stride, self.pad, self.dilation, False, (0, 0),
                          self.group)
        out = self.neg(out)
        return out


def test_convolution_data_parallel_loss_repeated_mean_true():
    """
    Feature: test convolution data parallel, loss_repeated_mean is True
    Description: shard n dimension
    Expectation: compile success, _VirtualDiv-0 in validate.ir graph
    """
    _x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    _b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    _w1_size = [8, 16, 2, 2]

    strategy1 = ((4, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((4, 1, 1, 1),)
    with no_init_parameters():
        net = Net(_w1_size, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1,
                  strategy2=strategy2)
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch", "loss_repeated_mean": True}
    train_net = set_parallel_mode(train_net, parallel_config)
    phase, _ = _cell_graph_executor.compile(train_net, _x, _b)

    # validation
    validator = ParallelValidator(train_net, phase)
    sub_graph = {
        '_VirtualDiv-0': ['Convolution-0'],
        'Neg-0': ['_VirtualDiv-0']
    }
    assert validator.check_graph_structure(sub_graph)


def test_convolution_data_parallel_loss_repeated_mean_false():
    """
    Feature: test convolution data parallel, loss_repeated_mean is False
    Description: shard n dimension
    Expectation: compile success, _VirtualDiv is not in validate.ir graph
    """
    _x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    _b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    _w1_size = [8, 16, 2, 2]

    strategy1 = ((4, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((4, 1, 1, 1),)
    with no_init_parameters():
        net = Net(_w1_size, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1,
                  strategy2=strategy2)
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch", "loss_repeated_mean": False}
    train_net = set_parallel_mode(train_net, parallel_config)
    phase, _ = _cell_graph_executor.compile(train_net, _x, _b)

    # validation
    validator = ParallelValidator(train_net, phase)
    sub_graph = {
        '_VirtualDiv-0': ['Convolution-0'],
        'Neg-0': ['_VirtualDiv-0']
    }
    assert validator.check_graph_structure(sub_graph) is False
