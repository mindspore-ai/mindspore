# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.train import Model
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.parallel import set_algo_parameters
from mindspore.common.api import _cell_graph_executor
from tests.dataset_mock import MindData


class Dataset(MindData):
    def __init__(self, input_ids, length=3):
        super(Dataset, self).__init__(size=length)
        self.input_ids = input_ids
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.input_ids

    def reset(self):
        self.index = 0


class Net(nn.Cell):
    def __init__(self,
                 param_init='normal',
                 height=40000,
                 width=5120,
                 compute_type=mstype.float16):
        super().__init__()
        self.param = Parameter(initializer(param_init, [height, width]),
                               name='param', parallel_optimizer=False)
        self.param_two = Parameter(initializer(param_init, [height, width]),
                                   name='param_two', parallel_optimizer=False)
        self.matmul = P.MatMul(transpose_b=True)
        self.cast = P.Cast()
        self.add = P.Add()
        self.gather = P.Gather()
        self.dtype = compute_type
        self.width = width

    def construct(self, input_ids):
        input_ids = self.add(input_ids, input_ids)
        output_g = self.gather(self.param, input_ids, 0)
        output_r = P.Reshape()(output_g, (-1, self.width))
        output_gt = self.gather(self.param_two, input_ids, 0)
        output_rt = P.Reshape()(output_gt, (-1, self.width))
        output_m = self.matmul(self.cast(output_r, self.dtype), self.cast(self.param, self.dtype))
        output_mt = self.matmul(output_rt, self.param_two)
        output = self.add(output_m, output_mt)
        return output


def test_rec_shared_param_strmodif():
    '''
    Feature: auto_parallel_recursive_programming strategy modification when two operators share the same parameter
    Description: Modify the strategy of Gather following MatMul/Cast
    Expectation: Get expected strategies by check key op
    '''
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8,
                                      search_mode="recursive_programming", full_batch=False)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    net = Net()
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    input_ids = Tensor(np.ones((2, 1024)), mstype.int32)
    dataset = Dataset(input_ids)
    opt = nn.Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    stras = _cell_graph_executor._get_shard_strategy(model._train_network)
    for (k, v) in stras.items():
        if re.search("Gather", k) is not None:
            assert v == [[1, 1], [8, 1]]
    context.reset_auto_parallel_context()
