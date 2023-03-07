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

import os
import numpy as np

from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.parallel._transformer.moe import MoE
from mindspore.parallel._transformer import TransformerOpParallelConfig, MoEConfig


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


EXPERT_NUM = 64
CAPACITY_FACTOR = 8.0
AUX_LOSS_FACTOR = 0.01

DATA_PARALLEL = 8
MODEL_PARALLEL = 1
EXPERT_PARALLEL = 8

HIDDEN_SIZE = 6144
FFN_HIDDEN_SIZE = 6144 * 4

_x = Tensor(np.random.randn(1024, HIDDEN_SIZE), dtype=mstype.float16)


class Net(Cell):

    def __init__(self, hidden_size, ffn_hidden_size, moe_config, parallel_config):
        super(Net, self).__init__()
        self.output = MoE(hidden_size=hidden_size,
                          dropout_rate=0.1,
                          ffn_hidden_size=ffn_hidden_size,
                          param_init_type=mstype.float16,
                          hidden_act="fast_gelu",
                          moe_config=moe_config,
                          parallel_config=parallel_config.moe_parallel_config)

    def construct(self, x):
        mlp_logit, aux_loss = self.output(x)
        return mlp_logit, aux_loss


def compile_net(net, *inputs):
    net.set_auto_parallel()
    net.set_train(False)
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


def test_compile_moe_with_gpea_1():
    """
    Feature: test compile MoE net which applies grouped pariwise exchange alltoall(gpea) method
    Description: set gpea_num=2 when 8 devices
    Expectation: compile success
    """
    os.environ['GPEA_NUM'] = "2"
    os.environ['GPEA_RESHAPE_SCALE_AXIS'] = "2,1"

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=8,
                                      global_rank=0,
                                      full_batch=True,
                                      enable_alltoall=True)

    moe_config = MoEConfig(
        expert_num=EXPERT_NUM,
        capacity_factor=CAPACITY_FACTOR,
        aux_loss_factor=AUX_LOSS_FACTOR,
    )

    parallel_config = TransformerOpParallelConfig(data_parallel=DATA_PARALLEL,
                                                  model_parallel=MODEL_PARALLEL,
                                                  expert_parallel=EXPERT_PARALLEL)

    net = Net(HIDDEN_SIZE, FFN_HIDDEN_SIZE, moe_config, parallel_config)
    compile_net(net, _x)
    del os.environ['GPEA_NUM']
    del os.environ['GPEA_RESHAPE_SCALE_AXIS']


def test_compile_moe_with_gpea_2():
    """
    Feature: test compile MoE net which applies grouped pariwise exchange alltoall(gpea) method
    Description: set gpea_num=4 when 8 devices
    Expectation: compile success
    """
    os.environ['GPEA_NUM'] = "4"
    os.environ['GPEA_RESHAPE_SCALE_AXIS'] = "2,1"

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=8,
                                      global_rank=0,
                                      full_batch=True,
                                      enable_alltoall=True)

    moe_config = MoEConfig(
        expert_num=EXPERT_NUM,
        capacity_factor=CAPACITY_FACTOR,
        aux_loss_factor=AUX_LOSS_FACTOR,
    )

    parallel_config = TransformerOpParallelConfig(data_parallel=DATA_PARALLEL,
                                                  model_parallel=MODEL_PARALLEL,
                                                  expert_parallel=EXPERT_PARALLEL)

    net = Net(HIDDEN_SIZE, FFN_HIDDEN_SIZE, moe_config, parallel_config)
    compile_net(net, _x)
    del os.environ['GPEA_NUM']
    del os.environ['GPEA_RESHAPE_SCALE_AXIS']
