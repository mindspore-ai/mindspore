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
""" test group info """
import os
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.optim import  Momentum
from mindspore.ops import operations as P
from mindspore import context
from mindspore.train.serialization import restore_group_info_list


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class Net3(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2, strategy3):
        super(Net3, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.fc3 = P.MatMul().shard(strategy3)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 16]).astype(np.float32)), name="weight2", parallel_optimizer=False)
        self.p3 = Parameter(Tensor(np.ones([16, 16]).astype(np.float32)), name="weight3")

    def construct(self, x, y):
        x = self.fc1(x, self.p1)
        x = self.fc2(x, self.p2)
        z = x - y
        z = self.fc3(z, self.p3)
        return z


def auto_parallel_compile_net(strategy1=None, strategy2=None, strategy3=None):
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([32, 48]).astype(np.float32))
    label = Tensor(np.zeros([32, 16]).astype(np.float32))
    net = Net3(strategy1, strategy2, strategy3)
    auto_parallel = context.get_auto_parallel_context("parallel_mode") in ["semi_auto_parallel", "auto_parallel"]
    if auto_parallel:
        net = _VirtualDatasetCell(net)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = TrainOneStepCell(net, optimizer).set_comm_fusion(4)
    train_network.set_train()
    _cell_graph_executor.compile(train_network, inputs, label, phase="train")



def test_mirror_group():
    """
    Feature: save and load mirror group
    Description: semi-auto parallel, disable parallel optimizer.
    Expectation: group info list match expectation value.
    """
    os.environ['GROUP_INFO_FILE'] = "./test_mirror_group.pb"
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=32, enable_parallel_optimizer=False)
    auto_parallel_compile_net(((8, 1), (1, 4)), ((32, 1), (1, 1)), ((8, 4), (4, 1)))
    group_info_list = restore_group_info_list("./test_mirror_group.pb")
    assert group_info_list == [0, 4, 8, 12, 16, 20, 24, 28]
    context.reset_auto_parallel_context()
    del os.environ['GROUP_INFO_FILE']

def test_mirror_group_auto_parallel():
    """
    Feature: save and load mirror group
    Description: auto parallel, disable parallel optimizer.
    Expectation: group info list match expectation value.
    """
    os.environ['GROUP_INFO_FILE'] = "./test_mirror_group_auto_parallel.pb"
    context.set_auto_parallel_context(parallel_mode="auto_parallel",
                                      device_num=32, enable_parallel_optimizer=False)
    auto_parallel_compile_net(((8, 1), (1, 4)), ((32, 1), (1, 1)), ((8, 4), (4, 1)))
    group_info_list = restore_group_info_list("./test_mirror_group_auto_parallel.pb")
    assert group_info_list == [0, 4, 8, 12, 16, 20, 24, 28]
    context.reset_auto_parallel_context()
    del os.environ['GROUP_INFO_FILE']

def test_data_parallel_group():
    """
    Feature: save and load mirror group
    Description: data parallel , disable parallel optimizer.
    Expectation: group info list match expectation value.
    """
    os.environ['GROUP_INFO_FILE'] = "./test_data_parallel_group.pb"
    context.set_auto_parallel_context(parallel_mode="data_parallel", dataset_strategy="data_parallel",
                                      device_num=32, enable_parallel_optimizer=False)
    auto_parallel_compile_net(((8, 1), (1, 4)), ((32, 1), (1, 1)), ((8, 4), (4, 1)))
    group_info_list = restore_group_info_list("./test_data_parallel_group.pb")
    assert group_info_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 31]
    context.reset_auto_parallel_context()
    del os.environ['GROUP_INFO_FILE']

def test_mirror_group_parallel_optimizer():
    """
    Feature: save and load mirror group
    Description: semi-auto parallel, enable parallel optimizer.
    Expectation: group info list match expectation value.
    """
    os.environ['GROUP_INFO_FILE'] = "./test_mirror_group_parallel_optimizer.pb"
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 1},
                                      enable_parallel_optimizer=True)
    auto_parallel_compile_net(((8, 1), (1, 4)), ((32, 1), (1, 1)), ((8, 4), (4, 1)))
    group_info_list = restore_group_info_list("./test_mirror_group_parallel_optimizer.pb")
    assert group_info_list == [0]
    context.reset_auto_parallel_context()
    del os.environ['GROUP_INFO_FILE']

def test_mirror_group_parallel_optimizer_not_full_shard():
    """
    Feature: save and load mirror group
    Description: semi-auto parallel, enable parallel optimizer but not fully shard.
    Expectation: group info list match expectation value.
    """
    os.environ['GROUP_INFO_FILE'] = "./test_mirror_group_parallel_optimizer_not_full_shard.pb"
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 2},
                                      enable_parallel_optimizer=True, optimizer_weight_shard_size=2)
    auto_parallel_compile_net(((8, 1), (1, 4)), ((32, 1), (1, 1)), ((8, 4), (4, 1)))
    group_info_list = restore_group_info_list("./test_mirror_group_parallel_optimizer_not_full_shard.pb")
    assert group_info_list == [0, 8, 16, 24]
    context.reset_auto_parallel_context()
    del os.environ['GROUP_INFO_FILE']

def test_pipeline_split_stage0_mirror_group():
    """
    Feature: save and load mirror group
    Description: semi-auto parallel, pipeline parallel.
    Expectation: group info list match expectation value.
    """
    import mindspore as ms
    from mindspore import Model
    from .test_pipeline_split import PipelineCell, PipelineSplit, DatasetLenet
    os.environ['GROUP_INFO_FILE'] = "./test_pipeline_split_stage0_mirror_group.pb"
    context.set_auto_parallel_context(device_num=64, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="data_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 8))
    strategy2 = ((4, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    group_info_list = restore_group_info_list("./test_pipeline_split_stage0_mirror_group.pb")
    assert group_info_list == [0, 8, 16, 24]
    del os.environ['GROUP_INFO_FILE']

def test_pipeline_split_stage1_mirror_group():
    """
    Feature: save and load mirror group
    Description: semi-auto parallel, pipeline parallel.
    Expectation: group info list match expectation value.
    """
    import mindspore as ms
    from mindspore import Model
    from .test_pipeline_split import PipelineCell, PipelineSplit, DatasetLenet
    os.environ['GROUP_INFO_FILE'] = "./test_pipeline_split_stage1_mirror_group.pb"
    context.set_auto_parallel_context(device_num=64, global_rank=63, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="data_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 8))
    strategy2 = ((4, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    group_info_list = restore_group_info_list("./test_pipeline_split_stage1_mirror_group.pb")
    assert group_info_list == [39, 47, 55, 63]
    del os.environ['GROUP_INFO_FILE']
