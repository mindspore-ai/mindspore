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
os.environ["MS_SIMULATION_LEVEL"] = "3"

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell, Momentum
from mindspore.nn import PipelineCell
from mindspore.communication.management import init, create_group, destroy_group, get_group_size, get_rank, \
    get_local_rank, get_world_rank_from_group_rank, get_group_rank_from_world_rank
from mindspore.communication.comm_func import barrier
from mindspore.mint.distributed.distributed import init_process_group, broadcast, recv, all_gather
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class DenseNet(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc2 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc3 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(q)
        v = self.fc3(k)
        return v

class PipelineNet(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(PipelineNet, self).__init__()
        self.stage1 = DenseNet(has_bias, activation)
        self.stage2 = DenseNet(has_bias, activation)
        self.stage1.pipeline_stage = 0
        self.stage2.pipeline_stage = 1

    def construct(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        return s2

input_ = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
label_ = Tensor(np.zeros([32, 128]).astype(np.float32))


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_run_graph_kbk():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 1.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    os.environ["OMPI_COMMAND"] = "1"
    os.environ["PMIX_RANK"] = "1"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net(input_, label_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_get_rank_id_env():
    """
    Feature: simulation level.
    Description: get rank id default when set simulation level 0.
    Expectation: return env rank id.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "8"
    os.environ["RANK_ID"] = "7"
    init()
    ret = get_rank()
    assert ret == 7
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_get_local_rank_id():
    """
    Feature: simulation level.
    Description: get local rank id when set simulation level 0.
    Expectation: return local rank id.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "9"
    init()
    ret = get_local_rank()
    assert ret == 1
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_create_group():
    """
    Feature: simulation level.
    Description: create group when set simulation level 0.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "7"
    init()
    group = "g-01234567"
    rank_ids = [i for i in range(8)]
    create_group(group, rank_ids)
    ret = get_group_size()
    assert ret == 32
    ret = get_group_size(group)
    assert ret == 8
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_destroy_group():
    """
    Feature: simulation level.
    Description: destroy group when set simulation level 0.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "7"
    init()
    group = "g8-01234567"
    rank_ids = [i for i in range(8)]
    create_group(group, rank_ids)
    destroy_group(group)
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_get_world_rank_from_group_rank():
    """
    Feature: simulation level.
    Description: get world rank from group rank when set simulation level 0.
    Expectation: return world rank.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "9"
    init()
    group = "g-to-w-8+01234567"
    rank_ids = [i + 8 for i in range(8)]
    create_group(group, rank_ids)
    ret = get_world_rank_from_group_rank(group, 3)
    assert ret == 11
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_get_group_rank_from_world_rank():
    """
    Feature: simulation level.
    Description: get local rank id default when set simulation level 0.
    Expectation: return local rank id.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "9"
    init()
    group = "w-to-g-8+01234567"
    rank_ids = [i + 8 for i in range(8)]
    create_group(group, rank_ids)
    ret = get_group_rank_from_world_rank(12, group)
    assert ret == 4
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_simulation_graph():
    """
    Feature: simulation level.
    Description: compile graph when set simulation level 0.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    _cell_graph_executor.compile(net, input_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_run_graph():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 1.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net(input_, label_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_build_model_with_dataset():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 1.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    data_list = []
    for _ in range(8):
        data_list.append((np.ones([32, 128]).astype(np.float32), np.zeros([32, 128]).astype(np.float32)))
    dataset = ds.GeneratorDataset(data_list, ["input", "label"])
    model = Model(net, loss_fn, optimizer)
    model.build(dataset)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_simu_execute_graph():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 3.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "3"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", enable_parallel_optimizer=True)
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net(input_, label_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_simu_execute_pipeline_graph():
    """
    Feature: simulation level.
    Description: run pipeline graph when set simulation level 3.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "3"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "0"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      pipeline_stages=2,
                                      enable_parallel_optimizer=True)
    net = PipelineNet()
    net.stage1.fc1.matmul.shard(((4, 1), (4, 1)))
    net.stage2.fc1.matmul.shard(((4, 1), (4, 1)))
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    loss_fn.pipeline_stage = 1
    net = WithLossCell(net, loss_fn)
    pipe_net = PipelineCell(net, 4)
    optimizer = Momentum(pipe_net.trainable_params(), learning_rate=0.1, momentum=0.9)

    data_list = []
    for _ in range(8):
        data_list.append((np.ones([32, 128]).astype(np.float32), np.zeros([32, 128]).astype(np.float32)))
    dataset = ds.GeneratorDataset(data_list, ["input", "label"])
    model = Model(pipe_net, optimizer=optimizer)
    model.train(1, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_simu_execute_simu_barrier():
    """
    Feature: simulation level.
    Description: run barrier when set simulation level 3.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "3"
    context.set_context(jit_level='O0')
    barrier()
    os.environ["MS_SIMULATION_LEVEL"] = ""

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_pyboost_comm():
    """
    Feature: simulation level.
    Description: run pyboost comm op when set simulation level 3.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "3"
    init_process_group()
    rank = get_rank()
    context.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True
    )
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=0)
    assert output_handle is None
    except_output_tensor = ms.Tensor(np.full(shape=(2, 4), fill_value=0.1, dtype=np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())

    output_tensor = [ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))]
    output_handle = all_gather(output_tensor, tensor)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor.asnumpy())

    out = recv(tensor, src=rank + 1)
    assert out == 0
    assert np.allclose(tensor.asnumpy(), except_output_tensor)
