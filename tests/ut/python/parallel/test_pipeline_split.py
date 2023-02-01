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
import os
import shutil
import glob

import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell, MicroBatchInterleaved, _MicroBatch, Cell

class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def get_batch_size(self):
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


class MatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        if param is not None:
            self.param = param
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)
        self.cast = P.Cast()
        self.dtype = dtype

    def construct(self, x):
        out = self.matmul(self.cast(x, self.dtype), self.cast(self.param, self.dtype))
        out = self.matmul1(out, self.cast(self.param1, self.dtype))
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            cell.pipeline_stage = i
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x


class PipelineSplit2(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.cell = Net(strategy1, strategy2, self.param, dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x


class PipelineDupCell(Cell):
    def __init__(self, network, micro_size):
        super(PipelineDupCell, self).__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        for _ in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add()
            self.add_list.append(self.add)

    def construct(self, *inputs):
        ret = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output = self.network(*micro_input)
            if ret is not None:
                ret = self.add_list[i](ret, output)
            else:
                ret = output
        return ret


def test_pipeline_split_no_end():
    """
    Feature: Test pipeline without end node.
    Description: Expect get runtimeError.
    Expectation: Successful.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineDupCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    with pytest.raises(RuntimeError):
        model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage0():
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1():
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage0():
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage1():
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage0_predict():
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineSplit2(strategy1, strategy2)
    model = Model(net)
    model.predict(data, label)


def test_pipeline_split_shared_parameter_stage1_predict():
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineSplit2(strategy1, strategy2)
    model = Model(net)
    model.predict(data, label)


def test_pipeline_split_stage0_opt_shard():
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage0_opt_shard_with_no_data_parallel():
    '''
    Feature: pipeline + opt_shard
    Description: In pipeline mode, if there is no data parallel, opt_shard is True, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=16, global_rank=2, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 1), (1, 4))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 8)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage0_opt_shard_with_requires_grad_false():
    '''
    Feature: pipeline + opt_shard
    Description: In pipeline mode, if opt_shard is True and param's requiers_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=2, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 8)
    net.network.cell.block[0].param1.requiers_grad = False
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1_opt_shard():
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage0_opt_shard():
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_stage1_opt_shard():
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit2(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_with_micro_batch_interleaved_stage0():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_with_micro_batch_interleaved_stage1():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_with_micro_batch_interleaved_stage0_opt_shard():
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit2(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_shared_parameter_with_micro_batch_interleaved_stage1_opt_shard():
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    net = PipelineCell(MicroBatchInterleaved(PipelineSplit2(strategy1, strategy2), micro_batch_interleaved), 4)
    params = net.network.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def run_pipeline_split_function(pipeline_net, micro_batch_interleaved=1):
    """
    Feature: test PipelineSplitSharedParameter with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = PipelineCell(MicroBatchInterleaved(pipeline_net, micro_batch_interleaved), 4)
    params = net.infer_param_pipeline_stage()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


class TestPipelineSplitWithNoOptimizer:
    def setup_method(self):
        self.output_path = './graphs' + self.__str__()
        context.set_context(save_graphs=3,
                            save_graphs_path=self.output_path)

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def cat_fp16_from_ir(self, pattern, target_count):
        """
        This function will check the float16 count with the golden one.
        :param pattern: The match pattern for the specific count
        :param target_count: The gold float16 count in the Ir files
        """
        ir_files = glob.glob(os.path.join(self.output_path, 'rank_0', '*_validate*.ir'))
        assert len(ir_files) == 1
        appear_count = 0
        with open(ir_files[0], 'r') as fp:
            for line in fp:
                if pattern in line:
                    appear_count += 1
        assert appear_count == target_count

    def test_pipeline_with_no_parallel_optimizer_and_micro(self):
        """
        Feature: Test Pipeline with Mirror Operator.
        Description: When using fp16 computation, there should be only one mirror operator for one parameter.
        Expectation: the number of the float16 tensor is not equal to 16, 16 is obtained by manually checked graph.
                     the number of the Mirror is not equal to 2, 2 is obtained by manually checked graph.
        """
        context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2,
                                          enable_parallel_optimizer=False)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        strategy1 = ((16, 1), (1, 1))
        strategy2 = ((8, 1), (1, 1))
        pipeline_net = PipelineSplit(strategy1, strategy2, dtype=ms.float16)
        run_pipeline_split_function(pipeline_net, micro_batch_interleaved=1)
        self.cat_fp16_from_ir(pattern='grad_mirror_MirrorMicroStepOperator',
                              target_count=2)
        self.cat_fp16_from_ir(pattern='Cast(',
                              target_count=14)

    def test_pipeline_with_micro_batch_no_parallel_optimizer(self):
        """
        Feature: Test Pipeline with Mirror Operator, when enabled the micro batch interleave.
        Description: When using fp16 computation, there should be only one mirror operator for one parameter.
        Expectation: the number of the float16 tensor is not equal to 16, 16 is obtained by manually checked graph.
                     the number of the Mirror is not equal to 2, 2 is obtained by manually checked graph.
        """
        context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2,
                                          enable_parallel_optimizer=False)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        strategy1 = ((16, 1), (1, 1))
        strategy2 = ((8, 1), (1, 1))
        pipeline_net = PipelineSplit(strategy1, strategy2, dtype=ms.float16)
        run_pipeline_split_function(pipeline_net, micro_batch_interleaved=2)
        self.cat_fp16_from_ir(pattern='grad_mirror_MirrorMicroStepOperator',
                              target_count=2)
        self.cat_fp16_from_ir(pattern='Cast(',
                              target_count=26)

def test_pipeline_split_stage0_device_num_48():
    """
    Feature: test PipelineSplit with 48 devices in auto parallel.
    Description: net with pipeline parallel in auto parallel mode using 48 devices, stage0.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=48, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(device_target="Ascend")
    data = Tensor(np.ones([32 * 6, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64 * 6, 64]), dtype=ms.float32)
    strategy1 = ((3, 8), (8, 1))
    strategy2 = ((24, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[0].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1_device_num_48():
    """
    Feature: test PipelineSplit with 48 devices in auto parallel.
    Description: net with pipeline parallel in auto parallel mode using 48 devices, stage1.
    Expectation: success.
    """
    context.set_auto_parallel_context(device_num=48, global_rank=24, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(device_target="Ascend")
    data = Tensor(np.ones([32 * 6, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64 * 6, 64]), dtype=ms.float32)
    strategy1 = ((3, 8), (8, 1))
    strategy2 = ((24, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.network.cell.block[1].trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
