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
# ============================================================================
""" test parallel optimizer shared test """
import os
import shutil
import glob
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell, MicroBatchInterleaved, PipelineCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore import context


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class Net(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 16]).astype(np.float32)), name="weight2", parallel_optimizer=False)
        self.sub = P.Sub()

    def construct(self, x, y):
        x = P.Cast()(x, ms.float16)
        p1 = P.Cast()(self.p1, ms.float16)
        p2 = P.Cast()(self.p2, ms.float16)
        x = self.fc1(x, p1)
        x = self.fc2(x, p2)
        return self.sub(x, y)


class Net2(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net2, self).__init__()
        self.net1 = Net(strategy1, strategy2)
        self.net2 = Net(strategy1, strategy2)
        self.net1.pipeline_stage = 0
        self.net2.pipeline_stage = 1
        self.sub = P.Sub()

    def construct(self, x, y):
        out1 = self.net1(x, y)
        out2 = self.net2(x, y)
        return self.sub(out1, out2)


def auto_parallel_compile_net(mode, dev_num, net, strategy1=None, strategy2=None,
                              interleaved_batch=2, stages=1, micro_size=1):
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode=mode, device_num=dev_num, pipeline_stages=stages,
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 1})
    inputs = Tensor(np.ones([64, 48]).astype(np.float32))
    label = Tensor(np.zeros([64, 16]).astype(np.float32))
    net = MicroBatchInterleaved(net(strategy1, strategy2), interleaved_batch)
    if stages > 1:
        net = PipelineCell(net, micro_size=micro_size)
    net = _VirtualDatasetCell(net)
    parameters = net.trainable_params() if stages == 1 else net.infer_param_pipeline_stage()
    optimizer = Momentum(parameters, learning_rate=0.1, momentum=0.9)
    train_network = TrainOneStepCell(net, optimizer).set_comm_fusion(4)
    train_network.set_train()
    _cell_graph_executor.compile(train_network, inputs, label, phase="train")
    context.reset_auto_parallel_context()
    return train_network


class TestSharedParameterCast:
    def setup_method(self):
        self.output_path = './graphs' + self.__str__()
        context.set_context(save_graphs=3,
                            save_graphs_path=self.output_path)

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def cat_fp16_from_ir(self, target_count):
        """
        This function will check the float16 count with the golden one.
        :param target_count: The gold float16 count in the Ir files
        """
        # Find the step_parallel_end
        ir_files = glob.glob(os.path.join(self.output_path, 'rank_0', '*_validate*.ir'))
        assert len(ir_files) == 1
        appear_count = 0
        with open(ir_files[0], 'r') as fp:
            for line in fp:
                if 'Float16' in line:
                    appear_count += 1
        assert appear_count == target_count

    def test_optimizer_fp16(self):
        """
        Feature: CastBeforeAllGather.
        Description: The order should be load, cast(from fp32 to fp16), AllGather.
        Expectation: the number of the float16 tensor is not equal to 27, 27 is obtained by manually checked graph.
        """
        auto_parallel_compile_net("semi_auto_parallel", 8, Net, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1)
        self.cat_fp16_from_ir(target_count=23)

    def test_optimizer_fp16_micro_batch(self):
        """
        Feature: CastBeforeAllGather with MicroBatchInterleave applied.
        Description: The order should be load, cast(from fp32 to fp16), AllGather.
        Expectation: the number of the float16 tensor is not equal to 41, 41 is obtained by manually checked graph.
        """
        auto_parallel_compile_net("semi_auto_parallel", 8, Net, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=2)
        self.cat_fp16_from_ir(target_count=39)

    def test_optimizer_fp16_pipeline(self):
        """
        Feature: CastBeforeAllGather with PipeLine applied.
        Description: The order should be load, cast(from fp32 to fp16), AllGather.
        Expectation: the number of the float16 tensor is not equal to 27, 27 is obtained by manually checked graph.
        """
        auto_parallel_compile_net("semi_auto_parallel", 8, Net, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1,
                                  stages=1, micro_size=1)
        self.cat_fp16_from_ir(target_count=23)

    def test_optimizer_fp16_pipeline_micro_batch(self):
        """
        Feature: CastBeforeAllGather with MicroBatchInterleave and PipeLine applied.
        Description: The order should be load, cast(from fp32 to fp16), AllGather.
        Expectation: the number of the float16 tensor is not equal to 41, 41 is obtained by manually checked graph.
        """
        auto_parallel_compile_net("semi_auto_parallel", 8, Net, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=2,
                                  stages=1, micro_size=1)
        self.cat_fp16_from_ir(target_count=39)
