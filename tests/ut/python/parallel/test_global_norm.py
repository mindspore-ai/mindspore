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
""" test global norm test """
import re
import os
import shutil
import glob
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor, Parameter, Model
from mindspore.train import DynamicLossScaleManager
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell, MicroBatchInterleaved, PipelineCell
from mindspore.nn.optim import AdamWeightDecay
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import context


class OneParameterNet(nn.Cell):
    """Net definition"""
    def __init__(self, param_type, strategy1, strategy2):
        super(OneParameterNet, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.p1 = Parameter(Tensor(np.ones([48, 16]).astype(param_type)), name="weight1")
        self.sub = P.Sub().shard(strategy2)

    def construct(self, x, y):
        x = P.Cast()(x, ms.float16)
        p1 = P.Cast()(self.p1, ms.float16)
        x = self.fc1(x, p1)
        return self.sub(x, 0)

class Net(nn.Cell):
    """Net definition"""
    def __init__(self, param_type, strategy1, strategy2):
        super(Net, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(param_type)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 16]).astype(param_type)), name="weight2", parallel_optimizer=False)
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
    def __init__(self, param_type, strategy1, strategy2):
        super(Net2, self).__init__()
        self.net1 = Net(param_type, strategy1, strategy2)
        self.net2 = Net(param_type, strategy1, strategy2)
        self.net1.pipeline_stage = 0
        self.net2.pipeline_stage = 1
        self.sub = P.Sub()

    def construct(self, x, y):
        out1 = self.net1(x, y)
        out2 = self.net2(x, y)
        return self.sub(out1, out2)


def get_dataset():
    inputs = np.ones([64, 48]).astype(np.float32)
    label = np.zeros([64, 16]).astype(np.float32)

    def dataset_generator():
        for _ in range(10):
            yield inputs, label

    dataset = ds.GeneratorDataset(dataset_generator, column_names=["inputs", "label"])

    return dataset


class CustomOptimizer(AdamWeightDecay):
    def __init__(self, params):
        super(CustomOptimizer, self).__init__(params)
        self.optimizer = super(CustomOptimizer, self).construct

    def construct(self, gradients):
        grads = C.clip_by_global_norm(gradients)
        return self.optimizer(grads)


def auto_parallel_compile_net(mode, dev_num, net, strategy1=None, strategy2=None,
                              interleaved_batch=2, stages=1, micro_size=1, param_type=np.float32,
                              loss_scale_manager=None):
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode=mode, device_num=dev_num, enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 1},
                                      pipeline_stages=stages)

    net = MicroBatchInterleaved(net(param_type, strategy1, strategy2), interleaved_batch)
    if stages > 1:
        net = PipelineCell(net, micro_size=micro_size)
    net = _VirtualDatasetCell(net).set_comm_fusion(4)
    parameters = net.trainable_params() if stages == 1 else net.infer_param_pipeline_stage()
    optimizer = CustomOptimizer(parameters)
    if loss_scale_manager:
        model = Model(net, optimizer=optimizer, loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net, optimizer=optimizer)
    dataset = get_dataset()
    model.train(1, dataset)


class TestGlobalNormInserted:
    def setup_method(self):
        self.output_path = './graphs' + self.__str__()
        context.set_context(save_graphs=True,
                            save_graphs_path=self.output_path)

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def run_count_check(self, target_count, pattern):
        """
        This function will check the target_key counts with the golden one.
        :param target_count: The gold float16 count in the Ir files.
        :param pattern: The generated keyword in the Ir files.

        """
        # Find the step_parallel_end
        ir_files = glob.glob(os.path.join(self.output_path, 'rank_0', 'step_parallel_end*.ir'))
        assert len(ir_files) == 1
        appear_count = 0
        with open(ir_files[0], 'r') as fp:
            for line in fp:
                res = re.findall(pattern, line)
                if len(res) >= 1:
                    appear_count += 1
        assert appear_count == target_count

    def test_nonpipeline_global_norm_one_parameter(self):
        """
        Feature: Parallel ClipByGlobalNorm
        Description: Test the global norm using one parameter, there should be only one allreduce
        Expectation:When there is no PARALLEL_GLOBALNORM_IN_STAGES inserted
        """
        auto_parallel_compile_net("semi_auto_parallel", 8, OneParameterNet, ((1, 8), (8, 1)), ((8, 1), ()),
                                  interleaved_batch=1, param_type=np.float32)
        self.run_count_check(target_count=1, pattern=r"PARALLEL_GLOBALNORM_IN_STAGES")

    def test_nonpipeline_global_norm(self):
        """
        Feature: Parallel ClipByGlobalNorm
        Description: Test the global norm when running in semi auto parallel mode, scale for data parallel should be 8
        Expectation:When there is no real div inserted or AllReduce inserted
        """
        auto_parallel_compile_net("semi_auto_parallel", 8, Net, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1, param_type=np.float32)
        self.run_count_check(target_count=1, pattern=r"=8.*PARALLEL_GLOBALNORM_DIV")
        self.run_count_check(target_count=2, pattern=r"PARALLEL_GLOBALNORM")

    def test_pipeline_global_norm(self):
        """
        Feature: Parallel ClipByGlobalNorm
        Description: Test the global norm when running in pipeline mode, scale for data parallel should be 8
        Expectation: When there is no real div inserted or AllReduce inserted
        """
        auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1, stages=2, micro_size=2, param_type=np.float32)
        self.run_count_check(target_count=1, pattern=r"=16.*PARALLEL_GLOBALNORM_DIV")
        self.run_count_check(target_count=3, pattern=r"PARALLEL_GLOBALNORM")

    def test_pipeline_global_norm_loss_scale(self):
        """
        Feature: Parallel ClipByGlobalNorm
        Description: Test the global norm when running in pipeline mode, scale for data parallel should be 8
        Expectation: When there is no real div inserted or AllReduce inserted
        """
        auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1, stages=2, micro_size=2, param_type=np.float32,
                                  loss_scale_manager=DynamicLossScaleManager())
        self.run_count_check(target_count=1, pattern=r"=16.*PARALLEL_GLOBALNORM_DIV")
        self.run_count_check(target_count=3, pattern=r"PARALLEL_GLOBALNORM")


    def test_pipeline_global_norm_fp16(self):
        """
        Feature: Parallel ClipByGlobalNorm
        Description: Test the global norm when running in pipeline mode, scale for data parallel should be 8
        Expectation: When there is no real div inserted or AllReduce inserted
        """
        auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1, stages=2, micro_size=2, param_type=np.float16)
        self.run_count_check(target_count=1, pattern=r"=16.*PARALLEL_GLOBALNORM_DIV")
        self.run_count_check(target_count=3, pattern=r"PARALLEL_GLOBALNORM")

    def test_pipeline_global_norm_loss_scale_fp16(self):
        """
        Feature: Parallel ClipByGlobalNorm
        Description: Test the global norm when running in pipeline mode, scale for data parallel should be 8
        Expectation: When there is no real div inserted or AllReduce inserted
        """
        auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)),
                                  interleaved_batch=1, stages=2, micro_size=2, param_type=np.float16,
                                  loss_scale_manager=DynamicLossScaleManager())
        self.run_count_check(target_count=1, pattern=r"=16.*PARALLEL_GLOBALNORM_DIV")
        self.run_count_check(target_count=3, pattern=r"PARALLEL_GLOBALNORM")
