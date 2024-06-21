# Copyright 2024 Huawei Technologies Co., Ltd
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
import subprocess
import shutil
import numpy as np
import mindspore
from mindspore import lazy_inline, context, nn, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import ParallelValidator


class FC0(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul = P.MatMul()

    def construct(self, x):
        return self.matmul(x, self.weight), self.weight.value()


class FC1(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul = P.MatMul()

    def construct(self, x):
        return self.matmul(x, self.weight)


class FC2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    def construct(self, x, w):
        return self.matmul(x, w)


class SimpleNet(nn.Cell):
    @lazy_inline
    def __init__(self, shape):
        super().__init__()
        self.fc0 = FC0(shape)
        self.fc1 = FC1(shape)
        self.fc2 = FC2()
        self.fc0.pipeline_stage = 0
        self.fc1.pipeline_stage = 1
        self.fc2.pipeline_stage = 1

    def construct(self, x):
        x, w = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x, w)
        return x


class ShareParaNet(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.fc0 = FC2()
        self.fc1 = FC2()
        self.fc0.pipeline_stage = 0
        self.fc1.pipeline_stage = 1

    def construct(self, x):
        x = self.fc0(x, self.w)
        y = self.fc1(x, self.w)
        return y


class PipelineInferenceWrapper(nn.Cell):
    def __init__(self, network, micro_batch_num):
        super().__init__()
        self.network = network
        self.micro_batch_num = micro_batch_num
        self.concat = P.Concat(axis=0)

    def construct(self, x):
        ret = ()
        for i in range(self.micro_batch_num):
            micro_batch_size = x.shape[0] // self.micro_batch_num
            start = micro_batch_size * i
            end = micro_batch_size * (i + 1)
            micro_input = x[start:end]
            y = self.network(micro_input)
            ret = ret + (y,)
        ret = self.concat(ret)
        return ret


def compile_infer_net(net: nn.Cell, *inputs):
    net.set_train(False)
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


def test_pipeline_inference_first_stage():
    """
    Feature: Test pipeline inference graph split
    Description: The first stage graph split.
    Expectation: Successful.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=4, global_rank=0, parallel_mode="semi_auto_parallel",
                                      full_batch=True, pipeline_stages=2)

    batch_size, hidden_size = 8, 32
    net = PipelineInferenceWrapper(SimpleNet(shape=(hidden_size, hidden_size)), micro_batch_num=2)
    net.set_train(False)

    x = Tensor(np.ones((batch_size, hidden_size)), mindspore.float32)
    phase = compile_infer_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Send-0', ['pipeline_inference_SimpleNet_construct'], graph_id=1)
    assert validator.check_node_inputs_has('Send-1', ['pipeline_inference_SimpleNet_construct'], graph_id=1)


def test_pipeline_inference_last_stage():
    """
    Feature: Test pipeline inference graph split
    Description: The last stage graph split.
    Expectation: Successful.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=4, global_rank=3, parallel_mode="semi_auto_parallel",
                                      full_batch=True, pipeline_stages=2)

    batch_size, hidden_size = 8, 32
    net = PipelineInferenceWrapper(SimpleNet(shape=(hidden_size, hidden_size)), micro_batch_num=2)
    net.set_train(False)

    x = Tensor(np.ones((batch_size, hidden_size)), mindspore.float32)
    phase = compile_infer_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('call @graph_0', ['network.fc1.weight', 'network.fc0.weight', 'Receive-1'],
                                           graph_id=1)


def test_pipeline_inference_without_lazy_inline_first_stage():
    """
    Feature: Test pipeline inference does not send embedding
    Description: Shared parameter does not send between different stage.
    Expectation: Successful.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=4, global_rank=0, parallel_mode="semi_auto_parallel",
                                      full_batch=True, pipeline_stages=2)
    context.set_context(save_graphs=True, save_graphs_path="./pp_no_send_embed")

    batch_size, hidden_size = 8, 32
    net = PipelineInferenceWrapper(ShareParaNet(shape=(hidden_size, hidden_size)), micro_batch_num=2)
    net.set_train(False)
    x = Tensor(np.ones((batch_size, hidden_size)), mindspore.float32)
    if os.path.exists("./pp_no_send_embed/rank_0"):
        shutil.rmtree("./pp_no_send_embed/rank_0")
    compile_infer_net(net, x)
    file = "./pp_no_send_embed/rank_0/*validate*.ir"
    prim_name = "Send("
    output = subprocess.check_output(
        ["grep -r '%s' %s |wc -l" % (prim_name, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == '2'
    context.set_context(save_graphs=False)


def test_pipeline_inference_without_lazy_inline_last_stage():
    """
    Feature: Test pipeline inference does not send embedding
    Description: Shared parameter does not send between different stage.
    Expectation: Successful.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=4, global_rank=2, parallel_mode="semi_auto_parallel",
                                      full_batch=True, pipeline_stages=2)
    context.set_context(save_graphs=True, save_graphs_path="./pp_no_send_embed")

    batch_size, hidden_size = 8, 32
    net = PipelineInferenceWrapper(ShareParaNet(shape=(hidden_size, hidden_size)), micro_batch_num=2)
    net.set_train(False)
    x = Tensor(np.ones((batch_size, hidden_size)), mindspore.float32)
    if os.path.exists("./pp_no_send_embed/rank_0"):
        shutil.rmtree("./pp_no_send_embed/rank_0")
    compile_infer_net(net, x)
    file = "./pp_no_send_embed/rank_0/*validate*.ir"
    prim_name = "Receive("
    output = subprocess.check_output(
        ["grep -r '%s' %s |wc -l" % (prim_name, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == '2'
    context.set_context(save_graphs=False)


def test_pipeline_inference_result_broadcast():
    """
    Feature: Test pipeline inference graph split
    Description: Broadcast the last stage to all others.
    Expectation: Successful.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=4, global_rank=3, parallel_mode="semi_auto_parallel",
                                      full_batch=True, pipeline_stages=2, pipeline_result_broadcast=True)
    assert context.get_auto_parallel_context('pipeline_result_broadcast') is True

    batch_size, hidden_size = 8, 32
    net = PipelineInferenceWrapper(SimpleNet(shape=(hidden_size, hidden_size)), micro_batch_num=2)
    net.set_train(False)

    x = Tensor(np.ones((batch_size, hidden_size)), mindspore.float32)
    phase = compile_infer_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('call @graph_0', ['network.fc1.weight', 'network.fc0.weight', 'Receive-1'],
                                           graph_id=1)
    assert validator.check_node_inputs_has('AllReduce-0', ['Depend-0'], graph_id=1)
