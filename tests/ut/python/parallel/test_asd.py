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
import numpy as np
import os
import subprocess
import shutil
import mindspore
from mindspore import context, nn, Tensor, Parameter
from mindspore.ops.silent_check import ASDBase
from mindspore.ops import operations as P
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.train import Model
from .test_pipeline_split import DatasetLenet

from mindspore.ops.silent_check import LayerNormASD

ir_path = "./asd_graph"


def setup_function():
    # operator substitution for silent check
    os.environ['NPU_ASD_ENABLE'] = "1"
    P.LayerNorm = LayerNormASD
    # set context
    context.set_context(save_graphs=True, save_graphs_path=ir_path)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")


def teardown_function():
    del os.environ['NPU_ASD_ENABLE']
    context.set_context(save_graphs=False)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)


class GatherASD(ASDBase):
    def __init__(self, *args, **kwargs):
        super().__init__(P.Gather, *args, **kwargs)
        self.pre_val, self.min_val, self.max_val, self.cnt = self.generate_params()

    def __call__(self, input_params, input_indices, axis):
        if self.enable_check:
            input_params = self.check_op(
                input_params, self.pre_val, self.min_val, self.max_val, self.cnt, None)
            self.cnt += 1
        return self.op(input_params, input_indices, axis)


class WordEmbedding(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul = P.MatMul()

    def construct(self, x):
        return self.matmul(x, self.w), self.w


class VocabEmbedding(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.embedding_table = Parameter(Tensor(np.ones(shape), mindspore.float32), name="embedding_table")
        self.gather = GatherASD()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_ids):
        input_ids = self.reshape(input_ids, (-1, 1))
        output = self.gather(self.embedding_table, input_ids, 0)
        temp_shape = self.shape(output)
        output = self.reshape(output, (-1, temp_shape[-1]))
        return output, self.embedding_table.value()


class FC(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul = P.MatMul()

    def construct(self, x):
        return self.matmul(x, self.w)


class LMHead(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x, w):
        x = self.matmul1(x, self.w)
        return self.matmul2(x, w), x + x


class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mamtul = P.MatMul()

    def construct(self, x, w):
        x = self.mamtul(x, w)
        return x


class NetWithParaGraph(nn.Cell):
    def __init__(self):
        super().__init__()
        shape = (8, 8)
        self.word_embedding = WordEmbedding(shape)
        self.word_embedding.pipeline_stage = 0
        self.decoder1 = FC(shape)
        self.decoder1.pipeline_stage = 0
        self.decoder2 = FC(shape)
        self.decoder2.pipeline_stage = 1
        self.lm_head = LMHead(shape)
        self.lm_head.pipeline_stage = 1
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1)
        self.layer_norm.pipeline_stage = 1
        normalized_shape = [8]
        self.gamma = Parameter(Tensor(np.ones(normalized_shape), mindspore.float32), name="gamma")
        self.gamma.pipeline_stage = 1
        self.beta = Parameter(Tensor(np.ones(normalized_shape), mindspore.float32), name="beta")
        self.beta.pipeline_stage = 1

    def construct(self, x):
        x, w = self.word_embedding(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x, _ = self.lm_head(x, w)
        x, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return x


class CustomASDNetWithParaGraph(nn.Cell):
    def __init__(self):
        super().__init__()
        shape = (8, 8)
        self.word_embedding = VocabEmbedding(shape)
        self.word_embedding.pipeline_stage = 0
        self.decoder1 = FC(shape)
        self.decoder1.pipeline_stage = 0
        self.decoder2 = FC(shape)
        self.decoder2.pipeline_stage = 1

    def construct(self, input_ids):
        x, _ = self.word_embedding(input_ids)
        x = self.decoder1(x)
        x = self.decoder2(x)
        return x


def check_output(num_cnt):
    file = os.path.join(ir_path, "rank_0/*pipeline_split*.ir")
    prim_name = "MirrorSilentCheck("
    output = subprocess.check_output(["grep -r '%s' %s |wc -l" % (prim_name, file)], shell=True)
    out = str(output, 'utf-8').strip()
    assert out == str(num_cnt)


def test_asd_with_custom_asd_op_stage0():
    """
    Feature: test silent check.
    Description: test custom asd op with parameter graph in stage0.
    Expectation: pass.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2)

    net = CustomASDNetWithParaGraph()
    net.word_embedding.gather.shard(((1, 1), (4, 1)))
    net.decoder1.matmul.shard(((4, 1), (1, 1)))
    net.decoder2.matmul.shard(((4, 1), (1, 1)))

    params = net.trainable_params()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    net = nn.WithLossCell(net, loss)

    micro_batch_num = 4
    pp_net = PipelineCell(net, micro_batch_num)

    data = Tensor(np.ones([4, 8]), dtype=mindspore.int32)
    label = Tensor(np.ones([32, 8]), dtype=mindspore.float32)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(pp_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)

    check_output(num_cnt=1)


def test_asd_with_parameter_graph_stage1():
    """
    Feature: test silent check.
    Description: test LayerNorm asd with parameter graph in stage1.
    Expectation: pass.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=4, pipeline_stages=2)

    net = NetWithParaGraph()
    net.word_embedding.matmul.shard(((4, 1), (1, 1)))
    net.decoder1.matmul.shard(((4, 1), (1, 1)))
    net.decoder2.matmul.shard(((4, 1), (1, 1)))
    net.lm_head.matmul1.shard(((4, 1), (1, 1)))
    net.lm_head.matmul2.shard(((4, 1), (1, 1)))

    params = net.trainable_params()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    net = nn.WithLossCell(net, loss)

    micro_batch_num = 4
    pp_net = PipelineCell(net, micro_batch_num)

    data = Tensor(np.ones([32, 8]), dtype=mindspore.float32)
    label = Tensor(np.ones([32, 8]), dtype=mindspore.float32)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(pp_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)

    check_output(num_cnt=micro_batch_num)
