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

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import _Context
from mindspore.parallel._transformer import TransformerOpParallelConfig
from mindspore.nn.wrap.cell_wrapper import PipelineCell, MicroBatchInterleaved
from mindspore.train import Model
from tests.ut.python.parallel.test_parallel_transformer import TransformerEncoderNet


def test_pipeline_with_micro_interleaved_and_slice_activation():
    """
    Feature: pipeline parallel with micro_interleaved
    Description:pipeline + micro_interleaved + recompute + slice_activation
    Expectation:success
    """
    _Context().set_backend_policy("vm")
    bs = 32
    pp = 2
    micro_interleaved = 2
    encoder_layers = 4
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=pp,
                                      full_batch=True, parallel_mode="semi_auto_parallel",
                                      enable_parallel_optimizer=True)
    cf = TransformerOpParallelConfig(data_parallel=2, model_parallel=2, pipeline_stage=pp,
                                     vocab_emb_dp=False, optimizer_shard=True)
    pipeline_net = TransformerEncoderNet(batch_size=bs // pp // micro_interleaved,
                                         en_layer=encoder_layers, de_layer=0, parallel_config=cf)
    pipeline_net.embedding.pipeline_stage = 0
    for index, block in enumerate(pipeline_net.network.encoder.blocks):
        block.recompute(recompute_slice_activation=True)
        block.pipeline_stage = index // (encoder_layers // pp)

    pipeline_cell_net = PipelineCell(MicroBatchInterleaved(pipeline_net, micro_interleaved), 4)
    encoder_input_value = Tensor(np.ones((bs, 20)), mstype.int32)
    encoder_input_mask = Tensor(np.ones((bs, 20, 20)), mstype.float16)
    label = Tensor(np.ones((bs, 20)), mstype.int32)
    mask = Tensor(np.ones((bs, 20)), mstype.float32)
    params = pipeline_cell_net.trainable_params()
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(pipeline_cell_net, optimizer=optimizer)
    model.train_network.compile(encoder_input_value, encoder_input_mask, label, mask)
    _Context().set_backend_policy("ge")
