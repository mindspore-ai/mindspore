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

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.context import set_auto_parallel_context, ParallelMode
from mindspore import context
from mindspore.ops import composite as C
from mindspore.parallel._transformer import Transformer
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell
from mindspore.train import Model
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss
grad_all = C.GradOperation(get_all=True)


class Dataset(MindData):
    def __init__(self, *inputs, length=3):
        super(Dataset, self).__init__(size=length)
        self.inputs = inputs
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.inputs

    def reset(self):
        self.index = 0


class NetWithLossFiveInputs(nn.Cell):
    def __init__(self, network):
        super(NetWithLossFiveInputs, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x1, x2, x3, x4, x5):
        predict, _, _ = self.network(x1, x2, x3, x4, x5)
        return self.loss(predict)


def parallel_shard(net, layer_num=2, data_parallel=4, model_parallel=2):
    dp_1_1 = ((data_parallel, 1, 1),)
    dp_1_1_1 = ((data_parallel, 1, 1, 1),)
    dp_mp_mp_1 = ((data_parallel, model_parallel), (model_parallel, 1))
    dp_1_1_mp = ((data_parallel, 1), (1, model_parallel))
    dp_1_mp_1 = ((data_parallel, 1), (model_parallel, 1))
    bias_add_config_1 = ((data_parallel, 1), (1,))
    bias_add_config_2 = ((data_parallel, model_parallel), (model_parallel,))
    transpose_dp_1_mp_1 = ((data_parallel, 1, model_parallel, 1),)
    transpose_dp_mp_1_1 = ((data_parallel, model_parallel, 1, 1),)
    batch_matmul_config = ((data_parallel, model_parallel, 1, 1), (data_parallel, model_parallel, 1, 1))
    real_div_config = ((data_parallel, model_parallel, 1, 1), ())
    sub_config = ((1,), (data_parallel, 1, 1, 1))
    mul_config = ((data_parallel, 1, 1, 1), (1,))
    add_config = ((data_parallel, 1, 1, 1), (data_parallel, model_parallel, 1, 1))
    drop_out_config = ((data_parallel, 1),)
    prob_dropout_config = ((data_parallel, model_parallel, 1, 1),)
    soft_max_config = ((data_parallel, model_parallel, 1, 1),)
    soft_max_3d_config = ((data_parallel, model_parallel, 1),)
    expand_dims_config = ((data_parallel, 1, 1),)
    layernorm_config = ((data_parallel, 1),)
    encode_add_config = ((data_parallel, 1), (data_parallel, 1))
    encode_add_3d_config = ((data_parallel, 1, 1), (data_parallel, 1, 1))
    act_config = ((data_parallel, model_parallel),)

    for i in range(layer_num):
        net.encoder.blocks[i].attention.projection.matmul.shard(dp_mp_mp_1)
        net.encoder.blocks[i].attention.projection.bias_add.shard(bias_add_config_1)
        net.encoder.blocks[i].attention.dense1.matmul.shard(dp_1_mp_1)
        net.encoder.blocks[i].attention.dense1.bias_add.shard(bias_add_config_2)
        net.encoder.blocks[i].attention.dense2.matmul.shard(dp_1_mp_1)
        net.encoder.blocks[i].attention.dense2.bias_add.shard(bias_add_config_2)
        net.encoder.blocks[i].attention.dense3.matmul.shard(dp_1_mp_1)
        net.encoder.blocks[i].attention.dense3.bias_add.shard(bias_add_config_2)
        net.encoder.blocks[i].attention.transpose.shard(transpose_dp_1_mp_1)
        net.encoder.blocks[i].attention.merger_head_transpose.shard(transpose_dp_mp_1_1)
        net.encoder.blocks[i].attention.batch_matmul.shard(batch_matmul_config)
        net.encoder.blocks[i].attention.real_div.shard(real_div_config)
        net.encoder.blocks[i].attention.sub.shard(sub_config)
        net.encoder.blocks[i].attention.mul.shard(mul_config)
        net.encoder.blocks[i].attention.add.shard(add_config)
        net.encoder.blocks[i].attention.dropout.dropout.shard(drop_out_config)
        net.encoder.blocks[i].attention.prob_dropout.dropout.shard(prob_dropout_config)
        net.encoder.blocks[i].attention.softmax.softmax.shard(soft_max_config)
        net.encoder.blocks[i].attention.softmax_3d.softmax.shard(soft_max_3d_config)
        net.encoder.blocks[i].attention.expand_dims.shard(expand_dims_config)
        net.encoder.blocks[i].output.mapping.matmul.shard(dp_1_1_mp)
        net.encoder.blocks[i].output.mapping.bias_add.shard(bias_add_config_2)
        net.encoder.blocks[i].output.mapping.activation.gelu.shard(act_config)
        net.encoder.blocks[i].output.projection.matmul.shard(dp_mp_mp_1)
        net.encoder.blocks[i].output.projection.bias_add.shard(bias_add_config_1)
        net.encoder.blocks[i].output.dropout.dropout.shard(drop_out_config)
        net.encoder.blocks[i].output.dropout_3d.dropout.shard(dp_1_1)
        net.encoder.blocks[i].output.dropout_4d.dropout.shard(dp_1_1_1)
        net.encoder.blocks[i].layernorm1.shard(layernorm_config)
        net.encoder.blocks[i].layernorm2.shard(layernorm_config)
        net.encoder.blocks[i].add.shard(encode_add_config)
        net.encoder.blocks[i].add_3d.shard(encode_add_3d_config)
        net.encoder.blocks[i].attention.dense1.matmul.recompute()
        net.encoder.blocks[i].attention.dense2.matmul.set_device("CPU")


def run_transformer_model():
    encoder_layer_num = 2
    net = Transformer(encoder_layers=encoder_layer_num,
                      decoder_layers=0,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64)

    parallel_shard(net, layer_num=encoder_layer_num, data_parallel=4, model_parallel=2)

    encoder_input_value = Tensor(np.ones((8, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((8, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((8, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((8, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((8, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_semi():
    """
    Feature: Test Transformer.
    Description: 3-dim input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    context.set_context(save_graphs=True)
    run_transformer_model()
