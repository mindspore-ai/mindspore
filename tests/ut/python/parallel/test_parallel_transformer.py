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

import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.context import set_auto_parallel_context, ParallelMode
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.ops as P
from mindspore.parallel.nn import TransformerEncoder, TransformerDecoder, Transformer, TransformerOpParallelConfig, \
    VocabEmbedding, CrossEntropyLoss, OpParallelConfig, EmbeddingOpParallelConfig, FixedSparseAttention
from mindspore.nn import Dense as Linear
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, TrainOneStepCell
from mindspore.nn.wrap.loss_scale import _TrainPipelineWithLossScaleCell
from mindspore.train import Model
from mindspore.parallel import set_algo_parameters
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


class TransformerNet(nn.Cell):
    def __init__(self, en_layer, de_layer, parallel_config):
        super(TransformerNet, self).__init__()
        self.embedding = VocabEmbedding(vocab_size=240, embedding_size=20,
                                        parallel_config=config.embedding_dp_mp_config)
        self.network = Transformer(encoder_layers=en_layer,
                                   decoder_layers=de_layer,
                                   batch_size=2,
                                   src_seq_length=20,
                                   tgt_seq_length=10,
                                   hidden_size=64,
                                   num_heads=8,
                                   ffn_hidden_size=64,
                                   parallel_config=parallel_config)
        self.head = Linear(in_channels=64, out_channels=200)
        self.loss = CrossEntropyLoss(parallel_config=config.dp_mp_config)

    def construct(self, x1, x2, x3, x4, x5, y, mask):
        predict, _, _ = self.network(x1, x2, x3, x4, x5)
        predict = P.Reshape()(predict, (-1, F.shape(predict)[-1]))
        return self.loss(predict, y, mask)

config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, vocab_emb_dp=False)
pipeline_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, pipeline_stage=4,
                                              micro_batch_num=4, vocab_emb_dp=False)


class NetWithLossFiveInputs(nn.Cell):
    def __init__(self, network):
        super(NetWithLossFiveInputs, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x1, x2, x3, x4, x5):
        predict, _, _ = self.network(x1, x2, x3, x4, x5)
        return self.loss(predict)


def run_total_transformer_model_head(e_layer,
                                     d_layer,
                                     arg_parallel_config,
                                     mode=ParallelMode.SEMI_AUTO_PARALLEL):
    dp = arg_parallel_config.data_parallel
    mp = arg_parallel_config.model_parallel
    pp = arg_parallel_config.pipeline_stage
    if dp * mp * pp != 1:
        set_auto_parallel_context(device_num=8,
                                  full_batch=True,
                                  global_rank=0, parallel_mode=mode)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    seq = 20
    if d_layer > 0:
        seq = 10
    label = Tensor(np.ones((2 * seq,)), mstype.int32)
    input_mask = Tensor(np.ones((2 * seq,)), mstype.float32)
    net = TransformerNet(en_layer=e_layer, de_layer=d_layer, parallel_config=arg_parallel_config)
    net = _VirtualDatasetCell(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask, label, input_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model():
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = Transformer(encoder_layers=1,
                      decoder_layers=2,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      parallel_config=config)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    net = _VirtualDatasetCell(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_2d_inputs():
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = Transformer(encoder_layers=1,
                      decoder_layers=2,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      parallel_config=config)

    encoder_input_value = Tensor(np.ones((40, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((20, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    net = _VirtualDatasetCell(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_int64_inputs():
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = Transformer(encoder_layers=1,
                      decoder_layers=2,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      parallel_config=config)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.int64)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    net = _VirtualDatasetCell(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    with pytest.raises(TypeError):
        model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_head_parallel_only_encoder():
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    run_total_transformer_model_head(e_layer=2, d_layer=0, arg_parallel_config=local_config)


def test_transformer_model_head_parallel():
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    run_total_transformer_model_head(e_layer=1, d_layer=1, arg_parallel_config=local_config)


def test_transformer_model_head_parallel_decoder():
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    with pytest.raises(ValueError):
        run_total_transformer_model_head(e_layer=0, d_layer=1, arg_parallel_config=local_config)


def test_transformer_model_head_stand_alone():
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=1)
    run_total_transformer_model_head(e_layer=2, d_layer=2, arg_parallel_config=local_config)


def test_transformer_model_auto_parallel_no_support():
    local_config = TransformerOpParallelConfig(data_parallel=8, model_parallel=1)
    with pytest.raises(RuntimeError):
        run_total_transformer_model_head(e_layer=2, d_layer=2, arg_parallel_config=local_config,
                                         mode=ParallelMode.AUTO_PARALLEL)


def test_pipeline_single_transformer():
    set_auto_parallel_context(device_num=32,
                              full_batch=True,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

    net = Transformer(batch_size=4 // pipeline_config.micro_batch_num,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      encoder_layers=2,
                      decoder_layers=2,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      parallel_config=pipeline_config)

    encoder_input_value = Tensor(np.ones((4, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((4, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((4, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((4, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((4, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    net = PipelineCell(net, pipeline_config.micro_batch_num)
    net = _VirtualDatasetCell(net)
    params = net.infer_param_pipeline_stage()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=1024, scale_factor=2, scale_window=1000)
    net_with_grad = _TrainPipelineWithLossScaleCell(net, optimizer=optimizer,
                                                    scale_sense=update_cell)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_wrong_head():
    set_auto_parallel_context(device_num=32,
                              full_batch=True,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    error_test_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, vocab_emb_dp=False)
    with pytest.raises(ValueError):
        net = Transformer(batch_size=4,
                          src_seq_length=20,
                          tgt_seq_length=10,
                          encoder_layers=2,
                          decoder_layers=2,
                          hidden_size=64,
                          num_heads=7,
                          ffn_hidden_size=64,
                          parallel_config=error_test_config)

    with pytest.raises(ValueError):
        net = Transformer(batch_size=4,
                          src_seq_length=20,
                          tgt_seq_length=10,
                          encoder_layers=2,
                          decoder_layers=2,
                          hidden_size=63,
                          num_heads=7,
                          ffn_hidden_size=64,
                          parallel_config=error_test_config)
        del net


def test_transformer_wrong_dp_no_error():
    set_auto_parallel_context(device_num=32, full_batch=False, parallel_mode=ParallelMode.DATA_PARALLEL,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0)
    check_config = TransformerOpParallelConfig(data_parallel=8, model_parallel=1, vocab_emb_dp=False)
    net = Transformer(batch_size=4, src_seq_length=20, tgt_seq_length=10, encoder_layers=2,
                      decoder_layers=2, hidden_size=64, num_heads=2, ffn_hidden_size=64,
                      parallel_config=check_config)
    del net


def test_transformer_wrong_semi_auto_dp_error():
    set_auto_parallel_context(device_num=32, full_batch=False, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0)
    check_config = TransformerOpParallelConfig(data_parallel=16, model_parallel=1, vocab_emb_dp=False)
    with pytest.raises(ValueError):
        net = Transformer(batch_size=4, src_seq_length=20, tgt_seq_length=10, encoder_layers=2,
                          decoder_layers=2, hidden_size=64, num_heads=2, ffn_hidden_size=64,
                          parallel_config=check_config)
        del net


def test_encoder():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2):
            predict, _ = self.network(x1, x2)
            return self.loss(predict)

    set_auto_parallel_context(device_num=8,
                              full_batch=True,
                              global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = TransformerEncoder(num_layers=2,
                             batch_size=2,
                             seq_length=16,
                             hidden_size=8,
                             ffn_hidden_size=64,
                             num_heads=8,
                             parallel_config=config)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), mstype.float16)

    net = NetWithLoss(net)

    net = _VirtualDatasetCell(net)

    dataset = Dataset(encoder_input_value, encoder_input_mask)

    model = Model(net)

    model.train(1, dataset, dataset_sink_mode=False)


def test_decoder():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2, x3, x4):
            predict, _, _ = self.network(x1, x2, x3, x4)
            return self.loss(predict)

    set_auto_parallel_context(device_num=8,
                              full_batch=True,
                              global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = TransformerDecoder(num_layers=1,
                             batch_size=8,
                             hidden_size=16,
                             ffn_hidden_size=8,
                             num_heads=8,
                             src_seq_length=20,
                             tgt_seq_length=10,
                             parallel_config=config)

    encoder_input_value = Tensor(np.ones((8, 20, 16)), mstype.float32)
    decoder_input_value = Tensor(np.ones((8, 10, 16)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((8, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((8, 10, 20)), mstype.float16)

    net = NetWithLoss(net)

    net = _VirtualDatasetCell(net)

    dataset = Dataset(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_vocabembedding_dp_true():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1):
            predict, _ = self.network(x1)
            return self.loss(predict)

    net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=config.embedding_dp_mp_config)
    net = NetWithLoss(net)
    net = _VirtualDatasetCell(net)
    encoder_input_value = Tensor(np.ones((2, 64)), mstype.int32)
    dataset = Dataset(encoder_input_value)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_vocabembedding_dp_false():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1):
            predict, _ = self.network(x1)
            return self.loss(predict)

    net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=config.embedding_dp_mp_config)
    net = NetWithLoss(net)
    net = _VirtualDatasetCell(net)
    encoder_input_value = Tensor(np.ones((2, 64)), mstype.int32)
    dataset = Dataset(encoder_input_value)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_sparse_attention_parallel_mp():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(fully_use_devices=False)
    sparse_attention_config = OpParallelConfig(model_parallel=8)
    net = FixedSparseAttention(batch_size=16,
                               seq_length=1024,
                               size_per_head=64,
                               num_heads=8,
                               block_size=64,
                               parallel_config=sparse_attention_config)
    q = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    k = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    v = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), mstype.float32)
    dataset = Dataset(q, k, v, mask)
    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_sparse_attention_parallel_mix():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(fully_use_devices=False)
    sparse_attention_config = OpParallelConfig(data_parallel=2, model_parallel=4)
    net = FixedSparseAttention(batch_size=16,
                               seq_length=1024,
                               size_per_head=64,
                               num_heads=8,
                               block_size=64,
                               parallel_config=sparse_attention_config)
    q = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    k = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    v = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), mstype.float32)
    dataset = Dataset(q, k, v, mask)
    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_sparse_attention_parallel_mix1():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(fully_use_devices=False)
    sparse_attention_config = OpParallelConfig(data_parallel=4, model_parallel=2)
    net = FixedSparseAttention(batch_size=16,
                               seq_length=1024,
                               size_per_head=64,
                               num_heads=8,
                               block_size=64,
                               parallel_config=sparse_attention_config)
    q = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    k = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    v = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), mstype.float32)
    dataset = Dataset(q, k, v, mask)
    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_sparse_attention_parallel_dp():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(fully_use_devices=False)
    sparse_attention_config = OpParallelConfig(data_parallel=8, model_parallel=1)
    net = FixedSparseAttention(batch_size=16,
                               seq_length=1024,
                               size_per_head=64,
                               num_heads=8,
                               block_size=64,
                               parallel_config=sparse_attention_config)
    net = _VirtualDatasetCell(net)
    q = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    k = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    v = Tensor(np.ones((2, 1024, 512)), mstype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), mstype.float32)
    dataset = Dataset(q, k, v, mask)
    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_parallel_cross_entroy_loss_semi_auto_parallel():
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)

    class NetWithLoss(nn.Cell):
        def __init__(self, network, config_setting):
            super(NetWithLoss, self).__init__()
            self.loss = CrossEntropyLoss(config_setting)
            self.network = network

        def construct(self, x1, x2, x3):
            predict, _ = self.network(x1)
            predict = P.Reshape()(predict, (-1, 16))
            return self.loss(predict, x2, x3)

    net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=config.embedding_dp_mp_config)
    net = NetWithLoss(net, config.dp_mp_config)
    net = _VirtualDatasetCell(net)
    embed_ids = Tensor(np.ones((2, 64)), mstype.int32)
    labels = Tensor(np.ones((2 * 64,)), mstype.int32)
    input_mask = Tensor(np.ones((2 * 64,)), mstype.float32)
    dataset = Dataset(embed_ids, labels, input_mask)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_args():

    with pytest.raises(TypeError):
        Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                    tgt_seq_length=20, decoder_layers="aa")

    with pytest.raises(TypeError):
        Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                    tgt_seq_length="a")

    with pytest.raises(TypeError):
        Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                    tgt_seq_length=20, softmax_compute_type=mstype.int64)

    with pytest.raises(TypeError):
        Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                    tgt_seq_length=20, layernorm_compute_type=mstype.int64)

    with pytest.raises(TypeError):
        Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                    tgt_seq_length=20, param_init_type=mstype.int64)

    with pytest.raises(TypeError):
        Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                    tgt_seq_length=20, hidden_dropout_rate=mstype.int64)

    Transformer(hidden_size=10, batch_size=2, ffn_hidden_size=20, src_seq_length=10,
                tgt_seq_length=20, softmax_compute_type=mstype.float16)


def test_transformer_parallel_config():
    parallel_test_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=3)

    with pytest.raises(TypeError):
        parallel_test_config.data_parallel = False

    with pytest.raises(ValueError):
        parallel_test_config.data_parallel = 0

    with pytest.raises(TypeError):
        parallel_test_config.model_parallel = False

    with pytest.raises(ValueError):
        parallel_test_config.model_parallel = 0

    with pytest.raises(TypeError):
        parallel_test_config.pipeline_stage = False

    with pytest.raises(ValueError):
        parallel_test_config.pipeline_stage = 0

    with pytest.raises(TypeError):
        parallel_test_config.micro_batch_num = False

    with pytest.raises(ValueError):
        parallel_test_config.micro_batch_num = 0

    with pytest.raises(TypeError):
        parallel_test_config.gradient_aggregation_group = False

    with pytest.raises(ValueError):
        parallel_test_config.gradient_aggregation_group = 0

    with pytest.raises(TypeError):
        parallel_test_config.recompute = 1

    parallel_test_config.recompute = False

    assert not parallel_test_config.recompute


def test_parallel_config():
    parallel_test_config = OpParallelConfig(data_parallel=1, model_parallel=3)

    with pytest.raises(ValueError):
        parallel_test_config.data_parallel = 0

    with pytest.raises(TypeError):
        parallel_test_config.model_parallel = False

    with pytest.raises(ValueError):
        parallel_test_config.model_parallel = 0

    assert parallel_test_config.model_parallel == 3


def test_embedding_parallel_config():
    parallel_test_config = EmbeddingOpParallelConfig(data_parallel=1, model_parallel=3, vocab_emb_dp=False)

    with pytest.raises(ValueError):
        parallel_test_config.data_parallel = 0

    with pytest.raises(TypeError):
        parallel_test_config.model_parallel = False

    with pytest.raises(ValueError):
        parallel_test_config.model_parallel = 0

    with pytest.raises(TypeError):
        parallel_test_config.vocab_emb_dp = 0

    assert not parallel_test_config.vocab_emb_dp
