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
import os
import glob

import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.context import set_auto_parallel_context, ParallelMode
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.ops as P
from mindspore.parallel._transformer import TransformerEncoder, TransformerDecoder, Transformer, TransformerOpParallelConfig, \
    VocabEmbedding, CrossEntropyLoss, OpParallelConfig, EmbeddingOpParallelConfig, FixedSparseAttention,\
    TransformerRecomputeConfig
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, TrainOneStepCell
from mindspore.nn.wrap.loss_scale import _TrainPipelineWithLossScaleCell
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train import Model
from mindspore.parallel import set_algo_parameters
from parallel.utils.utils import BasicValidator
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
        self.network = Transformer(encoder_layers=en_layer,
                                   decoder_layers=de_layer,
                                   batch_size=2,
                                   src_seq_length=20,
                                   tgt_seq_length=10,
                                   hidden_size=64,
                                   num_heads=8,
                                   ffn_hidden_size=64,
                                   parallel_config=parallel_config)
        self.loss = CrossEntropyLoss(parallel_config=config.dp_mp_config)

    def construct(self, x1, x2, x3, x4, x5, y, mask):
        predict, _, _ = self.network(x1, x2, x3, x4, x5)
        predict = P.Reshape()(predict, (-1, F.shape(predict)[-1]))
        return self.loss(predict, y, mask)


class TransformerEncoderNet(nn.Cell):
    def __init__(self, batch_size, en_layer, de_layer, parallel_config):
        super(TransformerEncoderNet, self).__init__()
        self.embedding = VocabEmbedding(vocab_size=240, embedding_size=64,
                                        parallel_config=parallel_config.embedding_dp_mp_config)
        self.network = Transformer(encoder_layers=en_layer,
                                   decoder_layers=de_layer,
                                   batch_size=batch_size,
                                   src_seq_length=20,
                                   tgt_seq_length=10,
                                   hidden_size=64,
                                   num_heads=8,
                                   ffn_hidden_size=64,
                                   parallel_config=parallel_config)
        self.loss = CrossEntropyLoss(parallel_config=config.dp_mp_config)

    def construct(self, x, encoder_mask, label, input_mask):
        embedded, _ = self.embedding(x)
        logits, _, = self.network(embedded, encoder_mask)
        logits = P.Reshape()(logits, (-1, F.shape(logits)[-1]))
        label = P.Reshape()(label, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        return self.loss(logits, label, input_mask)


config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, vocab_emb_dp=False)
slice_activation_recompute_config = TransformerRecomputeConfig(recompute=True, recompute_slice_activation=True)
parallel_opt_recompute_config = TransformerRecomputeConfig(recompute=True, parallel_optimizer_comm_recompute=True)
slice_activtion_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, vocab_emb_dp=False,
                                                     recompute=slice_activation_recompute_config)
parallel_opt_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, vocab_emb_dp=False,
                                                  recompute=parallel_opt_recompute_config)
pipeline_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=8, pipeline_stage=4,
                                              micro_batch_num=4, vocab_emb_dp=False)


class NetWithLossThreeInputs(nn.Cell):
    def __init__(self, network, config_setting):
        super(NetWithLossThreeInputs, self).__init__()
        self.loss = CrossEntropyLoss(config_setting)
        self.network = network

    def construct(self, x1, x2, x3):
        predict, _ = self.network(x1)
        predict = P.Reshape()(predict, (-1, 16))
        return self.loss(predict, x2, x3)


class NetWithLossFiveInputs(nn.Cell):
    def __init__(self, network):
        super(NetWithLossFiveInputs, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x1, x2, x3, x4, x5):
        predict, _, _ = self.network(x1, x2, x3, x4, x5)
        return self.loss(predict)


def run_network_function(dataset, pipeline_net):
    """
    Feature: Test transformer embedding shared.
    Description: a basic function for test compiling.
    Expectation: success.
    """
    params = pipeline_net.trainable_params()
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(pipeline_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


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


def run_transformer_model():
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


def test_transformer_model_semi():
    """
    Feature: Test Transformer.
    Description: 3-dim input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    run_transformer_model()


def test_transformer_model_sp():
    """
    Feature: Test Transformer with sharding propagation.
    Description: 3-dim input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True, search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_transformer_model()


def run_transformer_model_2d_inputs():
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


def test_transformer_model_2d_semi():
    """
    Feature: Test Transformer.
    Description: 2-dim input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    run_transformer_model_2d_inputs()


def test_transformer_model_2d_sp():
    """
    Feature: Test Transformer with sharding propagation.
    Description: 2-dim input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True, search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_transformer_model_2d_inputs()


class TestTransformerEmbeddingHead(BasicValidator):
    def virtual_assign_add_from_ir(self, pattern, target_count):
        """
        This function will check the assign aa count with the golden one.
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

    def run_pipeline_with_embedding(self):
        bs = 16
        pp = 2
        context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=pp,
                                          full_batch=True,
                                          enable_parallel_optimizer=True)
        cf = TransformerOpParallelConfig(data_parallel=1, model_parallel=4, pipeline_stage=pp, vocab_emb_dp=False)
        pipeline_net = TransformerEncoderNet(batch_size=bs // pp,
                                             en_layer=2, de_layer=0, parallel_config=cf)
        pipeline_net.embedding.pipeline_stage = 0
        pipeline_net.network.encoder.blocks[0].pipeline_stage = 0
        pipeline_net.network.encoder.blocks[1].pipeline_stage = 1

        pipeline_cell_net = PipelineCell(pipeline_net, 2)
        encoder_input_value = Tensor(np.ones((bs, 20)), mstype.int32)
        encoder_input_mask = Tensor(np.ones((bs, 20, 20)), mstype.float16)
        label = Tensor(np.ones((bs, 20)), mstype.int32)
        mask = Tensor(np.ones((bs, 20)), mstype.float32)
        dataset = Dataset(encoder_input_value, encoder_input_mask, label, mask)
        run_network_function(dataset, pipeline_cell_net)
        self.virtual_assign_add_from_ir(pattern=r'AssignAdd(', target_count=35)

    def test_pipeline_with_embedding_semi(self):
        """
        Feature: Test Transformer with embedding as shared
        Description: When do pipeline training and applied optimzier shard, the embedding which is model parallel will
                     raise the shape error. This test cast is ensure there is no error raised.
        Expectation: The number of AssignAdd is not as expected.
        """
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        self.run_pipeline_with_embedding()

    def test_pipeline_with_embedding_sp(self):
        """
        Feature: Test Transformer with embedding as shared, using sharding propagation.
        Description: When do pipeline training and applied optimzier shard, the embedding which is model parallel will
                     raise the shape error. This test cast is ensure there is no error raised.
        Expectation: The number of AssignAdd is not as expected.
        """
        context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")
        set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
        self.run_pipeline_with_embedding()


def run_transformer_model_int64_inputs():
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


def test_transformer_model_int64_semi():
    """
    Feature: Test Transformer.
    Description: int64 input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    run_transformer_model_int64_inputs()


def test_transformer_model_int64_sp():
    """
    Feature: Test Transformer with sharding propagation.
    Description: int64 input.
    Expectation: Successful graph compilation.
    """
    set_auto_parallel_context(device_num=8, global_rank=0,
                              full_batch=True, search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_transformer_model_int64_inputs()


def test_transformer_model_head_parallel_only_encoder():
    """
    Feature: Test Transformer.
    Description: only encoder.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    run_total_transformer_model_head(e_layer=2, d_layer=0, arg_parallel_config=local_config)


def test_transformer_model_head_parallel_only_encoder_sp():
    """
    Feature: Test Transformer with sharding propagation.
    Description: only encoder.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    set_auto_parallel_context(search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_total_transformer_model_head(e_layer=2, d_layer=0, arg_parallel_config=local_config,
                                     mode=ParallelMode.AUTO_PARALLEL)


def test_transformer_model_head_parallel():
    """
    Feature: Test Transformer
    Description: model parallel.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    run_total_transformer_model_head(e_layer=1, d_layer=1, arg_parallel_config=local_config)


def test_transformer_model_head_parallel_sp():
    """
    Feature: Test Transformer with sharding propagation.
    Description: 1 encode and 1 decode.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    set_auto_parallel_context(search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_total_transformer_model_head(e_layer=1, d_layer=1, arg_parallel_config=local_config,
                                     mode=ParallelMode.AUTO_PARALLEL)


def test_transformer_model_head_parallel_decoder():
    """
    Feature: Test Transformer
    Description: model parallel.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
    with pytest.raises(ValueError):
        run_total_transformer_model_head(e_layer=0, d_layer=1, arg_parallel_config=local_config)


def test_transformer_model_head_stand_alone():
    """
    Feature: Test Transformer
    Description: stand alone.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=1)
    run_total_transformer_model_head(e_layer=2, d_layer=2, arg_parallel_config=local_config)


def test_transformer_model_auto_parallel_no_support():
    """
    Feature: Test Transformer
    Description: auto parallel no support.
    Expectation: Successful graph compilation.
    """
    local_config = TransformerOpParallelConfig(data_parallel=8, model_parallel=1)
    with pytest.raises(RuntimeError):
        run_total_transformer_model_head(e_layer=2, d_layer=2, arg_parallel_config=local_config,
                                         mode=ParallelMode.AUTO_PARALLEL)


def pipeline_single_transformer(grad_accumulation_shard=False):
    """
    Feature: Gradient Accumulation Shard for Pipeline and Gradient Accumulation
    Description: Test a single transformer model with pipeline parallel with grad_accumulation_shard False
    Expectation: The compile passed
    """
    set_auto_parallel_context(device_num=64,
                              full_batch=True,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0)
    context.set_auto_parallel_context(parallel_optimizer_config=
                                      {"gradient_accumulation_shard": grad_accumulation_shard})

    net = Transformer(batch_size=8 // pipeline_config.micro_batch_num,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      encoder_layers=2,
                      decoder_layers=2,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      parallel_config=pipeline_config)

    encoder_input_value = Tensor(np.ones((8, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((8, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((8, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((8, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((8, 10, 20)), mstype.float16)
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


def test_pipeline_transformer_gradient_shard_true():
    """
    Feature: Gradient Accumulation Shard for Pipeline and Gradient Accumulation
    Description: Test a single transformer model with pipeline parallel with grad_accumulation_shard True
    Expectation: The compile passed
    """
    set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    pipeline_single_transformer(grad_accumulation_shard=True)


def test_pipeline_transformer_gradient_shard_true_sp():
    """
    Feature: Gradient Accumulation Shard for Pipeline and Gradient Accumulation with sharding propagation
    Description: Test a single transformer model with pipeline parallel with grad_accumulation_shard True
    Expectation: The compile passed
    """
    set_auto_parallel_context(search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    _set_multi_subgraphs()
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    pipeline_single_transformer(grad_accumulation_shard=True)


def test_pipeline_transformer_gradient_shard_false():
    """
    Feature: Gradient Accumulation Shard for Pipeline and Gradient Accumulation
    Description: Test a single transformer model with pipeline parallel with grad_accumulation_shard False
    Expectation: The compile passed
    """
    set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    pipeline_single_transformer(grad_accumulation_shard=False)


def test_pipeline_transformer_gradient_shard_false_sp():
    """
    Feature: Gradient Accumulation Shard for Pipeline and Gradient Accumulation with sharding propagation
    Description: Test a single transformer model with pipeline parallel with grad_accumulation_shard False
    Expectation: The compile passed
    """
    set_auto_parallel_context(search_mode="sharding_propagation",
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    _set_multi_subgraphs()
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    pipeline_single_transformer(grad_accumulation_shard=False)


def test_transformer_wrong_head():
    """
    Feature: test transformer api
    Description: Test transformer exception scene
    Expectation: Raise correct error.
    """
    set_auto_parallel_context(device_num=64,
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
    """
    Feature: test transformer api
    Description: Test transformer correct scene
    Expectation: no error.
    """
    set_auto_parallel_context(device_num=64, full_batch=False, parallel_mode=ParallelMode.DATA_PARALLEL,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0)
    check_config = TransformerOpParallelConfig(data_parallel=8, model_parallel=1, vocab_emb_dp=False)
    net = Transformer(batch_size=4, src_seq_length=20, tgt_seq_length=10, encoder_layers=2,
                      decoder_layers=2, hidden_size=64, num_heads=2, ffn_hidden_size=64,
                      parallel_config=check_config)
    del net


def test_transformer_wrong_semi_auto_dp_error():
    """
    Feature: test transformer api
    Description: Test transformer parallel batch with no check
    Expectation: Raise correct error.
    """
    set_auto_parallel_context(device_num=64, full_batch=False, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                              pipeline_stages=pipeline_config.pipeline_stage, global_rank=0)
    check_config = TransformerOpParallelConfig(data_parallel=16, model_parallel=1, vocab_emb_dp=False)
    Transformer(batch_size=4, src_seq_length=20, tgt_seq_length=10, encoder_layers=2,
                decoder_layers=2, hidden_size=64, num_heads=2, ffn_hidden_size=64,
                parallel_config=check_config)


def test_encoder():
    """
    Feature: test transformer api
    Description: Test encoder layers
    Expectation: Compile ok.
    """
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


def test_encoder_recompute_slice():
    """
    Feature: test transformer api
    Description: Test encoder layers with slice recompute activation
    Expectation: Compile ok.
    """
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
                             parallel_config=slice_activtion_config)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), mstype.float16)

    net = NetWithLoss(net)

    net = _VirtualDatasetCell(net)

    dataset = Dataset(encoder_input_value, encoder_input_mask)

    model = Model(net)

    model.train(1, dataset, dataset_sink_mode=False)


def test_decoder():
    """
    Feature: test transformer api
    Description: Test decoder layers
    Expectation: Compile ok.
    """
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
                             parallel_config=parallel_opt_config)

    encoder_input_value = Tensor(np.ones((8, 20, 16)), mstype.float32)
    decoder_input_value = Tensor(np.ones((8, 10, 16)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((8, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((8, 10, 20)), mstype.float16)

    net = NetWithLoss(net)

    net = _VirtualDatasetCell(net)

    dataset = Dataset(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_decoder_parallel_opt_recompute():
    """
    Feature: test transformer api
    Description: Test decoder layers with parallel optimizer recompute
    Expectation: Compile ok.
    """
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
    """
    Feature: test transformer api
    Description: Test vocab embedding
    Expectation: Compile ok.
    """
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
    """
    Feature: test transformer api
    Description: Test vocab embedding
    Expectation: Compile ok.
    """
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
    """
    Feature: test transformer api
    Description: Test sparse attention
    Expectation: Compile ok.
    """
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
    """
    Feature: test transformer api
    Description: Test sparse attention
    Expectation: Compile ok.
    """
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
    """
    Feature: test transformer api
    Description: Test sparse attention
    Expectation: Compile ok.
    """
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
    """
    Feature: test transformer api
    Description: Test sparse attention
    Expectation: Compile ok.
    """
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


def test_transformer_args():
    """
    Feature: test transformer api
    Description: Test transformer args
    Expectation: Assert ok.
    """
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
    """
    Feature: test transformer api
    Description: Test transformer op parallel config
    Expectation: Assert ok.
    """
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

    parallel_test_config.recompute.recompute = False

    assert not parallel_test_config.recompute.recompute


def test_parallel_config():
    """
    Feature: test transformer api
    Description: Test op parallel config
    Expectation: Assert ok.
    """
    parallel_test_config = OpParallelConfig(data_parallel=1, model_parallel=3)

    with pytest.raises(ValueError):
        parallel_test_config.data_parallel = 0

    with pytest.raises(TypeError):
        parallel_test_config.model_parallel = False

    with pytest.raises(ValueError):
        parallel_test_config.model_parallel = 0

    assert parallel_test_config.model_parallel == 3


def test_embedding_parallel_config():
    """
    Feature: test transformer api
    Description: Test embedding parallel config
    Expectation: Assert ok.
    """
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


class TestCrossEntropyLoss(BasicValidator):
    def test_parallel_cross_entropy_loss_auto_parallel(self):
        """
        Feature: Optimizer the memory usage for cross entropy loss
        Description: Test cross entropy loss in auto parallel, except work well as there will be sub graphs with
                 no bprop. This case will cause there is no J in the forward graphs, thus the forward marker will
                 be used for each subgraph. And there should be only one Virtual dataset.
        Expectation: When there are many virtual datasets, or there are no forward operators.
        """
        set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
        net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=config.embedding_dp_mp_config)
        net = NetWithLossThreeInputs(net, config.dp_mp_config)
        embed_ids = Tensor(np.ones((2, 64)), mstype.int32)
        labels = Tensor(np.ones((2 * 64,)), mstype.int32)
        input_mask = Tensor(np.ones((2 * 64,)), mstype.float32)
        dataset = Dataset(embed_ids, labels, input_mask)

        model = Model(net)
        model.train(1, dataset, dataset_sink_mode=False)
        self.validate_pattern_from_ir("= _VirtualDataset", target_count=1, file_name="step_parallel_end")
        self.validate_pattern_from_ir("= forward_op", target_count=3, file_name="step_parallel_end")

    def test_parallel_cross_entropy_loss_semi_auto_parallel_dp1_mp8(self):
        """
        Feature: Optimizer the memory usage for cross entropy loss
        Description: Test cross entropy loss in semi auto parallel, except work well as there will be sub graphs.
                 In the case, there should be less redistribution operator between the context of the subgraph.
        Expectation: When there are many virtual datasets and not expected redistribution operators.
        """
        set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

        dp_mp_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, vocab_emb_dp=False)
        net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=dp_mp_config.embedding_dp_mp_config)
        net = NetWithLossThreeInputs(net, dp_mp_config.dp_mp_config)
        embed_ids = Tensor(np.ones((2, 64)), mstype.int32)
        labels = Tensor(np.ones((2 * 64,)), mstype.int32)
        input_mask = Tensor(np.ones((2 * 64,)), mstype.float32)
        dataset = Dataset(embed_ids, labels, input_mask)

        model = Model(net, optimizer=AdamWeightDecay(net.trainable_params()))
        model.train(1, dataset, dataset_sink_mode=False)
        self.validate_pattern_from_ir("= _VirtualDataset", target_count=1, file_name="step_parallel_end")
        self.validate_pattern_from_ir("= _redistribution_op", target_count=4, file_name="step_parallel_end")

    def test_parallel_cross_entropy_loss_semi_auto_parallel_dp2_mp4(self):
        """
        Feature: Optimizer the memory usage for cross entropy loss
        Description: Test cross entropy loss in semi auto parallel, except work well as there will be sub graphs.
                 In the case, there should be some redistribution operators after the final subgraph, as the
                 redistributionPreNode should search the sub graphs and skip an AllReduce produced by model parallel.
        Expectation: When there are many virtual datasets and not expected redistribution operators.
        """
        set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

        dp_mp_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=4, vocab_emb_dp=False)
        net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=dp_mp_config.embedding_dp_mp_config)
        net = NetWithLossThreeInputs(net, dp_mp_config.dp_mp_config)
        embed_ids = Tensor(np.ones((2, 64)), mstype.int32)
        labels = Tensor(np.ones((2 * 64,)), mstype.int32)
        input_mask = Tensor(np.ones((2 * 64,)), mstype.float32)
        dataset = Dataset(embed_ids, labels, input_mask)

        model = Model(net, optimizer=AdamWeightDecay(net.trainable_params()))
        model.train(1, dataset, dataset_sink_mode=False)
        self.validate_pattern_from_ir("= _VirtualDataset", target_count=1, file_name="step_parallel_end")
        self.validate_pattern_from_ir("= _redistribution_op", target_count=14, file_name="step_parallel_end")

    def test_parallel_cross_entropy_loss_semi_auto_parallel_dp2_mp4_pipeline_global_rank0(self):
        """
        Feature: Optimizer the memory usage for cross entropy loss
        Description: Test cross entropy loss in semi auto parallel, except work well as there will be sub graphs.
                 In the case, test the pipeline training when the loss used.
        Expectation: When there are many virtual datasets and not expected redistribution operators.
        """
        set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                  pipeline_stages=2)

        dp_mp_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=2, vocab_emb_dp=False)
        net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=dp_mp_config.embedding_dp_mp_config)
        net = NetWithLossThreeInputs(net, dp_mp_config.dp_mp_config)
        net.network.pipeline_stage = 0
        net.network.loss = 1
        embed_ids = Tensor(np.ones((2, 64)), mstype.int32)
        labels = Tensor(np.ones((2 * 64,)), mstype.int32)
        input_mask = Tensor(np.ones((2 * 64,)), mstype.float32)
        dataset = Dataset(embed_ids, labels, input_mask)

        model = Model(net, optimizer=AdamWeightDecay(net.trainable_params()))
        model.train(1, dataset, dataset_sink_mode=False)
        self.validate_pattern_from_ir("= _VirtualDataset", target_count=1, file_name="step_parallel_end")
        self.validate_pattern_from_ir("= _redistribution_op", target_count=19, file_name="step_parallel_end")

    def test_parallel_cross_entropy_loss_semi_auto_parallel_dp2_mp4_pipeline_global_rank4(self):
        """
        Feature: Optimizer the memory usage for cross entropy loss
        Description: Test cross entropy loss in semi auto parallel, except work well as there will be sub graphs.
                 In the case, test the pipeline training when the loss used in the last stage.
        Expectation: When there are many virtual datasets and not expected redistribution operators.
        """
        set_auto_parallel_context(device_num=8, global_rank=4, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                  pipeline_stages=2)

        dp_mp_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=2, vocab_emb_dp=False)
        net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=dp_mp_config.embedding_dp_mp_config)
        net = NetWithLossThreeInputs(net, dp_mp_config.dp_mp_config)
        net.network.pipeline_stage = 0
        net.network.loss = 1
        embed_ids = Tensor(np.ones((2, 64)), mstype.int32)
        labels = Tensor(np.ones((2 * 64,)), mstype.int32)
        input_mask = Tensor(np.ones((2 * 64,)), mstype.float32)
        dataset = Dataset(embed_ids, labels, input_mask)

        model = Model(net, optimizer=AdamWeightDecay(net.trainable_params()))
        model.train(1, dataset, dataset_sink_mode=False)
        self.validate_pattern_from_ir("= _VirtualDataset", target_count=1, file_name="step_parallel_end")
        self.validate_pattern_from_ir("= _redistribution_op", target_count=19, file_name="step_parallel_end")
