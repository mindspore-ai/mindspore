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

import pytest
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.context import set_auto_parallel_context, ParallelMode
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._transformer import Transformer, TransformerOpParallelConfig, MoEConfig, CrossEntropyLoss
from mindspore.parallel import set_algo_parameters
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell, _VirtualDatasetCell
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


config = TransformerOpParallelConfig(data_parallel=2, model_parallel=8, vocab_emb_dp=False)
moe_config = MoEConfig(expert_num=4, num_experts_chosen=3)


class NetWithLossFiveInputs(nn.Cell):
    def __init__(self, network):
        super(NetWithLossFiveInputs, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x1, x2, x3, x4, x5):
        predict, _, _, _ = self.network(x1, x2, x3, x4, x5)
        return self.loss(predict)


class NetWithLossMoe(nn.Cell):
    def __init__(self, network):
        super(NetWithLossMoe, self).__init__()
        self.network = network
        self.add = P.Add().shard(((), ()))
        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((1, 1),))

    def construct(self, x1, x2, x3, x4, x5):
        predict, _, _, moe_loss = self.network(x1, x2, x3, x4, x5)
        predict = P.Reshape()(predict, (-1, 1))
        predict = self.reduce_mean(predict)
        return self.add(predict, moe_loss)


def run_transformer_model():
    net = Transformer(encoder_layers=1,
                      decoder_layers=1,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      moe_config=moe_config,
                      parallel_config=config)
    net = _VirtualDatasetCell(net)
    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossMoe(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_semi():
    """
    Feature: Test Transformer+MoE, with All2All enabled.
    Description: 3-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, global_rank=0,
                              full_batch=True, enable_alltoall=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    run_transformer_model()


def test_transformer_model_sp():
    """
    Feature: Test Transformer+MoE, with All2All enabled and sharding propagation.
    Description: 3-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, global_rank=0, search_mode="sharding_propagation",
                              full_batch=True, enable_alltoall=True,
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_transformer_model()


def run_transformer_model_2d():
    net = Transformer(encoder_layers=1,
                      decoder_layers=1,
                      batch_size=2,
                      src_seq_length=20,
                      tgt_seq_length=10,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      moe_config=moe_config,
                      parallel_config=config)
    net = _VirtualDatasetCell(net)

    encoder_input_value = Tensor(np.ones((40, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((20, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    net = NetWithLossFiveInputs(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_transformer_model_2d_semi():
    """
    Feature: Test Transformer+MoE, with All2All enabled.
    Description: 2-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, global_rank=0,
                              full_batch=True, enable_alltoall=True,
                              parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    run_transformer_model_2d()


def test_transformer_model_2d_sp():
    """
    Feature: Test Transformer+MoE, with All2All enabled and sharding propagation.
    Description: 2-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, global_rank=0, search_mode="sharding_propagation",
                              full_batch=True, enable_alltoall=True,
                              parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    run_transformer_model_2d()


class TransformerNet(nn.Cell):
    """Transformer with loss"""
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
                                   moe_config=moe_config,
                                   parallel_config=parallel_config)
        self.loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)

    def construct(self, x1, x2, x3, x4, x5, y, mask):
        predict, _, _ = self.network(x1, x2, x3, x4, x5)
        predict = P.Reshape()(predict, (-1, F.shape(predict)[-1]))
        return self.loss(predict, y, mask)


def moe_with_loss_plus_mutiparallel(local_parallel_config):
    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
    label = Tensor(np.ones((20,)), mstype.int32)
    input_mask = Tensor(np.ones((20,)), mstype.float32)

    net = TransformerNet(en_layer=1, de_layer=1, parallel_config=local_parallel_config)
    net = _VirtualDatasetCell(net)
    params = net.trainable_params()
    optimizer = AdamWeightDecay(params)
    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask, label, input_mask)
    net_with_grad = TrainOneStepCell(net, optimizer=optimizer)
    model = Model(net_with_grad)

    model.train(1, dataset, dataset_sink_mode=False)


def test_moe_expert_parallel1():
    """
    Feature: Test Transformer+MoE for data_parallel plus expert_parallel, with All2All enabled.
    Description: 3-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, enable_alltoall=True,
                              full_batch=True, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    local_p_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=4, expert_parallel=2)
    moe_with_loss_plus_mutiparallel(local_p_config)


def test_moe_expert_parallel2():
    """
    Feature: Test Transformer+MoE for data_parallel plus expert_parallel, with All2All enabled.
    Description: 3-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, enable_alltoall=True,
                              full_batch=True, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    local_p_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=8, expert_parallel=1)
    moe_with_loss_plus_mutiparallel(local_p_config)


def test_moe_expert_parallel3():
    """
    Feature: Test Transformer+MoE for data_parallel plus expert_parallel, with All2All enabled
             and sharding propagation.
    Description: 3-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, enable_alltoall=True, search_mode="sharding_propagation",
                              full_batch=True, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    local_p_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=4, expert_parallel=2)
    moe_with_loss_plus_mutiparallel(local_p_config)


def test_moe_expert_parallel4():
    """
    Feature: Test Transformer+MoE for data_parallel plus expert_parallel, with All2All enabled
             and sharding propagation.
    Description: 3-dim input.
    Expectation: Successful graph compilation with All2All included.
    """
    set_auto_parallel_context(device_num=16, enable_alltoall=True, search_mode="sharding_propagation",
                              full_batch=True, global_rank=0, parallel_mode=ParallelMode.AUTO_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    local_p_config = TransformerOpParallelConfig(data_parallel=2, model_parallel=8, expert_parallel=1)
    moe_with_loss_plus_mutiparallel(local_p_config)


def test_moe_expert_parallel_exception1():
    """
    Feature: Test Transformer+MoE for data_parallel plus expert_parallel, with All2All enabled.
    Description: 3-dim input.
    Expectation: Raise ValueError.
    """
    local_p_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, expert_parallel=2)
    with pytest.raises(ValueError):
        moe_with_loss_plus_mutiparallel(local_p_config)


def test_moe_expert_parallel_exception2():
    """
    Feature: Test Transformer+MoE for data_parallel plus expert_parallel, with All2All enabled.
    Description: data_parallel*model_parallel*expert_parallel > device_num
    Expectation: Raise ValueError.
    """
    local_p_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8, expert_parallel=4)
    with pytest.raises(ValueError):
        moe_with_loss_plus_mutiparallel(local_p_config)
