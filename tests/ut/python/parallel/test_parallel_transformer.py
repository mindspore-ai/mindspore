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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.context import set_auto_parallel_context, ParallelMode
from mindspore.ops import composite as C
from mindspore.nn.parallel import TransformerEncoder, TransformerDecoder, Transformer, TransformerParallelConfig,\
    VocabEmbedding
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


def test_transformer_model():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2, x3, x4, x5):
            predict, _, _ = self.network(x1, x2, x3, x4, x5)
            return self.loss(predict)

    config = TransformerParallelConfig(dp=1, mp=8)
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = Transformer(encoder_layers=1,
                      decoder_layers=2,
                      hidden_size=64,
                      num_heads=8,
                      ffn_hidden_size=64,
                      src_seq_length=20,
                      tgt_seq_length=20,
                      parallel_config=config)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 1, 20, 20)), mstype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 1, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 1, 10, 20)), mstype.float16)
    net = NetWithLoss(net)

    dataset = Dataset(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask,
                      memory_mask)

    model = Model(net)

    model.train(1, dataset, dataset_sink_mode=False)


def test_encoder():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2):
            predict, _ = self.network(x1, x2)
            return self.loss(predict)

    config = TransformerParallelConfig(dp=1, mp=8)
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = TransformerEncoder(num_layers=2,
                             hidden_size=8,
                             ffn_hidden_size=64,
                             seq_length=16,
                             num_heads=8,
                             parallel_config=config)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), mstype.float32)
    encoder_input_mask = Tensor(np.ones((2, 1, 16, 16)), mstype.float16)

    net = NetWithLoss(net)

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

    config = TransformerParallelConfig(dp=1, mp=8)
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = TransformerDecoder(num_layers=1,
                             hidden_size=16,
                             ffn_hidden_size=8,
                             num_heads=8,
                             seq_length=10,
                             parallel_config=config)

    encoder_input_value = Tensor(np.ones((2, 20, 16)), mstype.float32)
    decoder_input_value = Tensor(np.ones((2, 10, 16)), mstype.float32)
    decoder_input_mask = Tensor(np.ones((2, 1, 10, 10)), mstype.float16)
    memory_mask = Tensor(np.ones((2, 1, 10, 20)), mstype.float16)

    net = NetWithLoss(net)

    dataset = Dataset(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_vocabembedding_dp_true():
    config = TransformerParallelConfig(dp=1, mp=8)
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1):
            predict, _ = self.network(x1)
            return self.loss(predict)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1):
            return grad_all(self.network)(x1)

    net = VocabEmbedding(vocab_size=100, embedding_size=16, parallel_config=config)
    net = NetWithLoss(net)
    encoder_input_value = Tensor(np.ones((2, 64)), mstype.int32)
    dataset = Dataset(encoder_input_value)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)


def test_vocabembedding_dp_false():
    config = TransformerParallelConfig(dp=1, mp=8, vocab_emb_dp=False)
    set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)

    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1):
            predict, _ = self.network(x1)
            return self.loss(predict)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1):
            return grad_all(self.network)(x1)

    net = VocabEmbedding(vocab_size=160, embedding_size=16, parallel_config=config)
    net = NetWithLoss(net)
    encoder_input_value = Tensor(np.ones((2, 64)), mstype.int32)
    dataset = Dataset(encoder_input_value)

    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)
