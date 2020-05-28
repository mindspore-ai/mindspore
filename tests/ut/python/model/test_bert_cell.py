# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test bert of graph compile """
import functools
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore.ops import functional as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore.model_zoo.Bert_NEZHA import BertPretrainingLoss, GetNextSentenceOutput
from mindspore.model_zoo.Bert_NEZHA.bert_for_pre_training import clip_grad
from mindspore.model_zoo.Bert_NEZHA.bert_model import BertConfig, \
    EmbeddingLookup, EmbeddingPostprocessor, BertOutput, RelaPosMatrixGenerator, \
    RelaPosEmbeddingsGenerator, SaturateCast, BertAttention, BertSelfAttention, \
    BertEncoderCell, BertTransformer, CreateAttentionMaskFromInputMask, BertModel
from mindspore.nn.layer.basic import Norm
from mindspore.nn.optim import AdamWeightDecay, AdamWeightDecayDynamicLR
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward import \
    pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ....mindspore_test_framework.pipeline.gradient.compile_gradient import \
    pipeline_for_compile_grad_ge_graph_for_case_by_case_config
from ....ops_common import convert


def bert_trans():
    """bert_trans"""
    net = BertTransformer(batch_size=1,
                          hidden_size=768,
                          seq_length=128,
                          num_hidden_layers=1,
                          num_attention_heads=12,
                          intermediate_size=768,
                          attention_probs_dropout_prob=0.1,
                          use_one_hot_embeddings=False,
                          initializer_range=0.02,
                          use_relative_positions=False,
                          hidden_act="gelu",
                          compute_type=mstype.float32,
                          return_all_encoders=True)
    net.set_train()
    return net


def set_train(net):
    net.set_train()
    return net


class NetForAdam(nn.Cell):
    def __init__(self):
        super(NetForAdam, self).__init__()
        self.dense = nn.Dense(64, 10)

    def construct(self, x):
        x = self.dense(x)
        return x


class TrainStepWrapForAdam(nn.Cell):
    """TrainStepWrapForAdam definition"""

    def __init__(self, network):
        super(TrainStepWrapForAdam, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())
        self.optimizer = AdamWeightDecay(self.weights)
        self.hyper_map = C.HyperMap()

    def construct(self, x, sens):
        weights = self.weights
        grads = C.grad_by_list_with_sens(self.network, weights)(x, sens)
        grads = self.hyper_map(F.partial(clip_grad, 1, 1.0), grads)
        return self.optimizer(grads)


class TrainStepWrapForAdamDynamicLr(nn.Cell):
    """TrainStepWrapForAdamDynamicLr definition"""

    def __init__(self, network):
        super(TrainStepWrapForAdamDynamicLr, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())
        self.optimizer = AdamWeightDecayDynamicLR(self.weights, 10)
        self.sens = Tensor(np.ones(shape=(1, 10)).astype(np.float32))

    def construct(self, x):
        weights = self.weights
        grads = C.grad_by_list_with_sens(self.network, weights)(x, self.sens)
        return self.optimizer(grads)


class TempC2Wrap(nn.Cell):
    def __init__(self, op, c1=None, c2=None, ):
        super(TempC2Wrap, self).__init__()
        self.op = op
        self.c1 = c1
        self.c2 = c2
        self.hyper_map = C.HyperMap()

    def construct(self, x1):
        x = self.hyper_map(F.partial(self.op, self.c1, self.c2), x1)
        return x


test_case_cell_ops = [
    ('Norm_keepdims', {
        'block': Norm(keep_dims=True),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1]]}),
    ('SaturateCast', {
        'block': SaturateCast(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('RelaPosMatrixGenerator_0', {
        'block': RelaPosMatrixGenerator(length=128, max_relative_position=16),
        'desc_inputs': [],
        'desc_bprop': [[128, 128]],
        'skip': ['backward']}),
    ('RelaPosEmbeddingsGenerator_0', {
        'block': RelaPosEmbeddingsGenerator(length=128, depth=512,
                                            max_relative_position=16,
                                            initializer_range=0.2),
        'desc_inputs': [],
        'desc_bprop': [[16384, 512]],
        'skip': ['backward']}),
    ('RelaPosEmbeddingsGenerator_1', {
        'block': RelaPosEmbeddingsGenerator(length=128, depth=512,
                                            max_relative_position=16,
                                            initializer_range=0.2,
                                            use_one_hot_embeddings=False),
        'desc_inputs': [],
        'desc_bprop': [[128, 128, 512]],
        'skip': ['backward']}),
    ('RelaPosEmbeddingsGenerator_2', {
        'block': RelaPosEmbeddingsGenerator(length=128, depth=64,
                                            max_relative_position=16,
                                            initializer_range=0.2,
                                            use_one_hot_embeddings=False),
        'desc_inputs': [],
        'desc_bprop': [[128, 128, 64]],
        'skip': ['backward']}),
    ('BertAttention_0', {
        'block': BertAttention(batch_size=64,
                               from_tensor_width=768,
                               to_tensor_width=768,
                               from_seq_length=128,
                               to_seq_length=128,
                               num_attention_heads=12,
                               size_per_head=64,
                               query_act=None,
                               key_act=None,
                               value_act=None,
                               has_attention_mask=True,
                               attention_probs_dropout_prob=0.1,
                               use_one_hot_embeddings=False,
                               initializer_range=0.02,
                               do_return_2d_tensor=True,
                               use_relative_positions=False,
                               compute_type=mstype.float32),
        'desc_inputs': [[64, 128, 768], [64, 128, 768], [64, 128, 128]],
        'desc_bprop': [[8192, 768]]}),
    ('BertAttention_1', {
        'block': BertAttention(batch_size=64,
                               from_tensor_width=768,
                               to_tensor_width=768,
                               from_seq_length=128,
                               to_seq_length=128,
                               num_attention_heads=12,
                               size_per_head=64,
                               query_act=None,
                               key_act=None,
                               value_act=None,
                               has_attention_mask=True,
                               attention_probs_dropout_prob=0.1,
                               use_one_hot_embeddings=False,
                               initializer_range=0.02,
                               do_return_2d_tensor=True,
                               use_relative_positions=True,
                               compute_type=mstype.float32),
        'desc_inputs': [[64, 128, 768], [64, 128, 768], [64, 128, 128]],
        'desc_bprop': [[8192, 768]]}),
    ('BertAttention_2', {
        'block': BertAttention(batch_size=64,
                               from_tensor_width=768,
                               to_tensor_width=768,
                               from_seq_length=128,
                               to_seq_length=128,
                               num_attention_heads=12,
                               size_per_head=64,
                               query_act=None,
                               key_act=None,
                               value_act=None,
                               has_attention_mask=False,
                               attention_probs_dropout_prob=0.1,
                               use_one_hot_embeddings=False,
                               initializer_range=0.02,
                               do_return_2d_tensor=True,
                               use_relative_positions=True,
                               compute_type=mstype.float32),
        'desc_inputs': [[64, 128, 768], [64, 128, 768], [64, 128, 128]],
        'desc_bprop': [[8192, 768]]}),
    ('BertAttention_3', {
        'block': BertAttention(batch_size=64,
                               from_tensor_width=768,
                               to_tensor_width=768,
                               from_seq_length=128,
                               to_seq_length=128,
                               num_attention_heads=12,
                               size_per_head=64,
                               query_act=None,
                               key_act=None,
                               value_act=None,
                               has_attention_mask=True,
                               attention_probs_dropout_prob=0.1,
                               use_one_hot_embeddings=False,
                               initializer_range=0.02,
                               do_return_2d_tensor=False,
                               use_relative_positions=True,
                               compute_type=mstype.float32),
        'desc_inputs': [[64, 128, 768], [64, 128, 768], [64, 128, 128]],
        'desc_bprop': [[8192, 768]]}),
    ('BertOutput', {
        'block': BertOutput(in_channels=768,
                            out_channels=768,
                            initializer_range=0.02,
                            dropout_prob=0.1),
        'desc_inputs': [[8192, 768], [8192, 768]],
        'desc_bprop': [[8192, 768]]}),
    ('BertSelfAttention_0', {
        'block': BertSelfAttention(batch_size=64,
                                   seq_length=128,
                                   hidden_size=768,
                                   num_attention_heads=12,
                                   attention_probs_dropout_prob=0.1,
                                   use_one_hot_embeddings=False,
                                   initializer_range=0.02,
                                   hidden_dropout_prob=0.1,
                                   use_relative_positions=False,
                                   compute_type=mstype.float32),
        'desc_inputs': [[64, 128, 768], [64, 128, 128]],
        'desc_bprop': [[8192, 768]]}),
    ('BertEncoderCell', {
        'block': BertEncoderCell(batch_size=64,
                                 hidden_size=768,
                                 seq_length=128,
                                 num_attention_heads=12,
                                 intermediate_size=768,
                                 attention_probs_dropout_prob=0.02,
                                 use_one_hot_embeddings=False,
                                 initializer_range=0.02,
                                 hidden_dropout_prob=0.1,
                                 use_relative_positions=False,
                                 hidden_act="gelu",
                                 compute_type=mstype.float32),
        'desc_inputs': [[64, 128, 768], [64, 128, 128]],
        'desc_bprop': [[8192, 768]]}),
    ('BertTransformer_0', {
        'block': BertTransformer(batch_size=1,
                                 hidden_size=768,
                                 seq_length=128,
                                 num_hidden_layers=1,
                                 num_attention_heads=12,
                                 intermediate_size=768,
                                 attention_probs_dropout_prob=0.1,
                                 use_one_hot_embeddings=False,
                                 initializer_range=0.02,
                                 use_relative_positions=False,
                                 hidden_act="gelu",
                                 compute_type=mstype.float32,
                                 return_all_encoders=True),
        'desc_inputs': [[1, 128, 768], [1, 128, 128]]}),
    ('BertTransformer_1', {
        'block': BertTransformer(batch_size=64,
                                 hidden_size=768,
                                 seq_length=128,
                                 num_hidden_layers=2,
                                 num_attention_heads=12,
                                 intermediate_size=768,
                                 attention_probs_dropout_prob=0.1,
                                 use_one_hot_embeddings=False,
                                 initializer_range=0.02,
                                 use_relative_positions=True,
                                 hidden_act="gelu",
                                 compute_type=mstype.float32,
                                 return_all_encoders=False),
        'desc_inputs': [[64, 128, 768], [64, 128, 128]]}),
    ('EmbeddingLookup', {
        'block': EmbeddingLookup(vocab_size=32000,
                                 embedding_size=768,
                                 embedding_shape=[1, 128, 768],
                                 use_one_hot_embeddings=False,
                                 initializer_range=0.02),
        'desc_inputs': [Tensor(np.random.rand(128).astype(np.int32))],
        'desc_bprop': [[1, 128, 768], [1, 128, 768]],
        'num_output': 2}),
    ('EmbeddingPostprocessor', {
        'block': EmbeddingPostprocessor(embedding_size=768,
                                        embedding_shape=[1, 128, 768],
                                        use_token_type=True,
                                        token_type_vocab_size=16,
                                        use_one_hot_embeddings=False,
                                        initializer_range=0.02,
                                        max_position_embeddings=512,
                                        dropout_prob=0.1),
        'desc_inputs': [Tensor(np.random.rand(128).astype(np.int32)), [1, 128, 768]],
        'desc_bprop': [[1, 128, 768]]}),
    ('CreateAttentionMaskFromInputMask', {
        'block': CreateAttentionMaskFromInputMask(config=BertConfig(batch_size=1)),
        'desc_inputs': [[128]],
        'desc_bprop': [[1, 128, 128]]}),
    ('BertOutput_0', {
        'block': BertOutput(in_channels=768,
                            out_channels=768,
                            initializer_range=0.02,
                            dropout_prob=0.1),
        'desc_inputs': [[1, 768], [1, 768]],
        'desc_bprop': [[1, 768]]}),
    ('BertTransformer_2', {
        'block': bert_trans(),
        'desc_inputs': [[1, 128, 768], [1, 128, 128]]}),

    ('BertModel', {
        'block': BertModel(config=BertConfig(batch_size=1,
                                             num_hidden_layers=1,
                                             intermediate_size=768,
                                             token_type_ids_from_dataset=False),
                           is_training=True),
        'desc_inputs': [Tensor(np.random.rand(128).astype(np.int32)),
                        Tensor(np.random.rand(128).astype(np.int32)), [128]],
        'desc_bprop': [[1, 128, 768], [1, 128, 768], [1, 128, 768]],
        'num_output': 3}),

    ('BertModel_1', {
        'block': BertModel(config=BertConfig(batch_size=1,
                                             num_hidden_layers=1,
                                             intermediate_size=768,
                                             token_type_ids_from_dataset=False),
                           is_training=False),
        'desc_inputs': [Tensor(np.random.rand(128).astype(np.int32)),
                        Tensor(np.random.rand(128).astype(np.int32)), [128]],
        'desc_bprop': [[1, 128, 768], [1, 128, 768], [1, 128, 768]],
        'num_output': 3}),

    ('BertModel_2', {
        'block': BertModel(config=BertConfig(batch_size=1,
                                             num_hidden_layers=1,
                                             intermediate_size=768,
                                             token_type_ids_from_dataset=False,
                                             input_mask_from_dataset=False),
                           is_training=True),
        'desc_inputs': [Tensor(np.random.rand(128).astype(np.int32)),
                        Tensor(np.random.rand(128).astype(np.int32)), [128]],
        'desc_bprop': [[1, 128, 768], [1, 128, 768], [1, 128, 768]],
        'num_output': 3}),

    ('BertPretrainingLoss', {
        'block': BertPretrainingLoss(config=BertConfig(batch_size=1)),
        'desc_inputs': [[32000], [20, 2], Tensor(np.array([1]).astype(np.int32)),
                        [20], Tensor(np.array([20]).astype(np.int32))],
        'desc_bprop': [[1]],
        'num_output': 1}),
    ('Dense_1', {
        'block': nn.Dense(in_channels=768,
                          out_channels=3072,
                          activation='gelu',
                          weight_init=TruncatedNormal(0.02)),
        'desc_inputs': [[3, 768]],
        'desc_bprop': [[3, 3072]]}),
    ('Dense_2', {
        'block': set_train(nn.Dense(in_channels=768,
                                    out_channels=3072,
                                    activation='gelu',
                                    weight_init=TruncatedNormal(0.02), )),
        'desc_inputs': [[3, 768]],
        'desc_bprop': [[3, 3072]]}),
    ('GetNextSentenceOutput', {
        'block': GetNextSentenceOutput(BertConfig(batch_size=1)),
        'desc_inputs': [[128, 768]],
        'desc_bprop': [[128, 2]]}),
    ('Adam_1', {
        'block': set_train(TrainStepWrapForAdam(NetForAdam())),
        'desc_inputs': [[1, 64], [1, 10]],
        'skip': ['backward']}),
    ('Adam_2', {
        'block': set_train(TrainStepWrapForAdam(GetNextSentenceOutput(BertConfig(batch_size=1)))),
        'desc_inputs': [[128, 768], [128, 2]],
        'skip': ['backward']}),
    ('AdamWeightDecayDynamicLR', {
        'block': set_train(TrainStepWrapForAdamDynamicLr(NetForAdam())),
        'desc_inputs': [[1, 64]],
        'skip': ['backward']}),
    ('ClipGradients', {
        'block': TempC2Wrap(clip_grad, 1, 1.0),
        'desc_inputs': [tuple(convert(shp) for shp in [[1], [1], [1]])],
        'skip': ['backward', 'exec']}),
]

test_case = functools.reduce(lambda x, y: x + y, [test_case_cell_ops])
# use -k to select certain testcast
# pytest  tests/python/ops/test_ops.py::test_backward -k LayerNorm


test_exec_case = filter(lambda x: 'skip' not in x[1] or
                                  'exec' not in x[1]['skip'], test_case)
test_backward_exec_case = filter(lambda x: 'skip' not in x[1] or
                                           'backward' not in x[1]['skip'] and 'backward_exec'
                                           not in x[1]['skip'], test_case)
test_check_gradient_case = filter(lambda x: 'skip' not in x[1] or
                                            'backward' not in x[1]['skip'] and 'backward_exec'
                                            not in x[1]['skip'], test_case)


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    return test_exec_case


@mindspore_test(pipeline_for_compile_grad_ge_graph_for_case_by_case_config)
def test_backward_exec():
    return test_backward_exec_case
