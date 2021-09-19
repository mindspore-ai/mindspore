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

"""Test bert check gradient."""

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import context, nn
from mindspore.tests.models.Bert_NEZHA import GetNextSentenceOutput, BertNetworkWithLoss
from mindspore.tests.models.Bert_NEZHA.bert_model import BertConfig, \
    EmbeddingLookup, EmbeddingPostprocessor, BertOutput, \
    BertAttention, BertSelfAttention, SaturateCast, TruncatedNormal, \
    BertEncoderCell, BertTransformer, CreateAttentionMaskFromInputMask, BertModel
from .bert_attention_submodules import BertAttentionQueryKeyMul, BertAttentionRelativePositionKeys, \
    BertAttentionMaskBackward, BertAttentionSoftmax, BertAttentionRelativePositionValues, BertDense
from ..mindspore_test import mindspore_test
from ..pipeline.gradient.compare_gradient import \
    pipeline_for_compare_inputs_grad_with_numerical_diff_for_group_by_group_config, \
    pipeline_for_compare_params_grad_with_numerical_diff_for_group_by_group_config

verification_set = {
    'inputs': [
        {
            'id': 'SaturateCast_CICase',
            'group': 'bert',
            'desc_inputs': [
                [1, 3, 4, 4],
            ]
        },
        {
            'id': 'BertAttention',
            'group': 'bert',
            'desc_inputs': [
                [1, 128, 1024], [1, 128, 1024], [1, 128, 128],
            ]
        },
        {
            'id': 'BertOutput',
            'group': 'bert',
            'desc_inputs': [
                [8192, 1024], [8192, 1024],
            ]
        },
        {
            'id': 'BertSelfAttention',
            'group': 'bert',
            'desc_inputs': [
                [1, 128, 1024], [1, 128, 1024]
            ]
        },
        {
            'id': 'BertEncoderCell',
            'group': 'bert',
            'desc_inputs': [
                [1, 128, 1024], [1, 128, 128]
            ]
        },
        {
            'id': 'BertTransformer',
            'group': 'bert',
            'desc_inputs': [
                [1, 128, 1024], [1, 128, 128]
            ]
        },
        {
            'id': 'EmbeddingLookup',
            'group': 'bert',
            'desc_inputs': [
                np.random.rand(128).astype(np.int32)
            ]
        },
        {
            'id': 'EmbeddingPostprocessor',
            'group': 'bert',
            'desc_inputs': [
                np.random.rand(128).astype(np.int32), [1, 128, 1024]
            ]
        },
        {
            'id': 'CreateAttentionMaskFromInputMask',
            'group': 'bert',
            'desc_inputs': [
                [128]
            ]
        },
        {
            'id': 'BertModel',
            'group': 'bert',
            'desc_inputs': [
                np.random.rand(128).astype(np.int32),
                np.random.rand(128).astype(np.int32),
                [128]
            ]
        },
        {
            'id': 'Dense',
            'group': 'bert',
            'desc_inputs': [
                [3, 768]
            ]
        },
        {
            'id': 'GetNextSentenceOutput',
            'group': 'bert',
            'desc_inputs': [
                [128, 768]
            ]
        },
        {
            'id': 'BertDense_CICase',
            'group': 'bert',
            'desc_inputs': [
                np.ones(shape=(8, 8)).astype(np.float32)
            ]
        },
        {
            'id': 'BertNetworkWithLoss',
            'group': 'bert',
            'desc_inputs': [
                np.ones(shape=(1, 128)).astype(np.int32),
                np.ones(shape=(1, 128)).astype(np.int32),
                np.ones(shape=(1, 128)).astype(np.int32),
                np.ones(shape=(1, 1)).astype(np.int32),
                np.ones(shape=(1, 20)).astype(np.int32),
                np.ones(shape=(1, 20)).astype(np.int32),
                np.random.uniform(-0.1, 0.1, size=(1, 20)).astype(np.float32)
            ]
        },
        {
            'id': 'BertAttentionQueryKeyMul_CICase',
            'group': 'bert',
            'desc_inputs': [
                [1, 16, 128, 64],
                [1, 16, 128, 64]
            ]
        },
        {
            'id': 'BertAttentionRelativePositionKeys_CICase',
            'group': 'bert',
            'desc_inputs': [
                [1, 16, 128, 128],
                [1, 16, 128, 64]
            ]
        },
        {
            'id': 'BertAttentionRelativePositionValues_CICase',
            'group': 'bert',
            'desc_inputs': [
                [1, 16, 128, 64],
                [1, 16, 128, 128]
            ],
            'desc_bprop': [
                [128, 128, 64],
                [128, 1024]
            ]
        },
        {
            'id': 'BertAttentionMask_CICase',
            'group': 'bert',
            'desc_inputs': [
                [1, 16, 128, 128]
            ],
            'desc_bprop': [
                [1, 16, 128, 128]
            ]
        },
        {
            'id': 'BertAttentionSoftmax_CICase',
            'group': 'bert',
            'desc_inputs': [
                [128, 1024],
                [1, 16, 128, 128]
            ],
            'desc_bprop': [
                [1, 16, 128, 64],
                [1, 16, 128, 64]
            ]
        },
    ],
    'function': [
        {
            'id': 'SaturateCast_CICase',
            'group': 'bert',
            'block': SaturateCast(),
            'max_error': 2e-3,
            'reduce_output': False
        },
        {
            'id': 'BertAttention',
            'group': 'bert',
            'block': BertAttention(batch_size=1,
                                   from_tensor_width=1024,
                                   to_tensor_width=1024,
                                   from_seq_length=128,
                                   to_seq_length=128,
                                   num_attention_heads=16,
                                   size_per_head=64,
                                   query_act=None,
                                   key_act=None,
                                   value_act=None,
                                   has_attention_mask=True,
                                   attention_probs_dropout_prob=0.0,
                                   use_one_hot_embeddings=False,
                                   initializer_range=0.02,
                                   do_return_2d_tensor=True,
                                   use_relative_positions=True,
                                   compute_type=mstype.float32),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertOutput',
            'group': 'bert',
            'block': BertOutput(in_channels=1024,
                                out_channels=1024,
                                initializer_range=0.02,
                                dropout_prob=0.0),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertSelfAttention',
            'group': 'bert',
            'block': BertSelfAttention(batch_size=1,
                                       seq_length=128,
                                       hidden_size=1024,
                                       num_attention_heads=16,
                                       attention_probs_dropout_prob=0.0,
                                       use_one_hot_embeddings=False,
                                       initializer_range=0.02,
                                       hidden_dropout_prob=0.0,
                                       use_relative_positions=True,
                                       compute_type=mstype.float32),
            'reduce_output': False
        },
        {
            'id': 'BertEncoderCell',
            'group': 'bert',
            'block': BertEncoderCell(batch_size=1,
                                     hidden_size=1024,
                                     seq_length=128,
                                     num_attention_heads=16,
                                     intermediate_size=4096,
                                     attention_probs_dropout_prob=0.0,
                                     use_one_hot_embeddings=False,
                                     initializer_range=0.02,
                                     hidden_dropout_prob=0.0,
                                     use_relative_positions=True,
                                     hidden_act="gelu",
                                     compute_type=mstype.float32),
            'reduce_output': False
        },
        {
            'id': 'BertTransformer',
            'group': 'bert',
            'block': BertTransformer(batch_size=1,
                                     hidden_size=1024,
                                     seq_length=128,
                                     num_hidden_layers=2,
                                     num_attention_heads=16,
                                     intermediate_size=4096,
                                     attention_probs_dropout_prob=0.0,
                                     use_one_hot_embeddings=False,
                                     initializer_range=0.02,
                                     use_relative_positions=True,
                                     hidden_act="gelu",
                                     compute_type=mstype.float32,
                                     return_all_encoders=True)
        },
        {
            'id': 'EmbeddingLookup',
            'group': 'bert',
            'block': EmbeddingLookup(vocab_size=21128,
                                     embedding_size=1024,
                                     embedding_shape=[1, 128, 1024],
                                     use_one_hot_embeddings=False,
                                     initializer_range=0.02),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'EmbeddingPostprocessor',
            'group': 'bert',
            'block': EmbeddingPostprocessor(embedding_size=1024,
                                            embedding_shape=[1, 128, 1024],
                                            use_token_type=True,
                                            token_type_vocab_size=2,
                                            use_one_hot_embeddings=False,
                                            initializer_range=0.02,
                                            max_position_embeddings=512,
                                            dropout_prob=0.0),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'CreateAttentionMaskFromInputMask',
            'group': 'bert',
            'block': CreateAttentionMaskFromInputMask(config=BertConfig(
                batch_size=1,
                seq_length=128,
                vocab_size=21128,
                hidden_size=1024,
                num_hidden_layers=2,
                num_attention_heads=16,
                intermediate_size=4096,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                use_relative_positions=True,
                input_mask_from_dataset=True,
                token_type_ids_from_dataset=True,
                dtype=mstype.float32,
                compute_type=mstype.float32)),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertOutput',
            'group': 'bert',
            'block': BertOutput(in_channels=1024,
                                out_channels=1024,
                                initializer_range=0.02,
                                dropout_prob=0.0),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertModel',
            'group': 'BertModel',
            'block': BertModel(config=BertConfig(batch_size=1,
                                                 num_hidden_layers=2,
                                                 intermediate_size=4096,
                                                 token_type_ids_from_dataset=True),
                               is_training=True),
            'reduce_output': False
        },
        {
            'id': 'Dense',
            'group': 'Dense',
            'block': nn.Dense(in_channels=768,
                              out_channels=3072,
                              activation='gelu',
                              weight_init=TruncatedNormal(0.02)),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'GetNextSentenceOutput',
            'group': 'GetNextSentenceOutput',
            'block': GetNextSentenceOutput(BertConfig(batch_size=1)),
            'reduce_output': False
        },
        {
            'id': 'BertDense_CICase',
            'group': 'bert',
            'block': BertDense(
                hidden_size=8,
                intermediate_size=8,
                initializer_range=0.02),
            'reduce_output': False
        },
        {
            'id': 'BertNetworkWithLoss',
            'group': 'bert',
            'block': BertNetworkWithLoss(config=BertConfig(
                batch_size=1,
                seq_length=128,
                vocab_size=21128,
                hidden_size=1024,
                num_hidden_layers=2,
                num_attention_heads=16,
                intermediate_size=4096,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                use_relative_positions=True,
                input_mask_from_dataset=True,
                token_type_ids_from_dataset=True,
                dtype=mstype.float32,
                compute_type=mstype.float32), is_training=True),
            'reduce_output': False
        },
        {
            'id': 'BertAttentionQueryKeyMul_CICase',
            'group': 'bert',
            'block': BertAttentionQueryKeyMul(batch_size=1,
                                              from_tensor_width=1024,
                                              to_tensor_width=1024,
                                              from_seq_length=128,
                                              to_seq_length=128,
                                              num_attention_heads=16,
                                              size_per_head=64,
                                              query_act=None,
                                              key_act=None,
                                              initializer_range=0.02),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertAttentionRelativePositionKeys_CICase',
            'group': 'BertAttentionRelativePositionKeys',
            'block': BertAttentionRelativePositionKeys(batch_size=1,
                                                       from_seq_length=128,
                                                       to_seq_length=128,
                                                       num_attention_heads=16,
                                                       size_per_head=64,
                                                       use_one_hot_embeddings=False,
                                                       initializer_range=0.02,
                                                       use_relative_positions=True,
                                                       dtype=mstype.float32,
                                                       compute_type=mstype.float32),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertAttentionRelativePositionValues_CICase',
            'group': 'BertAttentionRelativePositionValues',
            'block': BertAttentionRelativePositionValues(batch_size=1,
                                                         from_seq_length=128,
                                                         to_seq_length=128,
                                                         num_attention_heads=16,
                                                         size_per_head=64,
                                                         use_one_hot_embeddings=False,
                                                         initializer_range=0.02,
                                                         do_return_2d_tensor=True,
                                                         use_relative_positions=True,
                                                         dtype=mstype.float32,
                                                         compute_type=mstype.float32),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertAttentionMask_CICase',
            'group': 'BertAttentionMask',
            'block': BertAttentionMaskBackward((1, 128, 128),
                                               has_attention_mask=True,
                                               dtype=mstype.float32),
            'sampling_times': 10,
            'reduce_output': False
        },
        {
            'id': 'BertAttentionSoftmax_CICase',
            'group': 'BertAttentionSoftmax',
            'block': BertAttentionSoftmax(batch_size=1,
                                          to_tensor_width=1024,
                                          from_seq_length=128,
                                          to_seq_length=128,
                                          num_attention_heads=16,
                                          size_per_head=64,
                                          value_act=None,
                                          attention_probs_dropout_prob=0,
                                          initializer_range=0.02),
            'sampling_times': 10,
            'reduce_output': False
        },
    ],
    'ext': {}
}


@mindspore_test(pipeline_for_compare_inputs_grad_with_numerical_diff_for_group_by_group_config)
def test_bert_check_gradient_wrt_inputs_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return verification_set


@mindspore_test(pipeline_for_compare_params_grad_with_numerical_diff_for_group_by_group_config)
def test_bert_check_gradient_wrt_params_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return verification_set
