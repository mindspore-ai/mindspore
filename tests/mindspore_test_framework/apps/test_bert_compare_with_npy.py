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

"""Test bert compare with npy."""

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.tests.models.Bert_NEZHA.bert_model import BertAttention, SaturateCast, \
    EmbeddingLookup, BertModel, \
    BertConfig, EmbeddingPostprocessor, \
    BertTransformer, BertEncoderCell, \
    BertSelfAttention, CreateAttentionMaskFromInputMask, \
    RelaPosMatrixGenerator, BertOutput, \
    RelaPosEmbeddingsGenerator
from .bert_attention_submodules import BertAttentionQueryKeyMul, BertAttentionRelativePositionKeys, BertAttentionMask, \
    BertAttentionSoftmax, BertAttentionRelativePositionValues, BertDense
from ..mindspore_test import mindspore_test
from ..pipeline.forward.compare_forward import \
    pipeline_for_compare_forward_with_npy_for_group_by_group_config_using_group_policy

verification_set = {
    'inputs': [
        {
            'id': 'BertModel',
            'group': 'BertModel',
            'desc_inputs': [
                'apps/bert_data/input_fn_IteratorGetNext_output_0.npy',
                'apps/bert_data/input_fn_IteratorGetNext_output_1.npy',
                'apps/bert_data/input_fn_IteratorGetNext_output_6.npy',
            ]
        },
        {
            'id': 'EmbeddingPostprocessor_CICase',
            'group': 'EmbeddingPostprocessor',
            'desc_inputs': [
                ('apps/bert_data/input_fn_IteratorGetNext_output_6.npy', {'dtype': np.int32}),
                ('apps/bert_data/bert_embeddings_Reshape_1_output_0.npy', {'dtype': np.float32})
            ]
        },
        {
            'id': 'BertTransformer',
            'group': 'BertTransformer',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_saturate_cast_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy',
            ]
        },
        {
            'id': 'SaturateCast_CICase',
            'group': 'SaturateCast',
            'desc_inputs': [
                'apps/bert_data/bert_embeddings_LayerNorm_batchnorm_add_1_output_0.npy',
            ]
        },
        {
            'id': 'RelaPosMatrixGenerator',
            'group': 'RelaPosMatrixGenerator',
            'desc_inputs': [
            ]
        },
        {
            'id': 'RelaPosEmbeddingsGenerator',
            'group': 'RelaPosEmbeddingsGenerator',
            'desc_inputs': [
            ]
        },
        {
            'id': 'BertAttention_0',
            'group': 'BertAttention',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy',
            ]
        },
        {
            'id': 'BertAttention_1',
            'group': 'BertAttention',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_attention_output_LayerNorm_batchnorm_add_1_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_output_LayerNorm_batchnorm_add_1_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy',
            ]
        },
        {
            'id': 'BertAttentionQueryKeyMul_CICase',
            'group': 'BertAttentionQueryKeyMul',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
            ],
            'dtype': 'float32'
        },
        {
            'id': 'BertAttentionRelativePositionKeys_CICase',
            'group': 'BertAttentionRelativePositionKeys',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_MatMul_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_self_transpose_output_0.npy',

            ],
            'dtype': 'float32'
        },
        {
            'id': 'BertAttentionMask_CICase',
            'group': 'BertAttentionMask',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_Mul_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy',

            ]
        },
        {
            'id': 'BertAttentionSoftmax_CICase',
            'group': 'BertAttentionSoftmax',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_self_add_1_output_0.npy',
            ]
        },
        {
            'id': 'BertAttentionRelativePositionValues_CICase',
            'group': 'BertAttentionRelativePositionValues',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_MatMul_2_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_self_Softmax_output_0.npy',
            ]
        },
        {
            'id': 'BertOutput_0_CICase',
            'group': 'BertOutput_0',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_intermediate_dense_gelu_mul_3_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_output_LayerNorm_batchnorm_add_1_output_0.npy',
            ]
        },
        {
            'id': 'BertOutput_1_CICase',
            'group': 'BertOutput_1',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_1_intermediate_dense_gelu_mul_3_output_0.npy',
                'apps/bert_data/bert_encoder_layer_1_attention_output_LayerNorm_batchnorm_add_1_output_0.npy',
            ]
        },
        {
            'id': 'BertSelfAttention_0',
            'group': 'BertSelfAttention',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy',
            ]
        },
        {
            'id': 'BertSelfAttention_1',
            'group': 'BertSelfAttention',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_attention_output_LayerNorm_batchnorm_add_1_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy',
            ]
        },
        {
            'id': 'BertEncoderCell',
            'group': 'BertEncoderCell',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
                'apps/bert_data/bert_encoder_mul_output_0.npy'
            ]
        },
        {
            'id': 'EmbeddingLookup_CICase',
            'group': 'EmbeddingLookup',
            'desc_inputs': [
                ('apps/bert_data/input_fn_IteratorGetNext_output_0.npy', {'dtype': np.int32})
            ]
        },
        {
            'id': 'BertDense_CICase',
            'group': 'BertDense',
            'desc_inputs': [
                'apps/bert_data/bert_encoder_layer_0_attention_output_LayerNorm_batchnorm_add_1_output_0.npy',
            ]
        },
        {
            'id': 'CreateAttentionMaskFromInputMask_CICase',
            'group': 'CreateAttentionMaskFromInputMask',
            'desc_inputs': [
                'apps/bert_data/input_fn_IteratorGetNext_output_1.npy',
            ]
        }
    ],
    'expect': [
        {
            'id': 'BertModel-BertModel',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_Cast_output_0.npy',
                'apps/bert_data/bert_pooler_dense_Tanh_output_0.npy',
                'apps/bert_data/bert_embeddings_word_embeddings_read_output_0.npy',
            ]
        },
        {
            'id': 'EmbeddingPostprocessor-EmbeddingPostprocessor_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_embeddings_LayerNorm_batchnorm_add_1_output_0.npy',
            ]
        },
        {
            'id': 'BertTransformer-BertTransformer',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_output_LayerNorm_batchnorm_add_1_output_0.npy',
                'apps/bert_data/bert_encoder_layer_1_output_LayerNorm_batchnorm_add_1_output_0.npy'
            ]
        },
        {
            'id': 'SaturateCast-SaturateCast_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_saturate_cast_output_0.npy'
            ]
        },
        {
            'id': 'RelaPosMatrixGenerator-RelaPosMatrixGenerator',
            'group': 'bert',
            'desc_expect': [
            ]
        },
        {
            'id': 'RelaPosEmbeddingsGenerator-RelaPosEmbeddingsGenerator',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_relative_positions_keys_GatherV2_output_0.npy',
                'apps/bert_data/bert_encoder_layer_1_attention_self_relative_positions_keys_GatherV2_output_0.npy'
            ]
        },
        {
            'id': 'BertAttention-BertAttention_0',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_Reshape_7_output_0.npy'
            ]
        },
        {
            'id': 'BertAttention-BertAttention_1',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_1_attention_self_Reshape_7_output_0.npy',
            ]
        },
        {
            'id': 'BertAttentionQueryKeyMul-BertAttentionQueryKeyMul_CICase',
            'group': 'bert',
            'desc_expect': [
                ('apps/bert_data/mul_query_layer.npy', {'max_error': 1e-3}),
                ('apps/bert_data/mul_key_layer.npy', {'max_error': 1e-3}),
                ('apps/bert_data/mul_attention_scores.npy', {'max_error': 1e-3})
            ]
        },
        {
            'id': 'BertAttentionRelativePositionKeys-BertAttentionRelativePositionKeys_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_relative_positions_keys_GatherV2_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_self_Mul_output_0.npy'
            ]
        },
        {
            'id': 'BertAttentionMask-BertAttentionMask_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_add_1_output_0.npy',
            ]
        },
        {
            'id': 'BertAttentionSoftmax-BertAttentionSoftmax_CICase',
            'group': 'bert',
            'desc_expect': [
                ('apps/bert_data/BertAttentionSoftmax_value_layer.npy', {'max_error': 1e-3}),
                ('apps/bert_data/BertAttentionSoftmax_context_layer.npy', {'max_error': 1e-3})
            ]
        },
        {
            'id': 'BertAttentionRelativePositionValues-BertAttentionRelativePositionValues_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_attention_self_relative_positions_values_GatherV2_output_0.npy',
                'apps/bert_data/bert_encoder_layer_0_attention_self_Reshape_7_output_0.npy'
            ]
        },
        {
            'id': 'BertOutput_0-BertOutput_0_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/BertOutput_output0.npy',
            ]
        },
        {
            'id': 'BertOutput_1-BertOutput_1_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/BertOutput_output1.npy',
            ]
        },
        {
            'id': 'BertSelfAttention-BertSelfAttention_0',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_attention_output_LayerNorm_batchnorm_add_1_output_0.npy'
            ]
        },
        {
            'id': 'BertSelfAttention-BertSelfAttention_1',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_1_attention_output_LayerNorm_batchnorm_add_1_output_0.npy'
            ]
        },
        {
            'id': 'BertEncoderCell-BertEncoderCell',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_layer_0_output_LayerNorm_batchnorm_add_1_output_0.npy',
            ]
        },
        {
            'id': 'EmbeddingLookup-EmbeddingLookup_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_embeddings_Reshape_1_output_0.npy',
            ]
        },
        {
            'id': 'BertDense-BertDense_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/dense_without_gelu.npy',
            ]
        },
        {
            'id': 'CreateAttentionMaskFromInputMask-CreateAttentionMaskFromInputMask_CICase',
            'group': 'bert',
            'desc_expect': [
                'apps/bert_data/bert_encoder_mul_output_0.npy',
            ]
        }
    ],
    'function': [
        {
            'id': 'SaturateCast',
            'group': 'SaturateCast',
            'block': SaturateCast()
        },
        {
            'id': 'RelaPosMatrixGenerator',
            'group': 'RelaPosMatrixGenerator',
            'block': RelaPosMatrixGenerator(length=128, max_relative_position=16)
        },
        {
            'id': 'RelaPosEmbeddingsGenerator',
            'group': 'RelaPosEmbeddingsGenerator',
            'block': RelaPosEmbeddingsGenerator(length=128, depth=64,
                                                max_relative_position=16,
                                                initializer_range=0.02,
                                                use_one_hot_embeddings=False)
        },
        {
            'id': 'BertAttention',
            'group': 'BertAttention',
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
                                   compute_type=mstype.float32)
        },
        {
            'id': 'BertAttentionQueryKeyMul',
            'group': 'BertAttentionQueryKeyMul',
            'block': BertAttentionQueryKeyMul(batch_size=1,
                                              from_tensor_width=1024,
                                              to_tensor_width=1024,
                                              from_seq_length=128,
                                              to_seq_length=128,
                                              num_attention_heads=16,
                                              size_per_head=64,
                                              query_act=None,
                                              key_act=None,
                                              initializer_range=0.02)
        },
        {
            'id': 'BertAttentionRelativePositionKeys',
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
                                                       compute_type=mstype.float32)
        },
        {
            'id': 'BertAttentionMask',
            'group': 'BertAttentionMask',
            'block': BertAttentionMask(has_attention_mask=True,
                                       dtype=mstype.float32)
        },
        {
            'id': 'BertAttentionSoftmax',
            'group': 'BertAttentionSoftmax',
            'block': BertAttentionSoftmax(batch_size=1,
                                          to_tensor_width=1024,
                                          from_seq_length=128,
                                          to_seq_length=128,
                                          num_attention_heads=16,
                                          size_per_head=64,
                                          value_act=None,
                                          attention_probs_dropout_prob=0.0,
                                          initializer_range=0.02)
        },
        {
            'id': 'BertAttentionRelativePositionValues',
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
                                                         compute_type=mstype.float32)
        },
        {
            'id': 'EmbeddingLookup',
            'group': 'EmbeddingLookup',
            'block': EmbeddingLookup(vocab_size=21128,
                                     embedding_size=1024,
                                     embedding_shape=[1, 128, 1024],
                                     use_one_hot_embeddings=False,
                                     initializer_range=0.02)
        },
        {
            'id': 'BertModel',
            'group': 'BertModel',
            'block': BertModel(config=BertConfig(batch_size=1,
                                                 num_hidden_layers=2,
                                                 intermediate_size=4096,
                                                 token_type_ids_from_dataset=True),
                               is_training=True)
        },
        {
            'id': 'EmbeddingPostprocessor',
            'group': 'EmbeddingPostprocessor',
            'block': EmbeddingPostprocessor(embedding_size=1024,
                                            embedding_shape=[1, 128, 1024],
                                            use_token_type=True,
                                            token_type_vocab_size=2,
                                            use_one_hot_embeddings=False,
                                            initializer_range=0.02,
                                            max_position_embeddings=512,
                                            dropout_prob=0.0)
        },
        {
            'id': 'BertTransformer',
            'group': 'BertTransformer',
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
            'id': 'BertEncoderCell',
            'group': 'BertEncoderCell',
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
                                     compute_type=mstype.float32)
        },
        {
            'id': 'BertSelfAttention',
            'group': 'BertSelfAttention',
            'block': BertSelfAttention(batch_size=1,
                                       seq_length=128,
                                       hidden_size=1024,
                                       num_attention_heads=16,
                                       attention_probs_dropout_prob=0.0,
                                       use_one_hot_embeddings=False,
                                       initializer_range=0.02,
                                       hidden_dropout_prob=0.0,
                                       use_relative_positions=True,
                                       compute_type=mstype.float32)
        },
        {
            'id': 'BertOutput_0',
            'group': 'BertOutput_0',
            'block': BertOutput(in_channels=4096,
                                out_channels=1024,
                                initializer_range=0.02,
                                dropout_prob=0.0)
        },
        {
            'id': 'BertOutput_1',
            'group': 'BertOutput_1',
            'block': BertOutput(in_channels=4096,
                                out_channels=1024,
                                initializer_range=0.02,
                                dropout_prob=0.0)
        },
        {
            'id': 'BertDense',
            'group': 'BertDense',
            'block': BertDense(
                hidden_size=1024,
                intermediate_size=4096,
                initializer_range=0.02)
        },
        {
            'id': 'CreateAttentionMaskFromInputMask',
            'group': 'CreateAttentionMaskFromInputMask',
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
                compute_type=mstype.float32))
        }
    ],
    'ext': {}
}


@mindspore_test(pipeline_for_compare_forward_with_npy_for_group_by_group_config_using_group_policy)
def test_bert_compare_with_npy_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return verification_set
