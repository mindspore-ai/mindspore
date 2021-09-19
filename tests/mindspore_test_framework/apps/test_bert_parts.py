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

"""Test bert submodules."""

import os

import numpy as np
from mindspore.tests.models.Bert_NEZHA import EmbeddingLookup, GetMaskedLMOutput, \
    BertConfig, BertPreTraining, BertNetworkWithLoss
from mindspore.tests.models.Bert_NEZHA.bert_model import BertModel

from mindspore import Tensor
from mindspore import nn, context
from ..mindspore_test import mindspore_test
from ..pipeline.forward.compile_forward import pipeline_for_compile_forward_anf_graph_for_case_by_case_config, \
    pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ..pipeline.gradient.compile_gradient import pipeline_for_compile_grad_anf_graph_for_case_by_case_config, \
    pipeline_for_compile_grad_ge_graph_for_case_by_case_config
from ..utils.block_util import get_output_cell
from ...dataset_mock import MindData

# pylint: disable=missing-docstring, W0612, arguments-differ
_current_dir = os.path.dirname(os.path.realpath(__file__)) + "/../python/test_data"


class BertPreTrainingNet(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=True):
        super(BertPreTrainingNet, self).__init__()
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.bert = BertPreTraining(config, self.is_training, self.use_one_hot_embeddings)

    def construct(self, input_ids_, input_mask_, token_type_id_,
                  next_sentence_labels_, masked_lm_positions_):
        t = next_sentence_labels_
        (prediction_scores, seq_relationship_score) = \
            self.bert(input_ids_, input_mask_, token_type_id_, masked_lm_positions_)
        return prediction_scores, seq_relationship_score


class BertModelNet(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(BertModelNet, self).__init__()
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.bert = BertModel(config, self.is_training, self.use_one_hot_embeddings)

    def construct(self, input_ids_, input_mask_, token_type_id_,
                  masked_lm_positions_):
        t = masked_lm_positions_
        sequence_output, pooled_output, embedding_table = \
            self.bert(input_ids_, input_mask_, token_type_id_)
        return sequence_output, pooled_output, embedding_table


def get_dataset(batch_size=1):
    dataset_types = (np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32)
    dataset_shapes = ((batch_size, 128), (batch_size, 128), (batch_size, 128), (batch_size, 1), \
                      (batch_size, 20), (batch_size, 20), (batch_size, 20))

    dataset = MindData(size=2, batch_size=batch_size,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset


def load_test_data():
    dataset = get_dataset()
    return dataset.next()


input_ids, input_mask, token_type_id, \
next_sentence_labels, masked_lm_positions, \
masked_lm_ids, masked_lm_weights = load_test_data()

test_sets = [
    ('BertNetworkWithLoss_1', {
        'block': BertNetworkWithLoss(BertConfig(batch_size=1), False, use_one_hot_embeddings=True),
        'desc_inputs': [input_ids, input_mask, token_type_id,
                        next_sentence_labels, masked_lm_positions,
                        masked_lm_ids, masked_lm_weights],
        'desc_bprop': [[1]]}),
    ('BertNetworkWithLoss_2', {
        'block': BertNetworkWithLoss(BertConfig(batch_size=1), False, True),
        'desc_inputs': [input_ids, input_mask, token_type_id,
                        next_sentence_labels, masked_lm_positions,
                        masked_lm_ids, masked_lm_weights],
        'desc_bprop': [[1]]}),
    ('BertPreTrainingNet_1', {
        'block': get_output_cell(BertPreTrainingNet(BertConfig(batch_size=1), True, True), 5, 0),
        'desc_inputs': [input_ids, input_mask, token_type_id,
                        next_sentence_labels, masked_lm_positions],
        'desc_bprop': [[20, 32000]]}),
    ('BertPreTrainingNet_2', {
        'block': get_output_cell(BertPreTrainingNet(BertConfig(batch_size=1), True, True), 5, 1),
        'desc_inputs': [input_ids, input_mask, token_type_id,
                        next_sentence_labels, masked_lm_positions],
        'desc_bprop': [[1, 2]],
        'skip': ['compile', 'exec']}),
    ('BertModelNet_1', {
        'block': get_output_cell(BertModelNet(BertConfig(batch_size=1), True, True), 4, 0),
        'desc_inputs': [input_ids, input_mask, token_type_id, masked_lm_positions],
        'desc_bprop': [[1, 128, 768]]}),
    ('BertModelNet_2', {
        'block': get_output_cell(BertModelNet(BertConfig(batch_size=1), True, True), 4, 1),
        'desc_inputs': [input_ids, input_mask, token_type_id, masked_lm_positions],
        'desc_bprop': [[1, 768]],
        'skip': ['compile', 'exec']}),
    ('BertModelNet_3', {
        'block': get_output_cell(BertModelNet(BertConfig(batch_size=1), True, True), 4, 2),
        'desc_inputs': [input_ids, input_mask, token_type_id, masked_lm_positions],
        'desc_bprop': [[32000, 768]],
        'skip': ['compile', 'exec']}),
    ('GetMaskedLMOutput', {
        'block': GetMaskedLMOutput(BertConfig(batch_size=1)),
        'desc_inputs': [[1, 128, 768], [32000, 768], Tensor(np.ones(shape=[20, 768]).astype(np.int32))],
        'desc_bprop': [[15360, 32000]]}),
    ('EmbeddingLookup_1', {
        'block': get_output_cell(EmbeddingLookup(vocab_size=32000,
                                                 embedding_size=768,
                                                 embedding_shape=[1, 128, 768],
                                                 use_one_hot_embeddings=False,
                                                 initializer_range=0.02), 1, 0),
        'desc_inputs': [input_ids],
        'desc_bprop': [[1, 128, 768]]}),
    ('EmbeddingLookup_2', {
        'block': get_output_cell(EmbeddingLookup(vocab_size=32000,
                                                 embedding_size=768,
                                                 embedding_shape=[1, 128, 768],
                                                 use_one_hot_embeddings=False,
                                                 initializer_range=0.02), 1, 1),
        'desc_inputs': [input_ids],
        'desc_bprop': [[128]]}),
    ('EmbeddingLookup_3', {
        'block': get_output_cell(EmbeddingLookup(vocab_size=32000,
                                                 embedding_size=768,
                                                 embedding_shape=[1, 128, 768],
                                                 use_one_hot_embeddings=True,
                                                 initializer_range=0.02), 1, 0),
        'desc_inputs': [input_ids],
        'desc_bprop': [[1, 128, 768]]}),
    ('EmbeddingLookup_4', {
        'block': get_output_cell(EmbeddingLookup(vocab_size=32000,
                                                 embedding_size=768,
                                                 embedding_shape=[1, 128, 768],
                                                 use_one_hot_embeddings=True,
                                                 initializer_range=0.02), 1, 1),
        'desc_inputs': [input_ids],
        'desc_bprop': [[128]]}),
    ('EmbeddingLookup_multi_outputs', {
        'block': EmbeddingLookup(vocab_size=32000,
                                 embedding_size=768,
                                 embedding_shape=[1, 128, 768],
                                 use_one_hot_embeddings=False,
                                 initializer_range=0.02),
        'desc_inputs': [input_ids],
        'desc_bprop': [[1, 128, 768], [128]]}),
    ('EmbeddingLookup_init_param', {
        'block': (get_output_cell(EmbeddingLookup(vocab_size=32000,
                                                  embedding_size=768,
                                                  embedding_shape=[1, 128, 768],
                                                  use_one_hot_embeddings=True,
                                                  initializer_range=0.02), 1, 1),
                  {'init_param_with': lambda shp: np.ones(shp).astype(np.float32)}),
        'desc_inputs': [input_ids],
        'desc_bprop': [[128]]}),
    ('EmbeddingLookup_multi_outputs_init_param', {
        'block': (EmbeddingLookup(vocab_size=32000,
                                  embedding_size=768,
                                  embedding_shape=[1, 128, 768],
                                  use_one_hot_embeddings=False,
                                  initializer_range=0.02),
                  {'init_param_with': lambda shp: np.ones(shp).astype(np.float32)}),
        'desc_inputs': [input_ids],
        'desc_bprop': [[1, 128, 768], [128]]}),
    ('EmbeddingLookup_multi_outputs_grad_with_no_sens', {
        'block': (EmbeddingLookup(vocab_size=32000,
                                  embedding_size=768,
                                  embedding_shape=[1, 128, 768],
                                  use_one_hot_embeddings=False,
                                  initializer_range=0.02),
                  {'init_param_with': lambda shp: np.ones(shp).astype(np.float32)}),
        'desc_inputs': [input_ids]}),
    ('GetMaskedLMOutput_grad_with_no_sens', {
        'block': GetMaskedLMOutput(BertConfig(batch_size=1)),
        'desc_inputs': [[1, 128, 768], [32000, 768], Tensor(np.ones(shape=[20, 768]).astype(np.int32))],
    }),
    ('BertModelNet_no_split_outputs', {
        'block': (BertModelNet(BertConfig(batch_size=1), True, True), {
            'split_outputs': False
        }),
        'desc_inputs': [input_ids, input_mask, token_type_id, masked_lm_positions],
        'desc_bprop': [[1, 128, 768], [1, 768], [32000, 768]]}),
]


@mindspore_test(pipeline_for_compile_forward_anf_graph_for_case_by_case_config)
def test_compile():
    context.set_context(mode=context.GRAPH_MODE)
    return test_sets


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_sets


@mindspore_test(pipeline_for_compile_grad_anf_graph_for_case_by_case_config)
def test_backward():
    context.set_context(mode=context.GRAPH_MODE)
    return test_sets


@mindspore_test(pipeline_for_compile_grad_ge_graph_for_case_by_case_config)
def test_backward_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_sets
