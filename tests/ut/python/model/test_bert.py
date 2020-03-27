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
""" test bert cell """
import numpy as np
import pytest
from mindspore import Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore.model_zoo.Bert_NEZHA import BertConfig, BertModel, BertNetworkWithLoss, BertTrainOneStepCell
from ....dataset_mock import MindData


def map_bert(record):
    target_data = {'input_ids': None, 'input_mask': None,
                   'segment_ids': None, 'next_sentence_labels': None,
                   'masked_lm_positions': None, 'masked_lm_ids': None,
                   'masked_lm_weights': None}

    sample = dt.parse_single_example(record, target_data)

    return sample['input_ids'], sample['input_mask'], sample['segment_ids'], \
           sample['next_sentence_labels'], sample['masked_lm_positions'], \
           sample['masked_lm_ids'], sample['masked_lm_weights']


def test_bert_model():
    # test for config.hidden_size % config.num_attention_heads != 0
    config_error = BertConfig(32, hidden_size=512, num_attention_heads=10)
    with pytest.raises(ValueError):
        BertModel(config_error, True)


def get_dataset(batch_size=1):
    dataset_types = (np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32)
    dataset_shapes = ((batch_size, 128), (batch_size, 128), (batch_size, 128), (batch_size, 1),
                      (batch_size, 20), (batch_size, 20), (batch_size, 20))

    dataset = MindData(size=2, batch_size=batch_size,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset
