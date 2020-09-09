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

"""
sample script of processing CLUE classification dataset using mindspore.dataset.text for fine-tuning bert
"""

import os
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.text as text
import mindspore.dataset.transforms.c_transforms as ops


def process_tnews_clue_dataset(data_dir, label_list, bert_vocab_path, data_usage='train', shuffle_dataset=False,
                               max_seq_len=128, batch_size=64, drop_remainder=True):
    """Process TNEWS dataset"""
    ### Loading TNEWS from CLUEDataset
    assert data_usage in ['train', 'eval', 'test']
    if data_usage == 'train':
        dataset = ds.CLUEDataset(os.path.join(data_dir, "train.json"), task='TNEWS',
                                 usage=data_usage, shuffle=shuffle_dataset)
    elif data_usage == 'eval':
        dataset = ds.CLUEDataset(os.path.join(data_dir, "dev.json"), task='TNEWS',
                                 usage=data_usage, shuffle=shuffle_dataset)
    else:
        dataset = ds.CLUEDataset(os.path.join(data_dir, "test.json"), task='TNEWS',
                                 usage=data_usage, shuffle=shuffle_dataset)
    ### Processing label
    if data_usage == 'test':
        dataset = dataset.map(operations=ops.Duplicate(), input_columns=["id"], output_columns=["id", "label_id"],
                              column_order=["id", "label_id", "sentence"])
        dataset = dataset.map(operations=ops.Fill(0), input_columns=["label_id"])
    else:
        label_vocab = text.Vocab.from_list(label_list)
        label_lookup = text.Lookup(label_vocab)
        dataset = dataset.map(operations=label_lookup, input_columns="label_desc", output_columns="label_id")
    ### Processing sentence
    vocab = text.Vocab.from_file(bert_vocab_path)
    tokenizer = text.BertTokenizer(vocab, lower_case=True)
    lookup = text.Lookup(vocab, unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer, input_columns=["sentence"])
    dataset = dataset.map(operations=ops.Slice(slice(0, max_seq_len)), input_columns=["sentence"])
    dataset = dataset.map(operations=ops.Concatenate(prepend=np.array(["[CLS]"], dtype='S'),
                                                     append=np.array(["[SEP]"], dtype='S')), input_columns=["sentence"])
    dataset = dataset.map(operations=lookup, input_columns=["sentence"], output_columns=["text_ids"])
    dataset = dataset.map(operations=ops.PadEnd([max_seq_len], 0), input_columns=["text_ids"])
    dataset = dataset.map(operations=ops.Duplicate(), input_columns=["text_ids"],
                          output_columns=["text_ids", "mask_ids"],
                          column_order=["text_ids", "mask_ids", "label_id"])
    dataset = dataset.map(operations=ops.Mask(ops.Relational.NE, 0, mstype.int32), input_columns=["mask_ids"])
    dataset = dataset.map(operations=ops.Duplicate(), input_columns=["text_ids"],
                          output_columns=["text_ids", "segment_ids"],
                          column_order=["text_ids", "mask_ids", "segment_ids", "label_id"])
    dataset = dataset.map(operations=ops.Fill(0), input_columns=["segment_ids"])
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def process_cmnli_clue_dataset(data_dir, label_list, bert_vocab_path, data_usage='train', shuffle_dataset=False,
                               max_seq_len=128, batch_size=64, drop_remainder=True):
    """Process CMNLI dataset"""
    ### Loading CMNLI from CLUEDataset
    assert data_usage in ['train', 'eval', 'test']
    if data_usage == 'train':
        dataset = ds.CLUEDataset(os.path.join(data_dir, "train.json"), task='CMNLI',
                                 usage=data_usage, shuffle=shuffle_dataset)
    elif data_usage == 'eval':
        dataset = ds.CLUEDataset(os.path.join(data_dir, "dev.json"), task='CMNLI',
                                 usage=data_usage, shuffle=shuffle_dataset)
    else:
        dataset = ds.CLUEDataset(os.path.join(data_dir, "test.json"), task='CMNLI',
                                 usage=data_usage, shuffle=shuffle_dataset)
    ### Processing label
    if data_usage == 'test':
        dataset = dataset.map(operations=ops.Duplicate(), input_columns=["id"], output_columns=["id", "label_id"],
                              column_order=["id", "label_id", "sentence1", "sentence2"])
        dataset = dataset.map(operations=ops.Fill(0), input_columns=["label_id"])
    else:
        label_vocab = text.Vocab.from_list(label_list)
        label_lookup = text.Lookup(label_vocab)
        dataset = dataset.map(operations=label_lookup, input_columns="label", output_columns="label_id")
    ### Processing sentence pairs
    vocab = text.Vocab.from_file(bert_vocab_path)
    tokenizer = text.BertTokenizer(vocab, lower_case=True)
    lookup = text.Lookup(vocab, unknown_token='[UNK]')
    ### Tokenizing sentences and truncate sequence pair
    dataset = dataset.map(operations=tokenizer, input_columns=["sentence1"])
    dataset = dataset.map(operations=tokenizer, input_columns=["sentence2"])
    dataset = dataset.map(operations=text.TruncateSequencePair(max_seq_len - 3),
                          input_columns=["sentence1", "sentence2"])
    ### Adding special tokens
    dataset = dataset.map(operations=ops.Concatenate(prepend=np.array(["[CLS]"], dtype='S'),
                                                     append=np.array(["[SEP]"], dtype='S')),
                          input_columns=["sentence1"])
    dataset = dataset.map(operations=ops.Concatenate(append=np.array(["[SEP]"], dtype='S')),
                          input_columns=["sentence2"])
    ### Generating segment_ids
    dataset = dataset.map(operations=ops.Duplicate(), input_columns=["sentence1"],
                          output_columns=["sentence1", "type_sentence1"],
                          column_order=["sentence1", "type_sentence1", "sentence2", "label_id"])
    dataset = dataset.map(operations=ops.Duplicate(),
                          input_columns=["sentence2"], output_columns=["sentence2", "type_sentence2"],
                          column_order=["sentence1", "type_sentence1", "sentence2", "type_sentence2", "label_id"])
    dataset = dataset.map(operations=[lookup, ops.Fill(0)], input_columns=["type_sentence1"])
    dataset = dataset.map(operations=[lookup, ops.Fill(1)], input_columns=["type_sentence2"])
    dataset = dataset.map(operations=ops.Concatenate(),
                          input_columns=["type_sentence1", "type_sentence2"], output_columns=["segment_ids"],
                          column_order=["sentence1", "sentence2", "segment_ids", "label_id"])
    dataset = dataset.map(operations=ops.PadEnd([max_seq_len], 0), input_columns=["segment_ids"])
    ### Generating text_ids
    dataset = dataset.map(operations=ops.Concatenate(),
                          input_columns=["sentence1", "sentence2"], output_columns=["text_ids"],
                          column_order=["text_ids", "segment_ids", "label_id"])
    dataset = dataset.map(operations=lookup, input_columns=["text_ids"])
    dataset = dataset.map(operations=ops.PadEnd([max_seq_len], 0), input_columns=["text_ids"])
    ### Generating mask_ids
    dataset = dataset.map(operations=ops.Duplicate(), input_columns=["text_ids"],
                          output_columns=["text_ids", "mask_ids"],
                          column_order=["text_ids", "mask_ids", "segment_ids", "label_id"])
    dataset = dataset.map(operations=ops.Mask(ops.Relational.NE, 0, mstype.int32), input_columns=["mask_ids"])
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
