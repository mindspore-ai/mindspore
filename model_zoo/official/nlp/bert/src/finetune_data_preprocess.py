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
import argparse
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.text as text
import mindspore.dataset.transforms.c_transforms as ops
from utils import convert_labels_to_index


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


def process_cluener_msra(data_file):
    """process MSRA dataset for CLUE"""
    content = []
    labels = []
    for line in open(data_file):
        line = line.strip()
        if line:
            word = line.split("\t")[0]
            if len(line.split("\t")) == 1:
                label = "O"
            else:
                label = line.split("\t")[1].split("\n")[0]
                if label[0] != "O":
                    label = label[0] + "_" + label[2:]
                if label[0] == "I":
                    label = "M" + label[1:]
            content.append(word)
            labels.append(label)
        else:
            for i in range(1, len(labels) - 1):
                if labels[i][0] == "B" and labels[i+1][0] != "M":
                    labels[i] = "S" + labels[i][1:]
                elif labels[i][0] == "M" and labels[i+1][0] != labels[i][0]:
                    labels[i] = "E" + labels[i][1:]
            last = len(labels) - 1
            if labels[last][0] == "B":
                labels[last] = "S" + labels[last][1:]
            elif labels[last][0] == "M":
                labels[last] = "E" + labels[last][1:]

            yield (np.array("".join(content)), np.array(list(labels)))
            content.clear()
            labels.clear()
            continue


def process_msra_clue_dataset(data_dir, label_list, bert_vocab_path, max_seq_len=128):
    """Process MSRA dataset"""
    ### Loading MSRA from CLUEDataset
    dataset = ds.GeneratorDataset(process_cluener_msra(data_dir), column_names=['text', 'label'])

    ### Processing label
    label_vocab = text.Vocab.from_list(label_list)
    label_lookup = text.Lookup(label_vocab)
    dataset = dataset.map(operations=label_lookup, input_columns="label", output_columns="label_ids")
    dataset = dataset.map(operations=ops.Concatenate(prepend=np.array([0], dtype='i')),
                          input_columns=["label_ids"])
    dataset = dataset.map(operations=ops.Slice(slice(0, max_seq_len)), input_columns=["label_ids"])
    dataset = dataset.map(operations=ops.PadEnd([max_seq_len], 0), input_columns=["label_ids"])
    ### Processing sentence
    vocab = text.Vocab.from_file(bert_vocab_path)
    lookup = text.Lookup(vocab, unknown_token='[UNK]')
    unicode_char_tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=unicode_char_tokenizer, input_columns=["text"], output_columns=["sentence"])
    dataset = dataset.map(operations=ops.Slice(slice(0, max_seq_len-2)), input_columns=["sentence"])
    dataset = dataset.map(operations=ops.Concatenate(prepend=np.array(["[CLS]"], dtype='S'),
                                                     append=np.array(["[SEP]"], dtype='S')), input_columns=["sentence"])
    dataset = dataset.map(operations=lookup, input_columns=["sentence"], output_columns=["input_ids"])
    dataset = dataset.map(operations=ops.PadEnd([max_seq_len], 0), input_columns=["input_ids"])
    dataset = dataset.map(operations=ops.Duplicate(), input_columns=["input_ids"],
                          output_columns=["input_ids", "input_mask"],
                          column_order=["input_ids", "input_mask", "label_ids"])
    dataset = dataset.map(operations=ops.Mask(ops.Relational.NE, 0, mstype.int32), input_columns=["input_mask"])
    dataset = dataset.map(operations=ops.Duplicate(), input_columns=["input_ids"],
                          output_columns=["input_ids", "segment_ids"],
                          column_order=["input_ids", "input_mask", "segment_ids", "label_ids"])
    dataset = dataset.map(operations=ops.Fill(0), input_columns=["segment_ids"])
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create mindrecord")
    parser.add_argument("--data_dir", type=str, default="", help="dataset path")
    parser.add_argument("--vocab_file", type=str, default="", help="Vocab file path")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--save_path", type=str, default="./my.mindrecord", help="Path to save mindrecord")
    parser.add_argument("--label2id", type=str, default="",
                        help="Label2id file path, must be set for cluener2020 task")
    args_opt = parser.parse_args()
    if args_opt.label2id == "":
        raise ValueError("label2id should not be empty")
    labels_list = []
    with open(args_opt.label2id) as f:
        for tag in f:
            labels_list.append(tag.strip())
    tag_to_index = list(convert_labels_to_index(labels_list).keys())
    ds = process_msra_clue_dataset(args_opt.data_dir, tag_to_index, args_opt.vocab_file, args_opt.max_seq_len)
    ds.save(args_opt.save_path)
