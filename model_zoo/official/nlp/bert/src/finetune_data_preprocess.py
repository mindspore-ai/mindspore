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
from lxml import etree

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


def process_msra(data_file, class_filter=None, split_begin=None, split_end=None):
    """
    Data pre-process for MSRA dataset
    Args:
        data_file (path): The original dataset file path.
        class_filter (list of str): Only tags within the class_filter will be counted unless the list is None.
        split_begin (float): Only data after split_begin part will be counted. Used for split dataset
                     into training and evaluation subsets if needed.
        split_end (float): Only data before split_end part will be counted. Used for split dataset
                     into training and evaluation subsets if needed.
    """
    tree = etree.parse(data_file)
    root = tree.getroot()
    print("original dataset length: ", len(root))
    dataset_size = len(root)
    beg = 0 if split_begin is None or not 0 <= split_begin <= 1.0 else int(dataset_size * split_begin)
    end = dataset_size if split_end is None or not 0 <= split_end <= 1.0 else int(dataset_size * split_end)
    print("preporcessed dataset_size: ", end - beg)
    for i in range(beg, end):
        sentence = root[i]
        tags = []
        content = ""
        for phrases in sentence:
            labeled_words = [word for word in phrases]
            if labeled_words:
                for words in phrases:
                    name = words.tag
                    label = words.get("TYPE")
                    words = words.text
                    if not words:
                        continue
                    content += words
                    if class_filter and name not in class_filter:
                        tags += ["O" for _ in words]
                    else:
                        length = len(words)
                        labels = ["S_"] if length == 1 else ["B_"] + ["M_" for i in range(length - 2)] + ["E_"]
                        tags += [ele + label for ele in labels]
            else:
                phrases = phrases.text
                if phrases:
                    content += phrases
                    tags += ["O" for ele in phrases]
        if len(content) != len(tags):
            raise ValueError("Mismathc length of content: ", len(content), " and label: ", len(tags))
        yield (np.array("".join(content)), np.array(list(tags)))


def process_ner_msra_dataset(data_dir, label_list, bert_vocab_path, max_seq_len=128, class_filter=None,
                             split_begin=None, split_end=None):
    """Process MSRA dataset"""
    ### Loading MSRA from CLUEDataset
    dataset = ds.GeneratorDataset(process_msra(data_dir, class_filter, split_begin, split_end),
                                  column_names=['text', 'label'])

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
    parser.add_argument("--save_path", type=str, default="./my.mindrecord", help="Path to save mindrecord")
    parser.add_argument("--label2id", type=str, default="",
                        help="Label2id file path, please keep in mind that each label name should be consistent with"
                             "the type name labeled in the oroginal dataset file")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--class_filter", nargs='*', help="Specified classes will be counted, if empty all in counted")
    parser.add_argument("--split_begin", type=float, default=None, help="Specified subsets of data will be counted,"
                        "if not None, the data will counted begin from split_begin")
    parser.add_argument("--split_end", type=float, default=None, help="Specified subsets of data will be counted,"
                        "if not None, the data before split_end will be counted ")

    args_opt = parser.parse_args()
    if args_opt.label2id == "":
        raise ValueError("label2id should not be empty")
    labels_list = []
    with open(args_opt.label2id) as f:
        for tag in f:
            labels_list.append(tag.strip())
    tag_to_index = list(convert_labels_to_index(labels_list).keys())
    ds = process_ner_msra_dataset(args_opt.data_dir, tag_to_index, args_opt.vocab_file, args_opt.max_seq_len,
                                  args_opt.class_filter, args_opt.split_begin, args_opt.split_end)
    ds.save(args_opt.save_path)
