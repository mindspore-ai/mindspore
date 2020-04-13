#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-12-07 17:03
"""
import json
import tokenization
import collections

import numpy as np
from mindspore.mindrecord import FileWriter

# pylint: skip-file

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def process_one_example(tokenizer, label2id, text, label, max_seq_len=128):
    # textlist = text.split(' ')
    # labellist = label.split(' ')
    textlist = list(text)
    labellist = list(label)
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                print("some unknown token...")
                labels.append(labels[0])
    # tokens = tokenizer.tokenize(example.text)  -2 的原因是因为序列需要加一个句首和句尾标志
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
        labels = labels[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # [CLS] [SEP] 可以为 他们构建标签，或者 统一到某个标签，反正他们是不变的，基本不参加训练 即：x-l 永远不变
    label_ids.append(0)  # label2id["[CLS]"]
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(0)  # label2id["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(label_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids, label_ids)
    return feature


def prepare_mindrecord_data(tokenizer, max_seq_len, label2id, path, out_path):
    """
        生成训练数据， *.mindrecord, 单标签分类模型, 随机打乱数据
    """
    writer = FileWriter(out_path)

    data_schema = {"input_ids": {"type": "int64", "shape": [-1]},
                   "input_mask": {"type": "int64", "shape": [-1]},
                   "segment_ids": {"type": "int64", "shape": [-1]},
                   "label_ids": {"type": "int64", "shape": [-1]}}
    writer.add_schema(data_schema, "CLUENER2020 schema")

    example_count = 0

    for line in open(path):
        if not line.strip():
            continue
        _ = json.loads(line.strip())
        len_ = len(_["text"])
        labels = ["O"] * len_
        for k, v in _["label"].items():
            for kk, vv in v.items():
                for vvv in vv:
                    span = vvv
                    s = span[0]
                    e = span[1] + 1
                    # print(s, e)
                    if e - s == 1:
                        labels[s] = "S_" + k
                    else:
                        labels[s] = "B_" + k
                        for i in range(s + 1, e - 1):
                            labels[i] = "M_" + k
                        labels[e - 1] = "E_" + k
            # print()
        # feature = process_one_example(tokenizer, label2id, row[column_name_x1], row[column_name_y],
        #                               max_seq_len=max_seq_len)
        feature = process_one_example(tokenizer, label2id, list(_["text"]), labels,
                                      max_seq_len=max_seq_len)

        features = collections.OrderedDict()
        # 序列标注任务
        features["input_ids"] = np.asarray(feature[0])
        features["input_mask"] = np.asarray(feature[1])
        features["segment_ids"] = np.asarray(feature[2])
        features["label_ids"] = np.asarray(feature[3])
        if example_count < 5:
            print("*** Example ***")
            print(_["text"])
            print(_["label"])
            print("input_ids: %s" % " ".join([str(x) for x in feature[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature[2]]))
            print("label: %s " % " ".join([str(x) for x in feature[3]]))

        writer.write_raw_data([features])
        example_count += 1

        # if example_count == 20:
        #     break
        if example_count % 3000 == 0:
            print(example_count)
    print("total example:", example_count)
    writer.commit()


if __name__ == "__main__":
    vocab_file = "./vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    label2id = json.loads(open("label2id.json").read())

    max_seq_len = 64

    prepare_mindrecord_data(tokenizer, max_seq_len, label2id, path="cluener_public/train.json",
                           out_path="data/train.mindrecord")
    prepare_mindrecord_data(tokenizer, max_seq_len, label2id, path="cluener_public/dev.json",
                           out_path="data/dev.mindrecord")
