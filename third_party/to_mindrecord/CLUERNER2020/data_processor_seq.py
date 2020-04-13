#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-12-07 17:03
"""
import json
import tokenization
import collections
import tensorflow as tf


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


def prepare_tf_record_data(tokenizer, max_seq_len, label2id, path, out_path):
    """
        生成训练数据， tf.record, 单标签分类模型, 随机打乱数据
    """
    writer = tf.python_io.TFRecordWriter(out_path)
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

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        # 序列标注任务
        features["input_ids"] = create_int_feature(feature[0])
        features["input_mask"] = create_int_feature(feature[1])
        features["segment_ids"] = create_int_feature(feature[2])
        features["label_ids"] = create_int_feature(feature[3])
        if example_count < 5:
            print("*** Example ***")
            print(_["text"])
            print(_["label"])
            print("input_ids: %s" % " ".join([str(x) for x in feature[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature[2]]))
            print("label: %s " % " ".join([str(x) for x in feature[3]]))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        example_count += 1

        # if example_count == 20:
        #     break
        if example_count % 3000 == 0:
            print(example_count)
    print("total example:", example_count)
    writer.close()


if __name__ == "__main__":
    vocab_file = "./vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    label2id = json.loads(open("label2id.json").read())

    max_seq_len = 64

    prepare_tf_record_data(tokenizer, max_seq_len, label2id, path="data/thuctc_train.json",
                           out_path="data/train.tf_record")
    prepare_tf_record_data(tokenizer, max_seq_len, label2id, path="data/thuctc_valid.json",
                           out_path="data/dev.tf_record")
