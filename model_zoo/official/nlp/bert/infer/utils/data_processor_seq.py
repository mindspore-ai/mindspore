# Copyright 2021 Huawei Technologies Co., Ltd
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
process the cluener json
"""
import argparse
import collections
import json
import os

import numpy as np
import tokenization


def parse_args():
    """set parameters."""
    parser = argparse.ArgumentParser(description="cluner data preprocess")
    parser.add_argument("--infer_json", type=str, default="valid.json", help="the format of infer file is json.")
    parser.add_argument("--label_file", type=str, default="label.txt", help="the label of infer json.")
    parser.add_argument("--vocab_file", type=str, default="vocab.txt", help="the vocab file of chinese dataset.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="sentence length, default is 128.")
    parser.add_argument("--output_path", type=str, default="./data", help="the path of convert dataset.")

    args_opt = parser.parse_args()
    return args_opt


def convert_labels_to_index(label_file):
    """
    Convert label name to label_list which indices for NER task.
    Args:
        label_file: parameters content label path

    Returns:
        label2id: list id of label and each label extend four
    """
    with open(label_file) as f:
        label_list = f.readlines()

    label2id = collections.OrderedDict()
    label2id["O"] = 0
    prefix = ["S_", "B_", "M_", "E_"]
    index = 0
    for label in label_list:
        for pre in prefix:
            index += 1
            sub_label = pre + label.split('\n')[0]
            label2id[sub_label] = index
    return label2id


def process_one_example(tokenizer, label2id, text, label, max_seq_len=128):
    """
    convert the text and label to tensor, result:
    text: [101 3946 3419 4638 4413 7339 5303 754 ...]
    mask: [1 1 1 1 1 1 1 1 ...]
    segment_ids: [0 0 0 0 0 0 0 0 0 0 0 0 ...]
    label: [0 26 28  0  0  0  0 ...]
    """
    text_list = list(text)
    label_list = list(label)
    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                print("some unknown token...")
                labels.append(labels[0])
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
        labels = labels[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    # set the begin symbol of sentence
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label2id[labels[i]])
    # set the end symbol of sentence
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")

    feature = (input_ids, input_mask, segment_ids, label_ids)
    return feature


def get_all_path(output_path):
    """
    Args:
        output_path: save path of convert dataset
    Returns:
        the path of ids, mask, token, label
    """
    ids_path = os.path.join(output_path, "00_data")
    mask_path = os.path.join(output_path, "01_data")
    token_path = os.path.join(output_path, "02_data")
    label_path = os.path.join(output_path, "03_data")
    for path in [ids_path, mask_path, token_path, label_path]:
        os.makedirs(path, 0o755, exist_ok=True)

    return ids_path, mask_path, token_path, label_path


def get_label_from_json(line_json):
    """
    Args:
        line_json: the line content from json file
    Returns:
        labels: convert label to label tensor
    """
    text_len = len(line_json["text"])
    labels = ["O"] * text_len
    for k, v in line_json["label"].items():
        for _, vv in v.items():
            for span in vv:
                s = span[0]
                e = span[1] + 1
                if e - s == 1:
                    labels[s] = "S_" + k
                else:
                    labels[s] = "B_" + k
                    for i in range(s + 1, e - 1):
                        labels[i] = "M_" + k
                    labels[e - 1] = "E_" + k
    return labels


def prepare_cluener_data(tokenizer, max_seq_len, label2id, path, out_path):
    """
    Args:
        tokenizer: token class convert word to id according vocab file
        max_seq_len: the length of sentence
        label2id:  label list of word to id
        path: dataset path
        out_path: output path of convert result
    """
    output_ids, output_mask, output_token, output_label = get_all_path(out_path)
    example_count = 0
    for line in open(path):
        if not line.strip():
            continue
        line_json = json.loads(line.strip())
        labels = get_label_from_json(line_json)
        feature = process_one_example(tokenizer, label2id, list(line_json["text"]), labels,
                                      max_seq_len=max_seq_len)
        file_name = "cluener_bs" + "_" + str(example_count) + ".bin"
        ids_file_path = os.path.join(output_ids, file_name)
        np.array(feature[0], dtype=np.int32).tofile(ids_file_path)

        mask_file_path = os.path.join(output_mask, file_name)
        np.array(feature[1], dtype=np.int32).tofile(mask_file_path)

        token_file_path = os.path.join(output_token, file_name)
        np.array(feature[2], dtype=np.int32).tofile(token_file_path)

        label_file_path = os.path.join(output_label, file_name)
        np.array(feature[3], dtype=np.int32).tofile(label_file_path)
        print("*** Example ***")
        print(line_json["text"])
        print(line_json["label"])
        print("input_ids: %s" % " ".join([str(x) for x in feature[0]]))
        print("input_mask: %s" % " ".join([str(x) for x in feature[1]]))
        print("segment_ids: %s" % " ".join([str(x) for x in feature[2]]))
        print("label: %s " % " ".join([str(x) for x in feature[3]]))
        example_count += 1
        if example_count % 3000 == 0:
            print(example_count)
    print("total example:", example_count)


def run():
    """
    convert infer json to bin, each sentence is one file bin
    """
    args = parse_args()
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file)
    label2id = convert_labels_to_index(args.label_file)

    prepare_cluener_data(tokenizer, args.max_seq_len, label2id, path=args.infer_json,
                         out_path=args.output_path)


if __name__ == "__main__":
    run()
