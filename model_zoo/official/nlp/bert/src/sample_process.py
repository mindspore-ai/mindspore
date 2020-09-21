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

"""process txt"""

import re
from src.tokenization import convert_tokens_to_ids

def process_one_example_p(tokenizer, vocab, text, max_seq_len=128):
    """process one testline"""
    textlist = list(text)
    tokens = []
    for _, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for _, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = convert_tokens_to_ids(vocab, ntokens)
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

    feature = (input_ids, input_mask, segment_ids)
    return feature

def label_generation(text="", probs=None, tag_to_index=None):
    """generate label"""
    data = [text]
    probs = [probs]
    result = []
    label2id = tag_to_index
    id2label = [k for k, v in label2id.items()]

    for index, prob in enumerate(probs):
        for v in prob[1:len(data[index]) + 1]:
            result.append(id2label[int(v)])

    labels = {}
    start = None
    index = 0
    for _, t in zip("".join(data), result):
        if re.search("^[BS]", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    labels[label] = {te_: [[start, index - 1]]}
            start = index
        if re.search("^O", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    labels[label] = {te_: [[start, index - 1]]}
            start = None
        index += 1
    if start is not None:
        label = result[start][2:]
        if labels.get(label):
            te_ = text[start:index]
            labels[label][te_] = [[start, index - 1]]
        else:
            te_ = text[start:index]
            labels[label] = {te_: [[start, index - 1]]}
    return labels
