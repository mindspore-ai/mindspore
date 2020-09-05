# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================

import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.text as text

# this file contains "home is behind the world head" each word is 1 line
DATA_FILE = "../data/dataset/testVocab/words.txt"
VOCAB_FILE = "../data/dataset/testVocab/vocab_list.txt"
HMM_FILE = "../data/dataset/jiebadict/hmm_model.utf8"
MP_FILE = "../data/dataset/jiebadict/jieba.dict.utf8"


def test_on_tokenized_line():
    data = ds.TextFileDataset("../data/dataset/testVocab/lines.txt", shuffle=False)
    jieba_op = text.JiebaTokenizer(HMM_FILE, MP_FILE, mode=text.JiebaMode.MP)
    with open(VOCAB_FILE, 'r') as f:
        for line in f:
            word = line.split(',')[0]
            jieba_op.add_word(word)
    data = data.map(operations=jieba_op, input_columns=["text"])
    vocab = text.Vocab.from_file(VOCAB_FILE, ",", special_tokens=["<pad>", "<unk>"])
    lookup = text.Lookup(vocab, "<unk>")
    data = data.map(operations=lookup, input_columns=["text"])
    res = np.array([[10, 1, 11, 1, 12, 1, 15, 1, 13, 1, 14],
                    [11, 1, 12, 1, 10, 1, 14, 1, 13, 1, 15]], dtype=np.int32)
    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["text"], res[i])


def test_on_tokenized_line_with_no_special_tokens():
    data = ds.TextFileDataset("../data/dataset/testVocab/lines.txt", shuffle=False)
    jieba_op = text.JiebaTokenizer(HMM_FILE, MP_FILE, mode=text.JiebaMode.MP)
    with open(VOCAB_FILE, 'r') as f:
        for line in f:
            word = line.split(',')[0]
            jieba_op.add_word(word)

    data = data.map(operations=jieba_op, input_columns=["text"])
    vocab = text.Vocab.from_file(VOCAB_FILE, ",")
    lookup = text.Lookup(vocab, "not")
    data = data.map(operations=lookup, input_columns=["text"])
    res = np.array([[8, 0, 9, 0, 10, 0, 13, 0, 11, 0, 12],
                    [9, 0, 10, 0, 8, 0, 12, 0, 11, 0, 13]], dtype=np.int32)
    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["text"], res[i])


if __name__ == '__main__':
    test_on_tokenized_line()
    test_on_tokenized_line_with_no_special_tokens()
