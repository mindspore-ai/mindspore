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
SIMPLE_VOCAB_FILE = "../data/dataset/testVocab/simple_vocab_list.txt"


def test_from_list_tutorial():
    vocab = text.Vocab.from_list("home IS behind the world ahead !".split(" "), ["<pad>", "<unk>"], True)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(input_columns=["text"], operations=lookup)
    ind = 0
    res = [2, 1, 4, 5, 6, 7]
    for d in data.create_dict_iterator():
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_file_tutorial():
    vocab = text.Vocab.from_file(VOCAB_FILE, ",", None, ["<pad>", "<unk>"], True)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(input_columns=["text"], operations=lookup)
    ind = 0
    res = [10, 11, 12, 15, 13, 14]
    for d in data.create_dict_iterator():
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_dict_tutorial():
    vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
    lookup = text.Lookup(vocab, 6)  # default value is -1
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(input_columns=["text"], operations=lookup)
    res = [3, 6, 2, 4, 5, 6]
    ind = 0
    for d in data.create_dict_iterator():
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_list():
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype='S'),)

    def test_config(lookup_str, vocab_input, special_tokens, special_first):
        try:
            vocab = text.Vocab.from_list(vocab_input, special_tokens, special_first)
            data = ds.GeneratorDataset(gen(lookup_str), column_names=["text"])
            data = data.map(input_columns=["text"], operations=text.Lookup(vocab))
            res = []
            for d in data.create_dict_iterator():
                res.append(d["text"].item())
            return res
        except ValueError as e:
            return str(e)

    # test normal operations
    assert test_config("w1 w2 w3 s1 s2", ["w1", "w2", "w3"], ["s1", "s2"], True) == [2, 3, 4, 0, 1]
    assert test_config("w1 w2 w3 s1 s2", ["w1", "w2", "w3"], ["s1", "s2"], False) == [0, 1, 2, 3, 4]
    assert test_config("w3 w2 w1", ["w1", "w2", "w3"], None, True) == [2, 1, 0]
    assert test_config("w3 w2 w1", ["w1", "w2", "w3"], None, False) == [2, 1, 0]

    # test exceptions
    assert "word_list contains duplicate" in test_config("w1", ["w1", "w1"], [], True)
    assert "special_tokens contains duplicate" in test_config("w1", ["w1", "w2"], ["s1", "s1"], True)
    assert "special_tokens and word_list contain duplicate" in test_config("w1", ["w1", "w2"], ["s1", "w1"], True)


def test_from_file():
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype='S'),)

    def test_config(lookup_str, special_tokens, special_first):
        try:
            vocab = text.Vocab.from_file(SIMPLE_VOCAB_FILE, special_tokens=special_tokens, special_first=special_first)
            data = ds.GeneratorDataset(gen(lookup_str), column_names=["text"])
            data = data.map(input_columns=["text"], operations=text.Lookup(vocab))
            res = []
            for d in data.create_dict_iterator():
                res.append(d["text"].item())
            return res
        except ValueError as e:
            return str(e)

    assert test_config("w1 w2 w3", ["s1", "s2", "s3"], True) == [3, 4, 5]
    assert test_config("w1 w2 w3", ["s1", "s2", "s3"], False) == [0, 1, 2]
    assert "special_tokens contains duplicate" in test_config("w1", ["s1", "s1"], True)


if __name__ == '__main__':
    test_from_list_tutorial()
    test_from_file_tutorial()
    test_from_dict_tutorial()
    test_from_list()
    test_from_file()
