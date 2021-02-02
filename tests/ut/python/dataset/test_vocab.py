# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import mindspore.common.dtype as mstype
from mindspore import log as logger

# this file contains "home is behind the world head" each word is 1 line
DATA_FILE = "../data/dataset/testVocab/words.txt"
VOCAB_FILE = "../data/dataset/testVocab/vocab_list.txt"
SIMPLE_VOCAB_FILE = "../data/dataset/testVocab/simple_vocab_list.txt"


def test_lookup_callable():
    """
    Test lookup is callable
    """
    logger.info("test_lookup_callable")
    vocab = text.Vocab.from_list(['深', '圳', '欢', '迎', '您'])
    lookup = text.Lookup(vocab)
    word = "迎"
    assert lookup(word) == 3

def test_from_list_tutorial():
    vocab = text.Vocab.from_list("home IS behind the world ahead !".split(" "), ["<pad>", "<unk>"], True)
    lookup = text.Lookup(vocab, "<unk>")
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [2, 1, 4, 5, 6, 7]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_file_tutorial():
    vocab = text.Vocab.from_file(VOCAB_FILE, ",", None, ["<pad>", "<unk>"], True)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [10, 11, 12, 15, 13, 14]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_dict_tutorial():
    vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
    lookup = text.Lookup(vocab, "<unk>")  # any unknown token will be mapped to the id of <unk>
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    res = [3, 6, 2, 4, 5, 6]
    ind = 0
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_dict_exception():
    try:
        vocab = text.Vocab.from_dict({"home": -1, "behind": 0})
        if not vocab:
            raise ValueError("Vocab is None")
    except ValueError as e:
        assert "is not within the required interval" in str(e)


def test_from_list():
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype='S'),)

    def test_config(lookup_str, vocab_input, special_tokens, special_first, unknown_token):
        try:
            vocab = text.Vocab.from_list(vocab_input, special_tokens, special_first)
            data = ds.GeneratorDataset(gen(lookup_str), column_names=["text"])
            data = data.map(operations=text.Lookup(vocab, unknown_token), input_columns=["text"])
            res = []
            for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(d["text"].item())
            return res
        except (ValueError, RuntimeError, TypeError) as e:
            return str(e)

    # test basic default config, special_token=None, unknown_token=None
    assert test_config("w1 w2 w3", ["w1", "w2", "w3"], None, True, None) == [0, 1, 2]
    # test normal operations
    assert test_config("w1 w2 w3 s1 s2 ephemeral", ["w1", "w2", "w3"], ["s1", "s2"], True, "s2") == [2, 3, 4, 0, 1, 1]
    assert test_config("w1 w2 w3 s1 s2", ["w1", "w2", "w3"], ["s1", "s2"], False, "s2") == [0, 1, 2, 3, 4]
    assert test_config("w3 w2 w1", ["w1", "w2", "w3"], None, True, "w1") == [2, 1, 0]
    assert test_config("w3 w2 w1", ["w1", "w2", "w3"], None, False, "w1") == [2, 1, 0]
    # test unknown token lookup
    assert test_config("w1 un1 w3 un2", ["w1", "w2", "w3"], ["<pad>", "<unk>"], True, "<unk>") == [2, 1, 4, 1]
    assert test_config("w1 un1 w3 un2", ["w1", "w2", "w3"], ["<pad>", "<unk>"], False, "<unk>") == [0, 4, 2, 4]

    # test exceptions
    assert "doesn't exist in vocab." in test_config("un1", ["w1"], [], False, "unk")
    assert "doesn't exist in vocab and no unknown token is specified." in test_config("un1", ["w1"], [], False, None)
    assert "doesn't exist in vocab" in test_config("un1", ["w1"], [], False, None)
    assert "word_list contains duplicate" in test_config("w1", ["w1", "w1"], [], True, "w1")
    assert "special_tokens contains duplicate" in test_config("w1", ["w1", "w2"], ["s1", "s1"], True, "w1")
    assert "special_tokens and word_list contain duplicate" in test_config("w1", ["w1", "w2"], ["s1", "w1"], True, "w1")
    assert "is not of type" in test_config("w1", ["w1", "w2"], ["s1"], True, 123)


def test_from_list_lookup_empty_string():
    # "" is a valid word in vocab, which can be looked up by LookupOp
    vocab = text.Vocab.from_list("home IS behind the world ahead !".split(" "), ["<pad>", ""], True)
    lookup = text.Lookup(vocab, "")
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [2, 1, 4, 5, 6, 7]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert d["text"] == res[ind], ind
        ind += 1

    # unknown_token of LookUp is None, it will convert to std::nullopt in C++,
    # so it has nothing to do with "" in vocab and C++ will skip looking up unknown_token
    vocab = text.Vocab.from_list("home IS behind the world ahead !".split(" "), ["<pad>", ""], True)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    try:
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    except RuntimeError as e:
        assert "token: \"is\" doesn't exist in vocab and no unknown token is specified" in str(e)


def test_from_file():
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype='S'),)

    def test_config(lookup_str, vocab_size, special_tokens, special_first):
        try:
            vocab = text.Vocab.from_file(SIMPLE_VOCAB_FILE, vocab_size=vocab_size, special_tokens=special_tokens,
                                         special_first=special_first)
            data = ds.GeneratorDataset(gen(lookup_str), column_names=["text"])
            data = data.map(operations=text.Lookup(vocab, "s2"), input_columns=["text"])
            res = []
            for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(d["text"].item())
            return res
        except ValueError as e:
            return str(e)

    # test special tokens are prepended
    assert test_config("w1 w2 w3 s1 s2 s3", None, ["s1", "s2", "s3"], True) == [3, 4, 5, 0, 1, 2]
    # test special tokens are appended
    assert test_config("w1 w2 w3 s1 s2 s3", None, ["s1", "s2", "s3"], False) == [0, 1, 2, 8, 9, 10]
    # test special tokens are prepended when not all words in file are used
    assert test_config("w1 w2 w3 s1 s2 s3", 3, ["s1", "s2", "s3"], False) == [0, 1, 2, 3, 4, 5]
    # text exception special_words contains duplicate words
    assert "special_tokens contains duplicate" in test_config("w1", None, ["s1", "s1"], True)
    # test exception when vocab_size is negative
    assert "Input vocab_size must be greater than 0" in test_config("w1 w2", 0, [], True)
    assert "Input vocab_size must be greater than 0" in test_config("w1 w2", -1, [], True)


def test_lookup_cast_type():
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype='S'),)

    def test_config(lookup_str, data_type=None):
        try:
            vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            data = ds.GeneratorDataset(gen(lookup_str), column_names=["text"])
            # if data_type is None, test the default value of data_type
            op = text.Lookup(vocab, "<unk>") if data_type is None else text.Lookup(vocab, "<unk>", data_type)
            data = data.map(operations=op, input_columns=["text"])
            res = []
            for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(d["text"])
            return res[0].dtype
        except (ValueError, RuntimeError, TypeError) as e:
            return str(e)

    # test result is correct
    assert test_config("w1", mstype.int8) == np.dtype("int8")
    assert test_config("w2", mstype.int32) == np.dtype("int32")
    assert test_config("w3", mstype.int64) == np.dtype("int64")
    assert test_config("unk", mstype.float32) != np.dtype("int32")
    assert test_config("unk") == np.dtype("int32")
    # test exception, data_type isn't the correct type
    assert "tldr is not of type (<class 'mindspore._c_expression.typing.Type'>,)" in test_config("unk", "tldr")
    assert "Lookup does not support a string to string mapping, data_type can only be numeric." in \
           test_config("w1", mstype.string)


if __name__ == '__main__':
    test_lookup_callable()
    test_from_dict_exception()
    test_from_list_tutorial()
    test_from_file_tutorial()
    test_from_dict_tutorial()
    test_from_list()
    test_from_file()
    test_lookup_cast_type()
