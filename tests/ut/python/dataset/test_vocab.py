# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.dataset as ds
import mindspore.dataset.text as text
import mindspore.common.dtype as mstype
from mindspore import log as logger

# this file contains "home is behind the world head" each word is 1 line
DATA_FILE = "../data/dataset/testVocab/words.txt"
VOCAB_FILE = "../data/dataset/testVocab/vocab_list.txt"
SIMPLE_VOCAB_FILE = "../data/dataset/testVocab/simple_vocab_list.txt"


def test_get_vocab():
    """
    Feature: Python text.Vocab class
    Description: Test vocab() method of text.Vocab
    Expectation: Success.
    """
    logger.info("test tokens_to_ids")
    vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
    vocab_ = vocab.vocab()
    assert "<unk>" in vocab_ and "w1" in vocab_ and "w2" in vocab_ and "w3" in vocab_


def test_vocab_tokens_to_ids():
    """
    Feature: Python text.Vocab class
    Description: Test tokens_to_ids() method of text.Vocab
    Expectation: Success.
    """
    logger.info("test tokens_to_ids")
    vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)

    ids = vocab.tokens_to_ids(["w1", "w3"])
    assert ids == [1, 3]

    ids = vocab.tokens_to_ids(["w1", "w4"])
    assert ids == [1, -1]

    ids = vocab.tokens_to_ids("<unk>")
    assert ids == 0

    ids = vocab.tokens_to_ids("hello")
    assert ids == -1

    ids = vocab.tokens_to_ids(np.array(["w1", "w3"]))
    assert ids == [1, 3]

    ids = vocab.tokens_to_ids(np.array("w1"))
    assert ids == 1


def test_vocab_ids_to_tokens():
    """
    Feature: Python text.Vocab class
    Description: Test ids_to_tokens() method of text.Vocab
    Expectation: Success.
    """
    logger.info("test ids_to_tokens")
    vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)

    tokens = vocab.ids_to_tokens([2, 3])
    assert tokens == ["w2", "w3"]

    tokens = vocab.ids_to_tokens([2, 7])
    assert tokens == ["w2", ""]

    tokens = vocab.ids_to_tokens(0)
    assert tokens == "<unk>"

    tokens = vocab.ids_to_tokens(7)
    assert tokens == ""

    tokens = vocab.ids_to_tokens(np.array([2, 3]))
    assert tokens == ["w2", "w3"]

    tokens = vocab.ids_to_tokens(np.array(2))
    assert tokens == "w2"


def test_vocab_exception():
    """
    Feature: Python text.Vocab class
    Description: Test exceptions of text.Vocab
    Expectation: Raise RuntimeError when vocab is not initialized, raise TypeError when input is wrong.
    """
    vocab = text.Vocab()
    with pytest.raises(RuntimeError):
        vocab.ids_to_tokens(2)
    with pytest.raises(RuntimeError):
        vocab.tokens_to_ids(["w3"])

    vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
    with pytest.raises(TypeError):
        vocab.ids_to_tokens("abc")
    with pytest.raises(TypeError):
        vocab.ids_to_tokens([2, 1.2, "abc"])
    with pytest.raises(ValueError):
        vocab.ids_to_tokens(-2)

    with pytest.raises(TypeError):
        vocab.tokens_to_ids([1, "w3"])
    with pytest.raises(TypeError):
        vocab.tokens_to_ids(999)


def test_lookup_callable():
    """
    Feature: Python text.Vocab class
    Description: Test Lookup with text.Vocab as the argument
    Expectation: Output is equal to the expected output
    """
    logger.info("test_lookup_callable")
    vocab = text.Vocab.from_list(['深', '圳', '欢', '迎', '您'])
    lookup = text.Lookup(vocab)
    word = "迎"
    assert lookup(word) == 3


def test_from_list_tutorial():
    """
    Feature: Python text.Vocab class
    Description: Test from_list() method from text.Vocab basic usage tutorial
    Expectation: Output is equal to the expected output
    """
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
    """
    Feature: Python text.Vocab class
    Description: Test from_file() method from text.Vocab basic usage tutorial
    Expectation: Output is equal to the expected output
    """
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
    """
    Feature: Python text.Vocab class
    Description: Test from_dict() method from text.Vocab basic usage tutorial
    Expectation: Output is equal to the expected output
    """
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
    """
    Feature: Python text.Vocab class
    Description: Test from_dict() method from text.Vocab with invalid input
    Expectation: Error is raised as expected
    """
    try:
        vocab = text.Vocab.from_dict({"home": -1, "behind": 0})
        if not vocab:
            raise ValueError("Vocab is None")
    except ValueError as e:
        assert "is not within the required interval" in str(e)


def test_from_list():
    """
    Feature: Python text.Vocab class
    Description: Test from_list() method from text.Vocab with various valid input cases and invalid input
    Expectation: Output is equal to the expected output, except for invalid input cases where correct error is raised
    """
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype=np.str_),)

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
    """
    Feature: Python text.Vocab class
    Description: Test from_list() with and without empty string in the Lookup op where unknown_token=None
    Expectation: Output is equal to the expected output when "" in Lookup op and error is raised otherwise
    """
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
    """
    Feature: Python text.Vocab class
    Description: Test from_file() method from text.Vocab with various valid and invalid special_tokens and vocab_size
    Expectation: Output is equal to the expected output for valid parameters and error is raised otherwise
    """
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype=np.str_),)

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
    """
    Feature: Python text.Vocab class
    Description: Test Lookup op cast type with various valid and invalid data types
    Expectation: Output is equal to the expected output for valid data types and error is raised otherwise
    """
    def gen(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype=np.str_),)

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
    assert "tldr is not of type [<class 'mindspore._c_expression.typing.Type'>]" in test_config("unk", "tldr")
    assert "Lookup : The parameter data_type must be numeric including bool." in \
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
