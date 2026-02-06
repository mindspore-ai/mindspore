# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test transform - vocab"""

import os
import platform
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
import mindspore.common.dtype as mstype
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"

VOCAB_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                          "vocab_file.txt")
VOCAB_DUPLICATE_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                                    "vocab_file_duplicate.txt")
VOCAB_FILE_MUL = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                              "vocab_file_mul.txt")
VOCAB_WORDS_MUL = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                               "words_mul.txt")
VOCAB_FILE_NULL = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                               "vocab_file_null.txt")
VOCAB_FILE_ZH = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                             "words_chinese.txt")
VOCAB_WORDS_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                                "words.txt")
VOCAB_FILE0 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                           "from_dataset", "words_repeated.txt")
VOCAB_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                           "from_dataset", "words.txt")
VOCAB_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile", "testVocab",
                           "from_dataset", "words_capitalized.txt")
DATA_FILE = "../data/dataset/testVocab/words.txt"
VOCAB_FILE3 = "../data/dataset/testVocab/vocab_list.txt"
SIMPLE_VOCAB_FILE = "../data/dataset/testVocab/simple_vocab_list.txt"


def test_vocab_operation_01():
    """
    Feature: Test vocab
    Description: Test parameters list, dict, dataset
    Expectation: success
    """
    # Test vocab from list contains chinese
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_FILE_ZH, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [0, 4, 3, 1, 2]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind], ind
        ind += 1

    # Test vocab from list contains Chinese, English, symbols, numbers
    vocab = text.Vocab.from_list("with 007 符号 数字 ? 英文 < 中文 &gt;& &test； ## copy the".split(" "))
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_MUL, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [0, 4, 11, 2, 10, 9, 12]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind]
        ind += 1

    # Test vocab from file vocab_size greater than the actual data entered
    vocab = text.Vocab.from_file(VOCAB_FILE, ",", 100)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [13, 14, 15, 16, 12, 17, 18, 19]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind]
        ind += 1

    # Test vocab from dict
    vocab = text.Vocab.from_dict({"With": 3, "allowed": 2, "the": 4, "password": 5, "<unk>": 6})
    lookup = text.Lookup(vocab, "With")
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [3, 3, 2, 3, 4, 3, 5, 3]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind]
        ind += 1

    # Test vocab from dict: ID is 0
    vocab_dict = {"With": 3, "allowed": 2, "the": 0, "password": 5, "to": 6}
    vocab = text.Vocab.from_dict(vocab_dict)
    lookup = text.Lookup(vocab, "allowed")
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [3, 2, 2, 2, 0, 2, 5, 2]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind]
        ind += 1

    # Test vocab from dict: worf ID is 1
    vocab_dict = {"With": 3, "allowed": 2, "the": 1, "password": 5, "to": 6}
    vocab = text.Vocab.from_dict(vocab_dict)
    lookup = text.Lookup(vocab, "to")
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [3, 6, 2, 6, 1, 6, 5, 6]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind]
        ind += 1

    # Test vocab from dict: worf ID is Discontinuous
    vocab = text.Vocab.from_dict({"With": 3, "allowed": 2, "the": 4, "password": 7})
    lookup = text.Lookup(vocab, "password")
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [3, 7, 2, 7, 4, 7, 7, 7]
    for d in data.create_dict_iterator(output_numpy=True):
        assert d["text"] == res[ind]
        ind += 1

    # Test vocab from dataset: repeated words
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: no repeated words
    data = ds.TextFileDataset(VOCAB_FILE1, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 1, 0, 2, 6, 3, 5, 4]


def test_vocab_operation_02():
    """
    Feature: Test vocab
    Description: Test parameters dataset with freq_range, top_k
    Expectation: success
    """
    # Test vocab from dataset: no repeated words and some words is capitalized
    data = ds.TextFileDataset(VOCAB_FILE2, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [0, 2, 1, 3, 7, 4, 6, 5]

    # Test vocab from dataset: freq_range is (2, 3)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(2, 3), top_k=None)
    data = data.map(operations=text.Lookup(vocab, unknown_token="allowed"), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]

    # Test vocab from dataset: freq_range is (1, 2)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(1, 2), top_k=None)
    data = data.map(operations=text.Lookup(vocab, unknown_token="allowed"), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [6, 1, 0, 0, 0, 5, 2, 4, 0, 3, 0]

    # Test vocab from dataset: freq_range is (1, 10)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(1, 10), top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: freq_range is (1, 11)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(1, 11), top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: freq_range is (0, 3)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(0, 3), top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: freq_range is (2, 2)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(2, 2), top_k=None)
    data = data.map(operations=text.Lookup(vocab, unknown_token="allowed"), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Test vocab from dataset: freq_range is (1, None)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(1, None), top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: freq_range is (None, 3)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(None, 3), top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: freq_range is (None, None)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(None, None), top_k=None)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: top_k is 3
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=3)
    data = data.map(operations=text.Lookup(vocab, unknown_token="allowed"), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0]


def test_vocab_operation_03():
    """
    Feature: Test vocab
    Description: Test parameters dataset with top_k, freq_range and ids_to_tokens, tokens_to_ids
    Expectation: success
    """
    # Test vocab from dataset: top_k is 1
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=1)
    data = data.map(operations=text.Lookup(vocab, unknown_token="by"), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Test vocab from dataset: top_k is 100
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=100)
    data = data.map(operations=text.Lookup(vocab), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [7, 2, 1, 1, 0, 6, 3, 5, 0, 4, 0]

    # Test vocab from dataset: freq_range is (0, 3) and top_k is 3
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=(0, 3), top_k=3)
    data = data.map(operations=text.Lookup(vocab, unknown_token="allowed"), input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(output_numpy=True):
        res.append(d["text"].item())
    assert res == [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0]

    # ids_to_tokens test input list
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    token = vocab.ids_to_tokens([0, 1, 2, 2, 8, 128])
    assert token == ['重复', '符号', '数字', '数字', '', '']

    # ids_to_tokens test input num
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    token = vocab.ids_to_tokens(4)
    assert token == '中文'

    # ids_to_tokens test input 10
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    token = vocab.ids_to_tokens(10)
    assert token == ''

    # ids_to_tokens test input -1
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    token = vocab.ids_to_tokens(2147483647)
    assert token == ''

    # tokens_to_ids test input list
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    ids = vocab.tokens_to_ids(["00", "重复", " 重复", "重 复", "重复 ", "英文", "英文 中文"])
    assert ids == [-1, 0, -1, -1, -1, 3, -1]

    # tokens_to_ids test input str
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    ids = vocab.tokens_to_ids("重复")
    assert ids == 0

    # tokens_to_ids test input out str
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    ids = vocab.tokens_to_ids("重复符号")
    assert ids == -1


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
    vocab = text.Vocab.from_file(VOCAB_FILE3, ",", None, ["<pad>", "<unk>"], True)
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


def test_vocab_exception_01():
    """
    Feature: Test vocab
    Description: Test parameters with exception1
    Expectation: success
    """
    # Test vocab from list is empty
    voc = []
    vocab = text.Vocab.from_list(voc)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [1, 1, 1, 1, 1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="doesn't exist in vocab and no unknown token is specified"):
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind], ind
            ind += 1

    # Test vocab from list contains repeated words
    with pytest.raises(ValueError, match="word_list contains duplicate word: the."):
        text.Vocab.from_list("keyboard-interactive client the With "
                             "authentications allowed by password the".split(" "))

    # Test vocab from list contains repeated words
    voc = ("test", "with", "home")
    with pytest.raises(TypeError, match=r"Argument word_list with value \(\'test\', \'with\', \'home\'\) is not"
                                        r" of type \[\<class \'list\'\>\]."):
        text.Vocab.from_list(voc)

    # Test vocab from_file with file does not exist
    if platform.system() == "Windows":
        with pytest.raises(RuntimeError, match="from_file: fail to open"):
            text.Vocab.from_file(VOCAB_FILE_NULL, ",")
    else:
        with pytest.raises(RuntimeError, match="Get real path failed"):
            text.Vocab.from_file(VOCAB_FILE_NULL, ",")

    # Test vocab from_file without  parameter
    with pytest.raises(TypeError, match="missing a required argument: 'file_path'"):
        text.Vocab.from_file()

    # Test vocab from_file with file_path is empty string
    if platform.system() == "Windows":
        with pytest.raises(RuntimeError, match="from_file: fail to open"):
            text.Vocab.from_file("")
    else:
        with pytest.raises(RuntimeError, match="Get real path failed"):
            text.Vocab.from_file("")

    # Test vocab from_file without delimiter parameter
    vocab = text.Vocab.from_file(VOCAB_FILE)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [13, 14, 15, 16, 12, 17, 18, 19]
    with pytest.raises(RuntimeError, match="doesn't exist in vocab and no unknown token is specified"):
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind]
            ind += 1

    # Test vocab from file vocab_size=-1
    with pytest.raises(ValueError, match="Input vocab_size must be greater than 0"):
        text.Vocab.from_file(VOCAB_FILE, ",", -1)

    # Test vocab from file vocab_size=0
    with pytest.raises(ValueError, match="Input vocab_size must be greater than 0"):
        text.Vocab.from_file(VOCAB_FILE, ",", 0)

    # Test vocab from file vocab_size=1
    vocab = text.Vocab.from_file(VOCAB_FILE, ",", 1)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [1, 1, 1, 1, 1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="doesn't exist in vocab and no unknown token is specified"):
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind]
            ind += 1

    # Test vocab from file vocab_size=2
    vocab = text.Vocab.from_file(VOCAB_FILE, ",", 2)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    with pytest.raises(RuntimeError, match="doesn't exist in vocab and no unknown token is specified"):
        res = [1, 1, 1, 1, 1, 1, 1, 1]
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind]
            ind += 1

    # Test vocab from file: contains symbols, numbers, English, Chinese
    vocab = text.Vocab.from_file(VOCAB_FILE_MUL)
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(VOCAB_WORDS_MUL, shuffle=False)
    data = data.map(operations=lookup, input_columns=["text"])
    ind = 0
    res = [0, 1, 1, 2, 10, 9, 18]
    with pytest.raises(RuntimeError, match="doesn't exist in vocab and no unknown token is specified"):
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind]
            ind += 1


def test_vocab_exception_02():
    """
    Feature: Test vocab
    Description: Test parameters with exception2
    Expectation: success
    """
    # Test vocab from file contains duplicate words
    with pytest.raises(RuntimeError, match="duplicate word:the"):
        text.Vocab.from_file(VOCAB_DUPLICATE_FILE, ",")

    # Test vocab from dict ,param is a list
    vocab_dict = [5, 6, 7, 8, 4, 3, 9, 2]
    with pytest.raises(TypeError, match=r"Argument word_dict with value \[5, 6, 7, 8, 4, 3, 9, 2\] is not of type"
                                        r" \[\<class \'dict\'\>\]."):
        text.Vocab.from_dict(vocab_dict)

    # Test vocab from dict: word ID is -1
    vocab_dict = {"With": 3, "allowed": 2, "the": -1, "password": 5, "to": 6}
    with pytest.raises(ValueError, match=r"Input word_id is not within the required interval of \[0, 2147483647\]"):
        text.Vocab.from_dict(vocab_dict)

    # Test vocab from dict is empty
    vocab = text.Vocab.from_dict({})
    lookup = text.Lookup(vocab, "")
    data = ds.TextFileDataset(VOCAB_WORDS_FILE, shuffle=False)
    data = data.map(operations=lookup, input_columns=['text'])
    with pytest.raises(RuntimeError, match="doesn't exist in vocab"):
        for _ in data.create_dict_iterator(output_numpy=True):
            pass

    # Test vocab from dict: type of word is int
    vocab_dict = {55: 3, "allowed": 2, "the": 4, "password": 5, "to": 6}
    with pytest.raises(TypeError, match=r"Argument word with value 55 is not of type \[\<class \'str\'\>\]."):
        text.Vocab.from_dict(vocab_dict)

    # Test vocab from dict: type of word is int
    vocab_list = ("int", "tp")
    vocab_dict = {vocab_list: 3, "allowed": 2, "the": 4, "password": 5, "to": 6}
    with pytest.raises(TypeError, match=r"Argument word with value \('int', 'tp'\) is not of type "
                                        r"\[\<class \'str\'\>\]."):
        text.Vocab.from_dict(vocab_dict)

    # Test vocab from dataset: freq_range is (-1, 3)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(ValueError, match=""):
        text.Vocab.from_dataset(data, "text", freq_range=(-1, 3), top_k=None)

    # Test vocab from dataset: freq_range is (1.0, 2.0)
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(ValueError,
                       match="freq_range needs to be either None or a tuple of 2 integers or an int and a None"):
        text.Vocab.from_dataset(data, "text", freq_range=(1.0, 2.0), top_k=None)

    # Test vocab from dataset: freq_range is [1, 3]
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(TypeError,
                       match=r"Argument freq_range with value \[1, 3\] is not of type \[\<class \'tuple\'\>\]."):
        text.Vocab.from_dataset(data, "text", freq_range=[1, 3], top_k=None)

    # Test vocab from dataset: freq_range is 'chekq& 2d*'
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(TypeError,
                       match=r"Argument freq_range with value chekq\& 2d\* is not of type \[\<class \'tuple\'\>\]."):
        text.Vocab.from_dataset(data, "text", freq_range='chekq& 2d*', top_k=None)

    # Test vocab from dataset: freq_range is 3
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(TypeError,
                       match=r"Argument freq_range with value 3 is not of type \[\<class \'tuple\'\>\]."):
        text.Vocab.from_dataset(data, "text", freq_range=3, top_k=None)

    # Test vocab from dataset: top_k is 0
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(ValueError, match="Input top_k must be greater than 0"):
        text.Vocab.from_dataset(data, "text", freq_range=None, top_k=0)

    # Test vocab from dataset: top_k is 2.0
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(TypeError, match=r"Argument top_k with value 2.0 is not of type \[\<class \'int\'\>,"
                                        r" \<class \'NoneType\'\>\]."):
        text.Vocab.from_dataset(data, "text", freq_range=None, top_k=2.0)

    # Test vocab from dataset: top_k is 'chekq& 2d*'
    data = ds.TextFileDataset(VOCAB_FILE0, shuffle=False)
    with pytest.raises(TypeError, match=r"Argument top_k with value chekq\& 2d\* is not of type \[\<class \'int\'\>,"
                                        r" \<class \'NoneType\'\>\]."):
        text.Vocab.from_dataset(data, "text", freq_range=None, top_k='chekq& 2d*')

    # ids_to_tokens 2147483648
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(ValueError, match=r"Input ids is not within the required interval of \[0, 2147483647\]."):
        vocab.ids_to_tokens(2147483648)

    # ids_to_tokens test input float
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"Argument ids with value 1.0 is not of type \[<class"
                                        r" 'int'>, <class 'list'>, <class 'numpy.ndarray'>\], "
                                        r"but got <class 'float'>."):
        vocab.ids_to_tokens(1.0)


def test_vocab_exception_03():
    """
    Feature: Test vocab
    Description: Test parameters with exception3
    Expectation: success
    """
    # ids_to_tokens test input list[str]
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    if platform.system() == "Windows":
        with pytest.raises(TypeError, match=r"Argument ids\[0\] with value 0 is not of "
                                            r"type \[<class 'int'>, <class 'numpy.int32'>\], but got <class 'str'>."):
            vocab.ids_to_tokens(["0", "1", 2])
    else:
        with pytest.raises(TypeError, match=r"Argument ids\[0\] with value 0 is not of "
                                            r"type \[<class 'int'>, <class 'numpy.int64'>\], but got <class 'str'>."):
            vocab.ids_to_tokens(["0", "1", 2])

    # ids_to_tokens test input tuple
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"Argument ids with value \(0, 1, 2\) is not of type \[<class"
                                        r" 'int'>, <class 'list'>, <class 'numpy.ndarray'>\], "
                                        r"but got <class 'tuple'>."):
        vocab.ids_to_tokens((0, 1, 2))

    # ids_to_tokens test input None
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"Argument ids with value None is not of type \[<class"
                                        r" 'int'>, <class 'list'>, <class 'numpy.ndarray'>\], "
                                        r"but got <class 'NoneType'>."):
        vocab.ids_to_tokens(None)

    # ids_to_tokens test no input
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"missing a required argument: 'ids'"):
        vocab.ids_to_tokens()

    # ids_to_tokens test two input
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"too many positional arguments"):
        vocab.ids_to_tokens(0, 1)

    # tokens_to_ids test input int
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"Argument tokens with value 1 is not of type \[<"
                                        r"class 'str'>, <class 'list'>, <class 'numpy.ndarray'>\], "
                                        r"but got <class 'int'>."):
        vocab.tokens_to_ids(1)

    # tokens_to_ids test input tuple
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"Argument tokens with value \('重复', '符号'\) is not of type"
                                        r" \[<class 'str'>, <class 'list'>, <class 'numpy.ndarray'>\], "
                                        r"but got <class 'tuple'>."):
        vocab.tokens_to_ids(("重复", "符号"))

    # tokens_to_ids test two input
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"too many positional arguments"):
        vocab.tokens_to_ids("重复", "符号")

    # tokens_to_ids test input None
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"Argument tokens with value None is not of type \[<class"
                                        r" 'str'>, <class 'list'>, <class 'numpy.ndarray'>\], "
                                        r"but got <class 'NoneType'>."):
        vocab.tokens_to_ids(None)

    # tokens_to_ids test no input
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(TypeError, match=r"missing a required argument: 'tokens'"):
        vocab.tokens_to_ids()

    # ids_to_tokens -1
    vocab = text.Vocab.from_list("重复 符号 数字 英文 中文".split(" "))
    with pytest.raises(ValueError, match=r"Input ids is not within the required interval of \[0, 2147483647\]."):
        vocab.ids_to_tokens(-1)
