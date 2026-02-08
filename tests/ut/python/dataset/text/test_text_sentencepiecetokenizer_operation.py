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
"""text transform - sentencepiecetokenizer"""

import copy
import os

import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType


VOCAB_FILE = "../data/dataset/test_sentencepiece/vocab.txt"
DATA_FILE = "../data/dataset/testTokenizerData/sentencepiece_tokenizer.txt"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
VOCAB_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data/testTextFile/textfile/test_sentencepiece/botchan.txt")


def test_sentencepiecetokenizer_operation_01():
    """
    Feature: SentencePieceTokenizer op
    Description: Test SentencePieceTokenizer op with different input types
    Expectation: Successfully tokenize strings using SentencePiece model
    """
    # mode is SentencePieceVocab, out_type=SPieceTokenizerOutType.STRING, input is str
    out_type = SPieceTokenizerOutType.STRING
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE2], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=out_type)
    data = "我爱我的祖国"
    res = tokenizer(data)
    assert (res == ['▁', '我爱我的祖国']).all()

    # mode is SentencePieceVocab, out_type=SPieceTokenizerOutType.STRING, input is list(str)
    out_type = SPieceTokenizerOutType.STRING
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE2], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=out_type)
    data = ["哈喽", "我爱我的祖国"]
    res = []
    for i in data:
        res.append(tokenizer(i))
    assert (res[0] == ['▁', '哈喽']).all()
    assert (res[1] == ['▁', '我爱我的祖国']).all()


def test_sentence_piece_tokenizer_callable():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with eager mode
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    data = "123"
    assert np.array_equal(tokenizer(data), ["▁", "1", "23"])


def test_from_vocab_to_str_unigram():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with UNIGRAM model
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ["▁", "I", "▁", "u", "s", "e", "▁MindSpore", "▁to", "▁", "t", "r", "a", "in", "▁", "m", "y", "▁model", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_str_bpe():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with BPE model
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.BPE, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ["▁", "I", "▁", "u", "s", "e", "▁", "M", "in", "d", "S", "p", "or", "e", "▁t", "o", "▁t", "ra", "in", "▁m",
              "y", "▁m", "ode", "l", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_str_char():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with CHAR model
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.CHAR, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ["▁", "I", "▁", "u", "s", "e", "▁", "M", "i", "n", "d", "S", "p", "o", "r", "e", "▁", "t", "o", "▁", "t",
              "r", "a", "i", "n", "▁", "m", "y", "▁", "m", "o", "d", "e", "l", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_str_word():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with WORD model
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.WORD, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ["▁I", "▁use", "▁MindSpore", "▁to", "▁train▁my▁model."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_int():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with out_type equal to int
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.INT)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = [3, 39, 3, 12, 5, 4, 47, 33, 3, 6, 97, 99, 24, 3, 14, 25, 45, 20]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


@pytest.mark.parametrize("cleanup_tmp_file", ["./m.model"], indirect=True)
def test_from_file_to_str(cleanup_tmp_file):
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with out_type equal to string
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    text.SentencePieceVocab.save_model(vocab, "./", "m.model")
    tokenizer = text.SentencePieceTokenizer("./m.model", out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ["▁", "I", "▁", "u", "s", "e", "▁MindSpore", "▁to", "▁", "t", "r", "a", "in", "▁", "m", "y", "▁model", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


@pytest.mark.parametrize("cleanup_tmp_file", ["./m.model"], indirect=True)
def test_from_file_to_int(cleanup_tmp_file):
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer while loading vocab model from file
    Expectation: Output is equal to the expected value
    """
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    text.SentencePieceVocab.save_model(vocab, "./", "m.model")
    tokenizer = text.SentencePieceTokenizer("./m.model", out_type=SPieceTokenizerOutType.INT)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = [3, 39, 3, 12, 5, 4, 47, 33, 3, 6, 97, 99, 24, 3, 14, 25, 45, 20]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_build_from_dataset():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer while loading vocab model from dataset
    Expectation: Output is equal to the expected value
    """
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    vocab = text.SentencePieceVocab.from_dataset(data, ["text"], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ["▁", "I", "▁", "u", "s", "e", "▁MindSpore", "▁to", "▁", "t", "r", "a", "in", "▁", "m", "y", "▁model", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def apply_func(dataset):
    input_columns = ["text"]
    output_columns = ["text2"]
    dataset = dataset.rename(input_columns, output_columns)
    return dataset


def zip_test(dataset):
    dataset_1 = copy.deepcopy(dataset)
    dataset_2 = copy.deepcopy(dataset)
    dataset_1 = dataset_1.apply(apply_func)
    dataset_zip = ds.zip((dataset_1, dataset_2))
    expect = ["▁", "I", "▁", "u", "s", "e", "▁MindSpore", "▁to", "▁", "t", "r", "a", "in", "▁", "m", "y", "▁model", "."]
    for i in dataset_zip.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def concat_test(dataset):
    dataset_1 = copy.deepcopy(dataset)
    dataset = dataset.concat(dataset_1)
    expect = ["▁", "I", "▁", "u", "s", "e", "▁MindSpore", "▁to", "▁", "t", "r", "a", "in", "▁", "m", "y", "▁model", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_with_zip_concat():
    """
    Feature: SentencePieceTokenizer
    Description: Test SentencePieceTokenizer with zip and concat operations
    Expectation: Output is equal to the expected value
    """
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    vocab = text.SentencePieceVocab.from_dataset(data, ["text"], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer, num_parallel_workers=2)
    zip_test(dataset)
    concat_test(dataset)


def test_sentencepiecetokenizer_exception_01():
    """
    Feature: SentencePieceTokenizer op
    Description: Test SentencePieceTokenizer op with missing or invalid parameters
    Expectation: Raise expected exceptions for missing required parameters
    """
    # Test no mode
    out_type = SPieceTokenizerOutType.STRING
    with pytest.raises(TypeError, match="missing a required argument: 'mode'"):
        text.SentencePieceTokenizer(out_type=out_type)

    # Test no out_type
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE2], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    with pytest.raises(TypeError, match="missing a required argument: 'out_type'"):
        text.SentencePieceTokenizer(vocab)

    # Test no para

    with pytest.raises(TypeError, match="missing a required argument: 'mode'"):
        text.SentencePieceTokenizer()

    # mode is SentencePieceVocab, out_type=SPieceTokenizerOutType.STRING, input is int
    out_type = SPieceTokenizerOutType.STRING
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE2], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=out_type)
    data = 1234567
    with pytest.raises(RuntimeError, match="SentencePieceTokenizer: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = tokenizer(data)
