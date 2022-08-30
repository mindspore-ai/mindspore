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
import copy
import numpy as np
import mindspore.dataset.text as text
import mindspore.dataset as ds
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType

VOCAB_FILE = "../data/dataset/test_sentencepiece/vocab.txt"
DATA_FILE = "../data/dataset/testTokenizerData/sentencepiece_tokenizer.txt"


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
    expect = ["▁", "I", "▁use", "▁MindSpore", "▁", "to", "▁", "t", "r", "a", "i", "n", "▁", "m", "y", "▁model", "."]
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
    expect = [3, 41, 59, 53, 3, 29, 3, 6, 12, 99, 7, 10, 3, 11, 20, 45, 19]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_file_to_str():
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
    expect = ["▁", "I", "▁use", "▁MindSpore", "▁", "to", "▁", "t", "r", "a", "i", "n", "▁", "m", "y", "▁model", "."]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_file_to_int():
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
    expect = [3, 41, 59, 53, 3, 29, 3, 6, 12, 99, 7, 10, 3, 11, 20, 45, 19]
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
    expect = ["▁", "I", "▁use", "▁MindSpore", "▁", "to", "▁", "t", "r", "a", "i", "n", "▁", "m", "y", "▁model", "."]
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
    expect = ["▁", "I", "▁use", "▁MindSpore", "▁", "to", "▁", "t", "r", "a", "i", "n", "▁", "m", "y", "▁model", "."]
    for i in dataset_zip.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def concat_test(dataset):
    dataset_1 = copy.deepcopy(dataset)
    dataset = dataset.concat(dataset_1)
    expect = ["▁", "I", "▁use", "▁MindSpore", "▁", "to", "▁", "t", "r", "a", "i", "n", "▁", "m", "y", "▁model", "."]
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


if __name__ == "__main__":
    test_sentence_piece_tokenizer_callable()
    test_from_vocab_to_str_unigram()
    test_from_vocab_to_str_bpe()
    test_from_vocab_to_str_char()
    test_from_vocab_to_str_word()
    test_from_vocab_to_int()
    test_from_file_to_str()
    test_from_file_to_int()
    test_build_from_dataset()
    test_with_zip_concat()
