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
# ==============================================================================
import copy
import numpy as np
import mindspore.dataset.text as text
import mindspore.dataset as ds
from mindspore.dataset.text import SentencePieceModel, to_str, SPieceTokenizerOutType

VOCAB_FILE = "../data/dataset/test_sentencepiece/botchan.txt"
DATA_FILE = "../data/dataset/testTokenizerData/sentencepiece_tokenizer.txt"


def test_sentence_piece_tokenizer_callable():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    data = '123'
    assert np.array_equal(tokenizer(data), ['▁', '12', '3'])


def test_from_vocab_to_str_UNIGRAM():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ['▁I', '▁sa', 'w', '▁a', '▁girl', '▁with', '▁a', '▁te', 'les', 'co', 'pe', '.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_str_BPE():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.BPE, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ['▁I', '▁saw', '▁a', '▁girl', '▁with', '▁a', '▁te', 'les', 'c', 'ope', '.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_str_CHAR():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.CHAR, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ['▁', 'I', '▁', 's', 'a', 'w', '▁', 'a', '▁', 'g', 'i', 'r', 'l', '▁', 'w', 'i', 't', 'h',\
              '▁', 'a', '▁', 't', 'e', 'l', 'e', 's', 'c', 'o', 'p', 'e', '.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_str_WORD():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.WORD, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ['▁I', '▁saw', '▁a', '▁girl', '▁with', '▁a', '▁telescope.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_vocab_to_int():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.INT)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = [6, 329, 183, 8, 945, 23, 8, 3783, 4382, 4641, 1405, 4]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_file_to_str():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    text.SentencePieceVocab.save_model(vocab, "./", "m.model")
    tokenizer = text.SentencePieceTokenizer("./m.model", out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ['▁I', '▁sa', 'w', '▁a', '▁girl', '▁with', '▁a', '▁te', 'les', 'co', 'pe', '.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_from_file_to_int():
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    text.SentencePieceVocab.save_model(vocab, "./", "m.model")
    tokenizer = text.SentencePieceTokenizer("./m.model", out_type=SPieceTokenizerOutType.INT)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = [6, 329, 183, 8, 945, 23, 8, 3783, 4382, 4641, 1405, 4]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for key, value in enumerate(ret):
            assert value == expect[key]


def test_build_from_dataset():
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    vocab = text.SentencePieceVocab.from_dataset(data, ["text"], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer)
    expect = ['▁I', '▁sa', 'w', '▁a', '▁girl', '▁with', '▁a', '▁te', 'les', 'co', 'pe', '.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def apply_func(dataset):
    input_columns = ['text']
    output_columns = ['text2']
    dataset = dataset.rename(input_columns, output_columns)
    return dataset


def zip_test(dataset):
    dataset_1 = copy.deepcopy(dataset)
    dataset_2 = copy.deepcopy(dataset)
    dataset_1 = dataset_1.apply(apply_func)
    dataset_zip = ds.zip((dataset_1, dataset_2))
    expect = ['▁I', '▁sa', 'w', '▁a', '▁girl', '▁with', '▁a', '▁te', 'les', 'co', 'pe', '.']
    for i in dataset_zip.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]


def concat_test(dataset):
    dataset_1 = copy.deepcopy(dataset)
    dataset = dataset.concat(dataset_1)
    expect = ['▁I', '▁sa', 'w', '▁a', '▁girl', '▁with', '▁a', '▁te', 'les', 'co', 'pe', '.']
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = to_str(i["text"])
        for key, value in enumerate(ret):
            assert value == expect[key]

def test_with_zip_concat():
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    vocab = text.SentencePieceVocab.from_dataset(data, ["text"], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    dataset = dataset.map(operations=tokenizer, num_parallel_workers=2)
    zip_test(dataset)
    concat_test(dataset)


if __name__ == "__main__":
    test_sentence_piece_tokenizer_callable()
    test_from_vocab_to_str_UNIGRAM()
    test_from_vocab_to_str_BPE()
    test_from_vocab_to_str_CHAR()
    test_from_vocab_to_str_WORD()
    test_from_vocab_to_int()
    test_from_file_to_str()
    test_from_file_to_int()
    test_build_from_dataset()
    test_with_zip_concat()
