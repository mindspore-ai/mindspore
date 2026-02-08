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
"""text transform - sentencepiecevocab"""

import os
import pytest
from mindspore.dataset import text
import mindspore.dataset as ds
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType


TEST_DATA_DATASET_FUNC ="../data/dataset/"


VOCAB_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data/testTextFile/textfile/test_sentencepiece/botchan.txt")
DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                  "SentencePieceTokenizer/sentencepiece_tokenizer.txt"))


def test_sentencepiecevocab_operation_01():
    """
    Feature: SentencePieceVocab op
    Description: Test SentencePieceVocab op with different character_coverage and input types
    Expectation: Tokenizer successfully processes string and list inputs
    """
    # character_coverage = 1.0,input is str
    vocab_size = 4000
    character_coverage = 1.0
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], vocab_size=vocab_size,
                                              character_coverage=character_coverage,
                                              model_type=model_type, params=params)
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    data = "1234567"
    res = tokenizer(data)
    assert (res == ['▁', '12', '3', '4', '5', '6', '7']).all()

    # character_coverage = 0.98, input is list(str)
    vocab_size = 4000
    character_coverage = 0.98
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], vocab_size=vocab_size,
                                              character_coverage=character_coverage,
                                              model_type=model_type, params=params)
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    data = ["123", "456"]
    res = []
    for i in data:
        res.append(tokenizer(i))
    assert (res[0] == ['▁', '123']).all()
    assert (res[1] == ['▁', '456']).all()


def test_sentencepiecevocab_exception_01():
    """
    Feature: SentencePieceVocab op
    Description: Test SentencePieceVocab op with invalid parameters (vocab_size, character_coverage, etc.)
    Expectation: Raise expected exceptions for invalid inputs
    """
    # test from_file, no vocab_size
    character_coverage = 0.9995
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    with pytest.raises(TypeError, match="missing a required argument: 'vocab_size'"):
        text.SentencePieceVocab.from_file([VOCAB_FILE], character_coverage=character_coverage,
                                          model_type=model_type, params=params)

    # test from_file, vocab_size = -1
    vocab_size = -1
    character_coverage = 0.9995
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    with pytest.raises(ValueError, match=r"Input vocab_size is not within the required interval of \[0, 4294967295\]"):
        text.SentencePieceVocab.from_file([VOCAB_FILE], vocab_size=vocab_size,
                                          character_coverage=character_coverage,
                                          model_type=model_type, params=params)

    # test from_file, vocab_size = "a"
    vocab_size = "a"
    character_coverage = 0.9995
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    with pytest.raises(TypeError, match=r"Argument vocab_size with value a is not of type \[\<class \'int\'\>\]."):
        text.SentencePieceVocab.from_file([VOCAB_FILE], vocab_size=vocab_size,
                                          character_coverage=character_coverage,
                                          model_type=model_type, params=params)

    # test from_file, vocab_size = 5000.0
    vocab_size = 5000.0
    character_coverage = 0.9995
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    with pytest.raises(TypeError,
                       match=r"Argument vocab_size with value 5000.0 is not of type \[\<class \'int\'\>\]."):
        text.SentencePieceVocab.from_file([VOCAB_FILE], vocab_size=vocab_size,
                                          character_coverage=character_coverage,
                                          model_type=model_type, params=params)

    # Test from_dataset, col_names is a tuple
    col_names = ""
    vocab_size = 5000
    character_coverage = 0.9995
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    with pytest.raises(TypeError, match=r"is not of type .*list.*"):
        text.SentencePieceVocab.from_dataset(data, col_names=col_names, vocab_size=vocab_size,
                                             character_coverage=character_coverage, model_type=model_type,
                                             params=params)

    # Test from_dataset, col_names = ""
    col_names = ""
    vocab_size = 5000
    character_coverage = 0.9995
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    with pytest.raises(TypeError, match=r"is not of type .*list.*"):
        text.SentencePieceVocab.from_dataset(data, col_names=col_names, vocab_size=vocab_size,
                                             character_coverage=character_coverage, model_type=model_type,
                                             params=params)

    # Test from_dataset, character_coverage = 2
    col_names = [""]
    vocab_size = 5000
    character_coverage = 2
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    with pytest.raises(TypeError,
                       match=r"Argument character_coverage with value 2 is not of type \[\<class \'float\'\>\]."):
        text.SentencePieceVocab.from_dataset(data, col_names=col_names, vocab_size=vocab_size,
                                             character_coverage=character_coverage, model_type=model_type,
                                             params=params)

    # Test from_dataset, character_coverage = "a"
    col_names = [""]
    vocab_size = 5000
    character_coverage = "a"
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    with pytest.raises(TypeError,
                       match=r"Argument character_coverage with value a is not of type \[\<class \'float\'\>\]."):
        text.SentencePieceVocab.from_dataset(data, col_names=col_names, vocab_size=vocab_size,
                                             character_coverage=character_coverage, model_type=model_type,
                                             params=params)


def test_sentencepiecevocab_exception_02():
    """
    Feature: SentencePieceVocab op
    Description: Test SentencePieceVocab op with invalid character_coverage and input data types
    Expectation: Raise expected exceptions for invalid parameters
    """
    # Test from_dataset, character_coverage = 2.0
    col_names = [""]
    vocab_size = 5000
    character_coverage = 2.0
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    data = ds.TextFileDataset(VOCAB_FILE, shuffle=False)
    with pytest.raises(RuntimeError,
                       match="[trainer_spec.character_coverage() >= 0.98 && trainer_spec.character_coverage() <= 1.0]"):
        text.SentencePieceVocab.from_dataset(data, col_names=col_names, vocab_size=vocab_size,
                                             character_coverage=character_coverage, model_type=model_type,
                                             params=params)

    # character_coverage = 0.98, input is int
    vocab_size = 4000
    character_coverage = 0.98
    model_type = SentencePieceModel.UNIGRAM
    params = {}
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], vocab_size=vocab_size,
                                              character_coverage=character_coverage,
                                              model_type=model_type, params=params)
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
    data = 123456
    with pytest.raises(RuntimeError, match="SentencePieceTokenizer: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = tokenizer(data)
