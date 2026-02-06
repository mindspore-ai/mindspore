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
"""text transform - whitespacetokenizer"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset import text


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC,
                         "text_data/testTextFile/textfile/testTokenizerData/testWhitespaceTokenizer/english.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testWhitespaceTokenizer/chinese.txt"))
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testWhitespaceTokenizer/mixed.txt")
DATA_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testWhitespaceTokenizer/numbers.txt"))
DATA_FILE4 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testWhitespaceTokenizer/space.txt")
DATA_FILE5 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testWhitespaceTokenizer/punctuation.txt"))
DATA_FILE6 = "../data/dataset/testTokenizerData/1.txt"
NORMALIZE_FILE = "../data/dataset/testTokenizerData/normalize.txt"
REGEX_REPLACE_FILE = "../data/dataset/testTokenizerData/regex_replace.txt"
REGEX_TOKENIZER_FILE = "../data/dataset/testTokenizerData/regex_tokenizer.txt"


def test_whitespacetokenizer_operation_01():
    """
    Feature: Test whitespacetokenizer
    Description: Test parameters which English, Chinese, number, ...
    Expectation: success
    """
    # Test WhitespaceTokenizer,English string
    whitespace_strs = [["Hello,", "welcome", "to", "Beijing!"]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens

    # Test WhitespaceTokenizer,Chinese string
    whitespace_strs = [["北京", "欢迎您！"]]
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    tokenizer = text.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens

    # Test WhitespaceTokenizer,contains Chinese and English
    whitespace_strs = [["I'm", "Chinese,", "我喜欢English!"]]
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    tokenizer = text.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens

    # Test WhitespaceTokenizer,numbers
    whitespace_strs = [["123", "456", "78", "9"]]
    dataset = ds.TextFileDataset(DATA_FILE3, shuffle=False)
    tokenizer = text.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens

    # Test WhitespaceTokenizer,special characters
    whitespace_strs = [["@#", "%^&", "*!()"]]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    tokenizer = text.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens

    # Test WhitespaceTokenizer,English string, with_offsets = True
    whitespace_strs = [["Hello,", "welcome", "to", "Beijing!"]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.WhitespaceTokenizer(with_offsets=True)
    dataset = dataset.map(input_columns=["text"], output_columns=["token", "offsets_start", "offsets_limit"],
                          operations=tokenizer)
    dataset = dataset.project(["token", "offsets_start", "offsets_limit"])
    count = 0
    expected_offsets_start = [[0, 7, 15, 18]]
    expected_offsets_limit = [[6, 14, 17, 26]]
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens

    # Test WhitespaceTokenizer,English string, with_offsets = True
    data = "Hello,"
    tokenizer = text.WhitespaceTokenizer(with_offsets=True)
    res = tokenizer(data)
    assert res[2] == 6

    # Test WhitespaceTokenizer,English string, with_offsets = True
    data = "!@#$%^&"
    tokenizer = text.WhitespaceTokenizer(with_offsets=True)
    res = tokenizer(data)
    assert res[2] == 7


def test_whitespacetokenizer_operation_02():
    """
    Feature: Test whitespacetokenizer
    Description: Test parameters with_offsets
    Expectation: success
    """
    # Test WhitespaceTokenizer,English string, with_offsets = True
    data = ["!@#$%^&", "white, ", "space,", "token"]
    tokenizer = text.WhitespaceTokenizer(with_offsets=True)
    res = []
    for i in data:
        res.append(tokenizer(i))
    assert res[0][2] == 7
    assert res[1][2] == 6
    assert res[2][2] == 6
    assert res[3][2] == 5

    # Test WhitespaceTokenizer,English string, with_offsets = False
    data = "Hello,"
    tokenizer = text.WhitespaceTokenizer(with_offsets=False)
    res = tokenizer(data)
    assert res == ["Hello,"]

    # Test WhitespaceTokenizer,English string, with_offsets = False
    data = ["one,", "two,", "three,"]
    tokenizer = text.WhitespaceTokenizer(with_offsets=False)
    res = []
    for i in data:
        res.append(tokenizer(i))
    assert res[0] == ['one,']
    assert res[1] == ['two,']
    assert res[2] == ['three,']


def test_whitespace_tokenizer_default():
    """
    Feature: WhitespaceTokenizer op
    Description: Test WhitespaceTokenizer op with default parameters
    Expectation: Output is equal to the expected output
    """
    whitespace_strs = [["Welcome", "to", "Beijing!"],
                       ["北京欢迎您！"],
                       ["我喜欢English!"],
                       [""]]
    dataset = ds.TextFileDataset(DATA_FILE6, shuffle=False)
    tokenizer = text.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text'].tolist()
        tokens.append(token)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens


def test_whitespace_tokenizer_with_offsets():
    """
    Feature: WhitespaceTokenizer op
    Description: Test WhitespaceTokenizer op with with_offsets=True
    Expectation: Output is equal to the expected output
    """
    whitespace_strs = [["Welcome", "to", "Beijing!"],
                       ["北京欢迎您！"],
                       ["我喜欢English!"],
                       [""]]
    dataset = ds.TextFileDataset(DATA_FILE6, shuffle=False)
    tokenizer = text.WhitespaceTokenizer(with_offsets=True)
    dataset = dataset.map(operations=tokenizer, input_columns=['text'],
                          output_columns=['token', 'offsets_start', 'offsets_limit'])
    tokens = []
    expected_offsets_start = [[0, 8, 11], [0], [0], [0]]
    expected_offsets_limit = [[7, 10, 19], [18], [17], [0]]
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count += 1

    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens


def test_whitespacetokenizer_exception_01():
    """
    Feature: Test whitespacetokenizer
    Description: Test parameters with exception
    Expectation: success
    """
    # Test WhitespaceTokenizer,English string, with_offsets = False
    data = 12345
    tokenizer = text.WhitespaceTokenizer(with_offsets=False)
    with pytest.raises(RuntimeError, match="WhitespaceTokenizerOp: the input shape should be scalar and the "
                                           "input datatype should be string."):
        _ = tokenizer(data)

    # Test WhitespaceTokenizer,English string, with_offsets = False
    data = ["hi", "why", "year"]
    tokenizer = text.WhitespaceTokenizer(with_offsets=False)
    with pytest.raises(RuntimeError, match="WhitespaceTokenizerOp: the input shape should be scalar and the "
                                           "input datatype should be string."):
        _ = tokenizer(data)

    # Test WhitespaceTokenizer,English string, with_offsets = "True"
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.WhitespaceTokenizer(with_offsets="True")

    # Test WhitespaceTokenizer,English string, with_offsets = 0
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.WhitespaceTokenizer(with_offsets=0)

    # Test WhitespaceTokenizer,English string, with_offsets = [True, False]
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.WhitespaceTokenizer(with_offsets=[True, False])
