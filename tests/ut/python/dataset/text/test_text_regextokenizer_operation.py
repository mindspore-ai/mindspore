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
"""text transform - regextokenizer"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC,
                         "text_data/testTextFile/textfile/testTokenizerData/RegexTokenizer/1.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexTokenizer/2.txt")
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexTokenizer/3.txt")
REGEX_TOKENIZER_FILE = "../data/dataset/testTokenizerData/regex_tokenizer.txt"


def test_regextokenizer_operation_01():
    """
    Feature: RegexTokenizer op
    Description: Test RegexTokenizer op with different delimiter and keep_delim patterns
    Expectation: Successfully tokenize strings based on regex patterns
    """
    # Test RegexTokenizer, keep_delim_pattern = "\\s+"
    delim_pattern = "\\s+"
    keep_delim_pattern = "\\s+"
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = [['Welcome', ' ', 'to', ' ', 'Shenzhen!']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        out_text.append(data)
    assert expect_str == out_text

    # Test RegexTokenizer, delim_pattern = "[\\p{P}|\\p{S}]+"
    delim_pattern = "[\\p{P}|\\p{S}]+"
    keep_delim_pattern = ""
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = [['12', '36']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        out_text.append(data)
    assert expect_str == out_text

    # Test RegexTokenizer, keep_delim_pattern = "[\\p{P}|\\p{S}]+"
    delim_pattern = "[\\p{P}|\\p{S}]+"
    keep_delim_pattern = "[\\p{P}|\\p{S}]+"
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = [['12', '￥+', '36', '￥=?']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        out_text.append(data)
    assert expect_str == out_text

    # Test RegexTokenizer, delim_pattern = "[\\p{N}]+"
    delim_pattern = "[\\p{N}]+"
    keep_delim_pattern = ""
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = [["￥+", "￥=?"]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        out_text.append(data)
    assert expect_str == out_text

    # Test RegexTokenizer, keep_delim_pattern = "", with_offsets = True
    delim_pattern = "\\s+"
    keep_delim_pattern = ""
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                                    with_offsets=True)
    dataset = dataset.map(operations=replace_op, input_columns=["text"],
                          output_columns=["token", "offsets_start", "offsets_limit"])
    dataset = dataset.project(columns=["token", "offsets_start", "offsets_limit"])
    out_token = []
    expect_str = [['Welcome', 'to', 'Shenzhen!']]
    expected_offsets_start = [[0, 8, 11]]
    expected_offsets_limit = [[7, 10, 20]]
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token'].tolist()
        out_token.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1
    assert expect_str == out_token

    # Test RegexTokenizer, keep_delim_pattern = "", with_offsets = True
    delim_pattern = "to"
    keep_delim_pattern = "to"
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                                    with_offsets=True)
    data = 'Welcometo hhhh'
    res = replace_op(data)
    assert (res[0] == ['Welcome', 'to', ' hhhh']).all()
    assert (res[1] == [0, 7, 9]).all()
    assert (res[2] == [7, 9, 14]).all()


def test_regextokenizer_operation_02():
    """
    Feature: RegexTokenizer op
    Description: Test RegexTokenizer op with offsets and different delimiter patterns
    Expectation: Successfully tokenize and provide offset information
    """
    # Test RegexTokenizer, keep_delim_pattern = "", with_offsets = True
    delim_pattern = r"\d+"
    keep_delim_pattern = " "
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                                    with_offsets=True)
    data = '我1爱2西3安4'
    res = replace_op(data)
    assert (res[0] == ['我', '爱', '西', '安']).all()
    assert (res[1] == [0, 4, 8, 12]).all()
    assert (res[2] == [3, 7, 11, 15]).all()

    # Test RegexTokenizer, keep_delim_pattern = "", with_offsets = True
    delim_pattern = r"\s"
    keep_delim_pattern = " "
    replace_op = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                                    with_offsets=False)
    data = '!@#$ %^&*'
    res = replace_op(data)
    assert (res == ['!@#$', ' ', '%^&*']).all()


def test_regex_tokenizer_default():
    """
    Feature: RegexTokenizer op
    Description: Test RegexTokenizer op with default parameters
    Expectation: Output is equal to the expected output
    """

    def regex_tokenizer(first, last, expect_str, delim_pattern, keep_delim_pattern):
        dataset = ds.TextFileDataset(REGEX_TOKENIZER_FILE, shuffle=False)
        if first > 1:
            dataset = dataset.skip(first - 1)
        if last >= first:
            dataset = dataset.take(last - first + 1)
        tokenizer_op = text.RegexTokenizer(delim_pattern, keep_delim_pattern)
        dataset = dataset.map(operations=tokenizer_op)
        out_text = []
        count = 0
        for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            token = i['text'].tolist()
            np.testing.assert_array_equal(token, expect_str[count])
            count += 1
            out_text.append(token)
        logger.info("Out:", out_text)
        logger.info("Exp:", expect_str)

    regex_tokenizer(1, 1, [['Welcome', 'to', 'Shenzhen!']], "\\s+", "")
    regex_tokenizer(1, 1, [['Welcome', ' ', 'to', ' ', 'Shenzhen!']], "\\s+", "\\s+")
    regex_tokenizer(2, 2, [['北', '京', '欢', '迎', '您', '!Welcome to Beijing!']], r"\p{Han}", r"\p{Han}")
    regex_tokenizer(3, 3, [['12', '￥+', '36', '￥=?']], r"[\p{P}|\p{S}]+", r"[\p{P}|\p{S}]+")
    regex_tokenizer(3, 3, [['12', '36']], r"[\p{P}|\p{S}]+", "")
    regex_tokenizer(3, 3, [['￥+', '￥=?']], r"[\p{N}]+", "")


def test_regex_tokenizer_with_offsets():
    """
    Feature: RegexTokenizer op
    Description: Test RegexTokenizer op with with_offsets=True
    Expectation: Output is equal to the expected output
    """

    def regex_tokenizer(first, last, expect_str, expected_offsets_start, expected_offsets_limit, delim_pattern,
                        keep_delim_pattern):
        dataset = ds.TextFileDataset(REGEX_TOKENIZER_FILE, shuffle=False)
        if first > 1:
            dataset = dataset.skip(first - 1)
        if last >= first:
            dataset = dataset.take(last - first + 1)
        tokenizer_op = text.RegexTokenizer(delim_pattern, keep_delim_pattern, with_offsets=True)
        dataset = dataset.map(operations=tokenizer_op, input_columns=['text'],
                              output_columns=['token', 'offsets_start', 'offsets_limit'])
        out_text = []
        count = 0
        for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            token = i['token'].tolist()
            np.testing.assert_array_equal(token, expect_str[count])
            np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
            np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
            count += 1
            out_text.append(token)
        logger.info("Out:", out_text)
        logger.info("Exp:", expect_str)

    regex_tokenizer(1, 1, [['Welcome', 'to', 'Shenzhen!']], [[0, 8, 11]], [[7, 10, 20]], "\\s+", "")
    regex_tokenizer(1, 1, [['Welcome', ' ', 'to', ' ', 'Shenzhen!']], [[0, 7, 8, 10, 11]], [[7, 8, 10, 11, 20]],
                    "\\s+", "\\s+")
    regex_tokenizer(2, 2, [['北', '京', '欢', '迎', '您', '!Welcome to Beijing!']], [[0, 3, 6, 9, 12, 15]],
                    [[3, 6, 9, 12, 15, 35]], r"\p{Han}", r"\p{Han}")
    regex_tokenizer(3, 3, [['12', '￥+', '36', '￥=?']], [[0, 2, 6, 8]], [[2, 6, 8, 13]],
                    r"[\p{P}|\p{S}]+", r"[\p{P}|\p{S}]+")
    regex_tokenizer(3, 3, [['12', '36']], [[0, 6]], [[2, 8]], r"[\p{P}|\p{S}]+", "")
    regex_tokenizer(3, 3, [['￥+', '￥=?']], [[2, 8]], [[6, 13]], r"[\p{N}]+", "")


def test_regextokenizer_exception_01():
    """
    Feature: RegexTokenizer op
    Description: Test RegexTokenizer op with invalid parameter types
    Expectation: Raise expected exceptions for invalid parameter types
    """
    # Test RegexTokenizer, keep_delim_pattern = " ", with_offsets = True,delim_pattern = True
    delim_pattern = True
    keep_delim_pattern = " "
    with pytest.raises(TypeError, match="Wrong input type for delim_pattern, should be string"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets=False)

    # Test RegexTokenizer, keep_delim_pattern = " ", with_offsets = True,delim_pattern = 0
    delim_pattern = 0
    keep_delim_pattern = " "
    with pytest.raises(TypeError, match="Wrong input type for delim_pattern, should be string"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets=False)

    # Test RegexTokenizer, keep_delim_pattern = False, with_offsets = True
    delim_pattern = r"\s"
    keep_delim_pattern = False
    with pytest.raises(TypeError, match="Wrong input type for keep_delim_pattern, should be string"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets=False)

    # Test RegexTokenizer, keep_delim_pattern = -1, with_offsets = True
    delim_pattern = r"\s"
    keep_delim_pattern = -1
    with pytest.raises(TypeError, match="Wrong input type for keep_delim_pattern, should be string"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets=False)

    # Test RegexTokenizer, keep_delim_pattern = " ", with_offsets = -1
    delim_pattern = r"\s"
    keep_delim_pattern = " "
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets=-1)

    # Test RegexTokenizer, keep_delim_pattern = " ", with_offsets = "True"
    delim_pattern = r"\s"
    keep_delim_pattern = " "
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets="True")

    # Test RegexTokenizer, keep_delim_pattern = " ", with_offsets = [True]
    delim_pattern = r"\s"
    keep_delim_pattern = " "
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.RegexTokenizer(delim_pattern=delim_pattern, keep_delim_pattern=keep_delim_pattern,
                               with_offsets=[True])
