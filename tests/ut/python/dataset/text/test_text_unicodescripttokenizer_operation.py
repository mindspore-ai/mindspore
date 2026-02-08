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
"""text transform - unicodescripttokenizer"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset import text


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                  "testUnicodeScriptTokenizer/cnandnum.txt"))
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testUnicodeScriptTokenizer/enandnum.txt"))
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testUnicodeScriptTokenizer/cnanden.txt"))
DATA_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testUnicodeScriptTokenizer/cnandpunctuation.txt"))
DATA_FILE4 = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData/"
                                                   "testUnicodeScriptTokenizer/enandpunctuation.txt"))
DATA_FILE5 = "../data/dataset/testTokenizerData/1.txt"
NORMALIZE_FILE = "../data/dataset/testTokenizerData/normalize.txt"
REGEX_REPLACE_FILE = "../data/dataset/testTokenizerData/regex_replace.txt"
REGEX_TOKENIZER_FILE = "../data/dataset/testTokenizerData/regex_tokenizer.txt"


def test_unicodescripttokenizer_operation_01():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op with different script types and keep_whitespace parameter
    Expectation: Successfully tokenize by Unicode script with optional whitespace handling
    """
    # Test UnicodeScriptTokenizer, English and numbers
    unicode_script_strs = [["Welcome", "to", "Beijing", "123"]]
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False)
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens

    # Test UnicodeScriptTokenizer, Chinese and punctuation
    unicode_script_strs = [["北京欢迎你", "！"]]
    dataset = ds.TextFileDataset(DATA_FILE3, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False)
    dataset = dataset.map(operations=tokenizer)

    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens

    # Test UnicodeScriptTokenizer, English and punctuation
    unicode_script_strs = [["Welcome", "to", "Beijing", "!!!&*>?"]]
    dataset = ds.TextFileDataset(DATA_FILE4, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False)
    dataset = dataset.map(operations=tokenizer)

    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens

    # Test keep_whitespace=True
    unicode_script_strs = [["北京", "123", "欢迎", " ", "你"]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=True)
    dataset = dataset.map(operations=tokenizer)

    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens

    # Test UnicodeScriptTokenizer, Chinese and numbers, with_offsets = True
    unicode_script_strs = [["北京", "123", "欢迎", "你"]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False, with_offsets=True)
    dataset = dataset.map(input_columns=["text"], output_columns=["token", "offsets_start", "offsets_limit"],
                          operations=tokenizer)
    dataset = dataset.project(columns=["token", "offsets_start", "offsets_limit"])

    tokens = []
    count = 0
    expected_offsets_start = [[0, 6, 9, 16]]
    expected_offsets_limit = [[6, 9, 15, 19]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1
    assert unicode_script_strs == tokens

    # Test UnicodeScriptTokenizer, English and numbers, with_offsets = True
    unicode_script_strs = [["Welcome", "to", "Beijing", "123"]]
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False, with_offsets=True)
    dataset = dataset.map(input_columns=["text"], output_columns=["token", "offsets_start", "offsets_limit"],
                          operations=tokenizer)
    dataset = dataset.project(columns=["token", "offsets_start", "offsets_limit"])

    tokens = []
    count = 0
    expected_offsets_start = [[0, 8, 11, 18]]
    expected_offsets_limit = [[7, 10, 18, 21]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens


def test_unicodescripttokenizer_operation_02():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op in eager mode with offsets and whitespace handling
    Expectation: Successfully tokenize in eager mode with expected outputs
    """
    # Test UnicodeScriptTokenizer, Chinese and numbers, with_offsets = True
    data = ["北京", "123", "欢迎", "你"]
    res = []
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False, with_offsets=True)
    for i in data:
        res.append(tokenizer(i))
    assert res[0][2] == 6
    assert res[1][2] == 3
    assert res[2][2] == 6
    assert res[3][2] == 3

    # Test UnicodeScriptTokenizer, Chinese and numbers, keep_whitespace=True, with_offsets = False
    data = ["欢 迎"]
    res = []
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=True, with_offsets=False)
    for i in data:
        res.append(tokenizer(i))
    assert (res[0] == ['欢', ' ', '迎']).all()

    # Test UnicodeScriptTokenizer, Chinese and numbers, with_offsets = True
    data = ["北京", "123", "欢迎", "你"]
    res = []
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False, with_offsets=False)
    for i in data:
        res.append(tokenizer(i))
    assert res[0] == ['北京']
    assert res[1] == ['123']
    assert res[2] == ['欢迎']
    assert res[3] == ['你']

    # Test UnicodeScriptTokenizer, Chinese and numbers, keep_whitespace=True, with_offsets = False
    data = "!@#$%^&*"
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=True, with_offsets=False)
    res = tokenizer(data)
    assert res == ['!@#$%^&*']


def test_unicode_script_tokenizer_default():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op with para keep_whitespace=False
    Expectation: Output is equal to the expected output
    """
    unicode_script_strs = [["Welcome", "to", "Beijing", "!"],
                           ["北京欢迎您", "！"],
                           ["我喜欢", "English", "!"],
                           [""]]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False)
    dataset = dataset.map(operations=tokenizer)

    tokens = []
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text'].tolist()
        tokens.append(token)
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens


def test_unicode_script_tokenizer_default2():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op with para keep_whitespace=True
    Expectation: Output is equal to the expected output
    """
    unicode_script_strs2 = [["Welcome", " ", "to", " ", "Beijing", "!"],
                            ["北京欢迎您", "！"],
                            ["我喜欢", "English", "!"],
                            ["  "]]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=True)
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text'].tolist()
        tokens.append(token)
    logger.info("The out tokens is :", tokens)
    assert unicode_script_strs2 == tokens


def test_unicode_script_tokenizer_with_offsets():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op with para keep_whitespace=False and with_offsets=True
    Expectation: Output is equal to the expected output
    """
    unicode_script_strs = [["Welcome", "to", "Beijing", "!"],
                           ["北京欢迎您", "！"],
                           ["我喜欢", "English", "!"],
                           [""]]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=False, with_offsets=True)
    dataset = dataset.map(operations=tokenizer, input_columns=['text'],
                          output_columns=['token', 'offsets_start', 'offsets_limit'])
    tokens = []
    expected_offsets_start = [[0, 8, 11, 18], [0, 15], [0, 9, 16], [0]]
    expected_offsets_limit = [[7, 10, 18, 19], [15, 18], [9, 16, 17], [0]]
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count += 1
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens


def test_unicode_script_tokenizer_with_offsets2():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op with para keep_whitespace=True and with_offsets=True
    Expectation: Output is equal to the expected output
    """
    unicode_script_strs2 = [["Welcome", " ", "to", " ", "Beijing", "!"],
                            ["北京欢迎您", "！"],
                            ["我喜欢", "English", "!"],
                            ["  "]]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=True, with_offsets=True)
    dataset = dataset.map(operations=tokenizer, input_columns=['text'],
                          output_columns=['token', 'offsets_start', 'offsets_limit'])
    tokens = []
    expected_offsets_start = [[0, 7, 8, 10, 11, 18], [0, 15], [0, 9, 16], [0]]
    expected_offsets_limit = [[7, 8, 10, 11, 18, 19], [15, 18], [9, 16, 17], [2]]
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count += 1
    logger.info("The out tokens is :", tokens)
    assert unicode_script_strs2 == tokens


def test_unicodescripttokenizer_exception_01():
    """
    Feature: UnicodeScriptTokenizer op
    Description: Test UnicodeScriptTokenizer op with invalid parameter types
    Expectation: Raise expected exceptions for invalid keep_whitespace parameter
    """
    # Test keep_whitespace=None
    unicode_script_strs = [["北京", "123", "欢迎", "你"]]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    with pytest.raises(TypeError, match="Wrong input type for keep_whitespace, should be boolean."):
        tokenizer = text.UnicodeScriptTokenizer(keep_whitespace=None)
        dataset = dataset.map(operations=tokenizer)
        tokens = []
        for i in dataset.create_dict_iterator(output_numpy=True):
            data = i['text'].tolist()
            tokens.append(data)
        logger.info("The out tokens is : {}".format(tokens))
        assert unicode_script_strs == tokens

    # Test keep_whitespace="cdack ^$*&%R"
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    with pytest.raises(TypeError, match="Wrong input type for keep_whitespace, should be boolean"):
        tokenizer = text.UnicodeScriptTokenizer(keep_whitespace="cdack ^$*&%R")
        dataset.map(operations=tokenizer)
