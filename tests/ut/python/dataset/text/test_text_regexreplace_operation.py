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
"""text transform - regexreplace"""

import os
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/1.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/2.txt")
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/3.txt")
DATA_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/4.txt")
DATA_FILE = "../data/dataset/testTokenizerData/1.txt"
NORMALIZE_FILE = "../data/dataset/testTokenizerData/normalize.txt"
REGEX_REPLACE_FILE = "../data/dataset/testTokenizerData/regex_replace.txt"
REGEX_TOKENIZER_FILE = "../data/dataset/testTokenizerData/regex_tokenizer.txt"


def test_regexreplace_operation_01():
    """
    Feature: RegexReplace op
    Description: Test RegexReplace op with different patterns and replace_all settings
    Expectation: Successfully replace matched patterns in strings
    """
    # Test RegexReplace,"^(\\d:|b:)"
    pattern = "^(\\d:|b:)"
    replace = ""
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    replace_op = text.RegexReplace(pattern=pattern, replace=replace)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = ['hello', 'world', '31:beijing']
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        out_text.append(data)
    assert expect_str == out_text

    # Test RegexReplace,"\\s+"
    pattern = "\\s+"
    replace = ""
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    replace_op = text.RegexReplace(pattern=pattern, replace=replace)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = ["WelcometoChina!"]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        out_text.append(data)
    assert expect_str == out_text

    # Test RegexReplace,replace_all=True
    pattern = "one"
    replace = "two"
    data = 'onetwoonetwoone'
    replace_op = text.RegexReplace(pattern=pattern, replace=replace, replace_all=True)
    result = replace_op(data)
    assert result == 'twotwotwotwotwo'

    # Test RegexReplace,replace_all=False
    pattern = "one"
    replace = "two"
    data = 'onetwoonetwoone'
    replace_op = text.RegexReplace(pattern=pattern, replace=replace, replace_all=False)
    result = replace_op(data)
    assert result == 'twotwoonetwoone'


def test_regex_replace():
    """
    Feature: RegexReplace op
    Description: Test RegexReplace op basic usage
    Expectation: Output is equal to the expected output
    """

    def regex_replace(first, last, expect_str, pattern, replace):
        dataset = ds.TextFileDataset(REGEX_REPLACE_FILE, shuffle=False)
        if first > 1:
            dataset = dataset.skip(first - 1)
        if last >= first:
            dataset = dataset.take(last - first + 1)
        replace_op = text.RegexReplace(pattern, replace)
        dataset = dataset.map(operations=replace_op)
        out_text = []
        for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            token = i['text'].tolist()
            out_text.append(token)
        logger.info("Out:", out_text)
        logger.info("Exp:", expect_str)
        assert expect_str == out_text

    regex_replace(1, 2, ['H____ W____', "L__'_ G_"], "\\p{Ll}", '_')
    regex_replace(3, 5, ['hello', 'world', '31:beijing'], "^(\\d:|b:)", "")
    regex_replace(6, 6, ["WelcometoChina!"], "\\s+", "")
    regex_replace(7, 8, ['我不想长大', 'WelcometoShenzhen!'], "\\p{Cc}|\\p{Cf}|\\s+", "")


def test_regexreplace_exception_01():
    """
    Feature: RegexReplace op
    Description: Test RegexReplace op with invalid parameter types
    Expectation: Raise expected exceptions for invalid parameter types
    """
    # Test RegexReplace,replace_all=0
    pattern = "one"
    replace = "two"
    with pytest.raises(TypeError, match=r'Argument replace_all with value 0 is not of type \[\<class \'bool\'\>\].'):
        _ = text.RegexReplace(pattern=pattern, replace=replace, replace_all=0)

    # Test RegexReplace,pattern=True
    replace = "two"
    with pytest.raises(TypeError, match=r'Argument pattern with value True is not of type \[\<class \'str\'\>\].'):
        _ = text.RegexReplace(pattern=True, replace=replace, replace_all=True)

    # Test RegexReplace,replace=False
    pattern = "one"
    with pytest.raises(TypeError, match=r'Argument replace with value False is not of type \[\<class \'str\'\>\].'):
        _ = text.RegexReplace(pattern=pattern, replace=False, replace_all=True)
