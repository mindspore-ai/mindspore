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
"""text transform - unicodechartokenizer"""

import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset import text


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_FILE = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/file1.txt"
DATA_FILE_CN = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/chinese.txt"
DATA_FILE_EN = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/english.txt"
DATA_FILE_CNEN = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/cnanden.txt"
DATA_FILE_CNNUM = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/cnandnum.txt"
DATA_FILE_ENNUM = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/enandnum.txt"
DATA_FILE_ONECN = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/one_chinese.txt"
DATA_FILE_ONEEN = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/one_english.txt"
DATA_FILE_ONENUM = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/one_num.txt"
DATA_FILE_MUL = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/mul.txt"
DATA_FILE_GARBLED = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/garbled.txt"
DATA_FILE_NON = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/non.txt"
DATA_FILE_LARGEFILE = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testTokenizerData/largefile.txt"
DATA_FILE = "../data/dataset/testTokenizerData/1.txt"
NORMALIZE_FILE = "../data/dataset/testTokenizerData/normalize.txt"
REGEX_REPLACE_FILE = "../data/dataset/testTokenizerData/regex_replace.txt"
REGEX_TOKENIZER_FILE = "../data/dataset/testTokenizerData/regex_tokenizer.txt"


def split_by_unicode_char(input_strs):
    """
    Split utf-8 strings to unicode characters
    """
    out = []
    for s in input_strs:
        out.append(list(s))
    return out


def test_unicodechartokenizer_operation_01():
    """
    Feature: UnicodeCharTokenizer op
    Description: Test UnicodeCharTokenizer op with different character types (English, Chinese, numbers, symbols)
    Expectation: Successfully split strings into unicode characters
    """
    # Test UnicodeCharTokenizer,English string
    input_strs = ("hello Welcome to Beijing",)
    dataset = ds.TextFileDataset(DATA_FILE_EN, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer contains Chinese and number
    input_strs = ("黄河之水天上来123456奔流到海555不复回",)
    dataset = ds.TextFileDataset(DATA_FILE_CNNUM, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer contains Chinese and number
    input_strs = ("123 the water of the Yellow River comes to heaven",)
    dataset = ds.TextFileDataset(DATA_FILE_ENNUM, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer only one Chinese
    input_strs = ("我",)
    dataset = ds.TextFileDataset(DATA_FILE_ONECN, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer only one english
    input_strs = ("k",)
    dataset = ds.TextFileDataset(DATA_FILE_ONEEN, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer only one num
    input_strs = ("9",)
    dataset = ds.TextFileDataset(DATA_FILE_ONENUM, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer with multiple symbols
    input_strs = ("050400000008    test1050400000205       test2",
                  "<   &lt;>       &gt;&       &amp;¥       &yen;",
                  "©       &copy;®       &reg;°       &deg;±       &plusmn;",
                  "×       &times;÷       &divide;²       &sup2;",
                  "³       &sup3;abcABC[a-c6] ['a', 'c']{'a6':1}######",
                  "~!@#$%^&*()_+ , ，< 《 》> ? / \\|", "中文")
    dataset = ds.TextFileDataset(DATA_FILE_MUL, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer with garbled
    input_strs = ("浠婂澶皵澶浜嗘垜浠竴璧峰幓澶栭潰鐜惂",)
    dataset = ds.TextFileDataset(DATA_FILE_GARBLED, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens


def test_unicodechartokenizer_operation_02():
    """
    Feature: UnicodeCharTokenizer op
    Description: Test UnicodeCharTokenizer op with edge cases (empty, large file) and with_offsets parameter
    Expectation: Successfully handle edge cases and provide offset information
    """
    # Test UnicodeCharTokenizer with no string
    input_strs = ("    ",)
    dataset = ds.TextFileDataset(DATA_FILE_NON, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.append(data)
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer with largefile
    dataset = ds.TextFileDataset(DATA_FILE_LARGEFILE, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i['text'].tolist()
        tokens.extend(data)
    assert len(tokens) == 1343

    # Test UnicodeCharTokenizer,Chinese string, with_offsets=True
    input_strs = ("小明硕士毕业于中国科学院计算所",)
    dataset = ds.TextFileDataset(DATA_FILE_CN, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer(with_offsets=True)
    dataset = dataset.map(input_columns=['text'], output_columns=['token', 'offsets_start', 'offsets_limit'],
                          operations=tokenizer)
    dataset = dataset.project(columns=['token', 'offsets_start', 'offsets_limit'])
    tokens = []
    count = 0
    expected_offsets_start = [[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]]
    expected_offsets_limit = [[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count += 1
    assert split_by_unicode_char(input_strs) == tokens

    # Test UnicodeCharTokenizer,English string, with_offsets=True
    input_strs = ("hello Welcome to Beijing",)
    dataset = ds.TextFileDataset(DATA_FILE_EN, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer(with_offsets=True)
    dataset = dataset.map(input_columns=['text'], output_columns=['token', 'offsets_start', 'offsets_limit'],
                          operations=tokenizer)
    dataset = dataset.project(columns=['token', 'offsets_start', 'offsets_limit'])
    count = 0
    expected_offsets_start = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    expected_offsets_limit = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    tokens = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count += 1
    assert split_by_unicode_char(input_strs) == tokens

    # data = mindspore
    data = "mindspore"
    tokenizer = text.UnicodeCharTokenizer(with_offsets=True)
    res = tokenizer(data)
    assert (res[1] == [0, 1, 2, 3, 4, 5, 6, 7, 8]).all()
    assert (res[2] == [1, 2, 3, 4, 5, 6, 7, 8, 9]).all()

    # data = 12345!?#$123
    data = "12345!?#$123"
    tokenizer = text.UnicodeCharTokenizer(with_offsets=True)
    res = tokenizer(data)
    assert (res[1] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).all()
    assert (res[2] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).all()

    # data = 12345!?#$123 with_offsets = False
    data = "12345!?#$123"
    tokenizer = text.UnicodeCharTokenizer()
    res = tokenizer(data)
    assert (res == ['1', '2', '3', '4', '5', '!', '?', '#', '$', '1', '2', '3']).all()


def test_unicode_char_tokenizer_default():
    """
    Feature: UnicodeCharTokenizer op
    Description: Test UnicodeCharTokenizer op with default parameters
    Expectation: Output is equal to the expected output
    """
    input_strs = ("Welcome to Beijing!", "北京欢迎您！", "我喜欢English!", "  ")
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text'].tolist()
        tokens.append(token)
    logger.info("The out tokens is : {}".format(tokens))
    assert split_by_unicode_char(input_strs) == tokens


def test_unicode_char_tokenizer_with_offsets():
    """
    Feature: UnicodeCharTokenizer op
    Description: Test UnicodeCharTokenizer op with with_offsets=True
    Expectation: Output is equal to the expected output
    """
    input_strs = ("Welcome to Beijing!", "北京欢迎您！", "我喜欢English!", "  ")
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.UnicodeCharTokenizer(with_offsets=True)
    dataset = dataset.map(operations=tokenizer, input_columns=['text'],
                          output_columns=['token', 'offsets_start', 'offsets_limit'])
    tokens = []
    expected_offsets_start = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                              [0, 3, 6, 9, 12, 15], [0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16], [0, 1]]
    expected_offsets_limit = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                              [3, 6, 9, 12, 15, 18], [3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17], [1, 2]]
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['token'].tolist()
        tokens.append(token)
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count += 1
    logger.info("The out tokens is : {}".format(tokens))
    assert split_by_unicode_char(input_strs) == tokens


def test_unicodechartokenizer_exception_01():
    """
    Feature: UnicodeCharTokenizer op
    Description: Test UnicodeCharTokenizer op with invalid input types
    Expectation: Raise expected exceptions for invalid inputs
    """
    # data = 12345
    data = 12345
    tokenizer = text.UnicodeCharTokenizer(with_offsets=True)
    with pytest.raises(RuntimeError, match="UnicodeCharTokenizerOp: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = tokenizer(data)

    # with_offsets = 'True'
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.UnicodeCharTokenizer(with_offsets='True')

    # with_offsets = 0
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.UnicodeCharTokenizer(with_offsets=0)
