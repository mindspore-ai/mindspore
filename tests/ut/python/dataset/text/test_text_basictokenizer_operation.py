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
"""text transform - basictokenizer"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import NormalizeForm
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"
BASIC_TOKENIZER_FILE = "../data/dataset/testTokenizerData/basic_tokenizer.txt"


def test_basictokenizer_operation_01():
    """
    Feature: BasicTokenizer op
    Description: Test BasicTokenizer op with different parameters (lower_case, keep_whitespace, normalization_form)
    Expectation: Successfully tokenize text with specified settings
    """
    # Test BasicTokenizer, default parameter
    data_file1 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/english.txt")
    expected_tokens = [['Welcome', 'to', 'Beijing', '!']]
    dataset = ds.TextFileDataset(data_file1, shuffle=False)
    op = text.BasicTokenizer()
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, lower_case is True
    data_file4 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/accents.txt")
    expected_tokens = [['orcpzsiayd']]
    dataset = ds.TextFileDataset(data_file4, shuffle=False)
    op = text.BasicTokenizer(lower_case=True)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, Chinese and English text, lower_case is True
    data_file2 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/cnanden.txt")
    expected_tokens = [["我", "喜", "欢", "english", "!"]]
    dataset = ds.TextFileDataset(data_file2, shuffle=False)
    op = text.BasicTokenizer(lower_case=True)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, keep_whitespace is True
    data_file1 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/english.txt")
    expected_tokens = [['welcome', ' ', 'to', ' ', 'beijing', '!']]
    dataset = ds.TextFileDataset(data_file1, shuffle=False)
    op = text.BasicTokenizer(lower_case=True, keep_whitespace=True)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, normalization_form is NormalizeForm.NONE
    data_file4 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/accents.txt")
    expected_tokens = ["Orčpžsíáýd"]
    dataset = ds.TextFileDataset(data_file4, shuffle=False)
    op = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NONE,
                             preserve_unused_token=False)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['text']
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, normalization_form is NormalizeForm.NFC
    data_file4 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/accents.txt")
    expected_tokens = ["Orčpžsíáýd"]
    dataset = ds.TextFileDataset(data_file4, shuffle=False)
    op = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFC,
                             preserve_unused_token=False)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['text']
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, normalization_form is NormalizeForm.NFKC
    data_file4 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/accents.txt")
    expected_tokens = ["Orčpžsíáýd"]
    dataset = ds.TextFileDataset(data_file4, shuffle=False)
    op = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFKC,
                             preserve_unused_token=False)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['text']
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1


def test_basictokenizer_operation_02():
    """
    Feature: BasicTokenizer op
    Description: Test BasicTokenizer op with different normalization forms and preserve_unused_token
    Expectation: Successfully apply normalization and preserve special tokens
    """
    # Test BasicTokenizer, normalization_form is NormalizeForm.NFD
    data_file4 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/accents.txt")
    expected_tokens = ["Orčpžsíáýd"]
    dataset = ds.TextFileDataset(data_file4, shuffle=False)
    op = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFD,
                             preserve_unused_token=False)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['text']
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, normalization_form is NormalizeForm.NFKD
    data_file4 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/accents.txt")
    expected_tokens = ["Orčpžsíáýd"]
    dataset = ds.TextFileDataset(data_file4, shuffle=False)
    op = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFKD,
                             preserve_unused_token=False)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['text']
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, preserve_unused_token is False
    data_file3 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/token.txt")
    expected_tokens = [['[', 'asd', ']', '[', 'cls', ']']]
    dataset = ds.TextFileDataset(data_file3, shuffle=False)
    op = text.BasicTokenizer(lower_case=True, keep_whitespace=True, normalization_form=NormalizeForm.NFKD,
                             preserve_unused_token=False)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, preserve_unused_token is True
    data_file3 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/token.txt")
    expected_tokens = [['[', 'asd', ']', '[CLS]']]
    dataset = ds.TextFileDataset(data_file3, shuffle=False)
    op = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NONE,
                             preserve_unused_token=True)
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, with_offsets is True
    data_file1 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/english.txt")
    expected_tokens = [['Welcome', 'to', 'Beijing', '!']]
    dataset = ds.TextFileDataset(data_file1, shuffle=False)
    op = text.BasicTokenizer(with_offsets=True)
    dataset = dataset.map(input_columns=['text'], output_columns=['token', 'offsets_start', 'offsets_limit'],
                          operations=op)
    dataset = dataset.project(columns=['token', 'offsets_start', 'offsets_limit'])
    count = 0
    expected_offsets_start = [[0, 8, 11, 18]]
    expected_offsets_limit = [[7, 10, 18, 19]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["token"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1

    # Test BasicTokenizer, datasetfile is space
    data_file5 = os.path.join(TEST_DATA_DATASET_FUNC,
                              "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/space.txt")
    expected_tokens = [['']]
    dataset = ds.TextFileDataset(data_file5, shuffle=False)
    op = text.BasicTokenizer()
    dataset = dataset.map(operations=op)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["text"]
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1

    # Test BasicTokenizer, eager mode, default parameter
    data = 'Welcome to beijing!'
    expected_tokens = ['Welcome', 'to', 'beijing', '!']
    tokens = []
    res = text.BasicTokenizer()(data)
    for i in res:
        tokens.append(i)
    np.testing.assert_array_equal(tokens, expected_tokens)


def test_basictokenizer_operation_03():
    """
    Feature: BasicTokenizer op
    Description: Test BasicTokenizer op with different parameters (lower_case, keep_whitespace, normalization_form)
    Expectation: Successfully tokenize text with specified settings
    """
    # Test BasicTokenizer, eager mode, lower_case is True
    data = '你好！ Beijing！'
    expected_tokens = ['你', '好', '!', ' ', 'beijing', '!']
    tokens = []
    res = text.BasicTokenizer(lower_case=True, keep_whitespace=True)(data)
    for i in res:
        tokens.append(i)
    np.testing.assert_array_equal(tokens, expected_tokens)

    # Test BasicTokenizer, eager mode, keep_whitespace is True
    data = '你好！ Beijing！'
    expected_tokens = ['你', '好', '!', ' ', 'beijing', '!']
    tokens = []
    res = text.BasicTokenizer(lower_case=True, keep_whitespace=True)(data)
    for i in res:
        tokens.append(i)
    np.testing.assert_array_equal(tokens, expected_tokens)

    # Test BasicTokenizer, eager mode, normalization_form is NormalizeForm.NONE
    data = 'čp'
    expected_tokens = ['čp']
    expected_tokens_asc = [r"'\u010dp'"]
    tokens = []
    tokens_asc = []
    res = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NONE)(data)
    for i in res:
        tokens.append(i)
        tokens_asc.append(ascii(i))
    np.testing.assert_array_equal(tokens, expected_tokens)
    assert tokens_asc == expected_tokens_asc

    # Test BasicTokenizer, eager mode, normalization_form is NormalizeForm.NFC
    data = 'čp'
    expected_tokens = ['čp']
    expected_tokens_asc = [r"'\u010dp'"]
    tokens = []
    tokens_asc = []
    res = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFC)(data)
    for i in res:
        tokens.append(i)
        tokens_asc.append(ascii(i))
    np.testing.assert_array_equal(tokens, expected_tokens)
    assert tokens_asc == expected_tokens_asc

    # Test BasicTokenizer, eager mode, normalization_form is NormalizeForm.NFKC
    data = 'čp'
    expected_tokens = ['čp']
    expected_tokens_asc = [r"'\u010dp'"]
    tokens = []
    tokens_asc = []
    res = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFKC)(data)
    for i in res:
        tokens.append(i)
        tokens_asc.append(ascii(i))
    np.testing.assert_array_equal(tokens, expected_tokens)
    assert tokens_asc == expected_tokens_asc

    # Test BasicTokenizer, eager mode, normalization_form is NormalizeForm.NFD
    data = 'čp'
    expected_tokens_asc = [r"'c\u030cp'"]
    tokens_asc = []
    res = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFD)(data)
    for i in res:
        tokens_asc.append(ascii(i))
    assert tokens_asc == expected_tokens_asc

    # Test BasicTokenizer, eager mode, normalization_form is NormalizeForm.NFKD
    data = 'čp'
    expected_tokens_asc = [r"'c\u030cp'"]
    tokens_asc = []
    res = text.BasicTokenizer(lower_case=False, keep_whitespace=True, normalization_form=NormalizeForm.NFKD)(data)
    for i in res:
        tokens_asc.append(ascii(i))
    assert tokens_asc == expected_tokens_asc

    # Test BasicTokenizer, eager mode, preserve_unused_token is True
    data = '[123],[CLS]'
    expected_tokens = ['[', '123', ']', ',', '[CLS]']
    tokens = []
    res = text.BasicTokenizer(lower_case=True, keep_whitespace=True, preserve_unused_token=True)(data)
    for i in res:
        tokens.append(i)

    np.testing.assert_array_equal(tokens, expected_tokens)

    # Test BasicTokenizer, eager mode, preserve_unused_token is False
    data = '[123],[CLS]'
    expected_tokens = ['[', '123', ']', ',', '[', 'cls', ']']
    tokens = []
    res = text.BasicTokenizer(lower_case=True, keep_whitespace=True, preserve_unused_token=False)(data)
    for i in res:
        tokens.append(i)
    np.testing.assert_array_equal(tokens, expected_tokens)


test_paras = [
    {
        "first": 1,
        "last": 6,
        "expected_tokens":
        [['Welcome', 'to', 'Beijing', '北', '京', '欢', '迎', '您'],
         ['長', '風', '破', '浪', '會', '有', '時', '，', '直', '掛', '雲', '帆', '濟', '滄', '海'],
         ['😀', '嘿', '嘿', '😃', '哈', '哈', '😄', '大', '笑', '😁', '嘻', '嘻'],
         ['明', '朝', '（', '1368', '—', '1644', '年', '）', '和', '清', '朝',
          '（', '1644', '—', '1911', '年', '）', '，', '是', '中', '国', '封',
          '建', '王', '朝', '史', '上', '最', '后', '两', '个', '朝', '代'],
         ['明', '代', '（', '1368', '-', '1644', '）', 'と', '清', '代',
          '（', '1644', '-', '1911', '）', 'は', '、', '中', '国', 'の', '封',
          '建', '王', '朝', 'の', '歴', '史', 'における', '最', '後', 'の2つの', '王', '朝', 'でした'],
         ['명나라', '(', '1368', '-', '1644', ')', '와', '청나라', '(', '1644', '-', '1911', ')', '는',
          '중국', '봉건', '왕조의', '역사에서', '마지막', '두', '왕조였다']],
        "expected_offsets_start": [[0, 8, 11, 18, 21, 24, 27, 30],
                                   [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
                                   [0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37],
                                   [0, 3, 6, 9, 13, 16, 20, 23, 26, 29, 32, 35, 38, 42, 45, 49,
                                    52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100],
                                   [0, 3, 6, 9, 13, 14, 18, 21, 24, 27, 30, 33, 37, 38, 42, 45, 48, 51,
                                    54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 93, 96, 99, 109, 112, 115],
                                   [0, 10, 11, 15, 16, 20, 21, 25, 35, 36, 40, 41, 45, 46, 50, 57, 64, 74, 87,
                                    97, 101]],
        "expected_offsets_limit": [[7, 10, 18, 21, 24, 27, 30, 33],
                                   [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
                                   [4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40],
                                   [3, 6, 9, 13, 16, 20, 23, 26, 29, 32, 35, 38, 42, 45, 49, 52, 55, 58,
                                    61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103],
                                   [3, 6, 9, 13, 14, 18, 21, 24, 27, 30, 33, 37, 38, 42, 45, 48, 51, 54,
                                    57, 60, 63, 66, 69, 72, 75, 78, 81, 93, 96, 99, 109, 112, 115, 124],
                                   [9, 11, 15, 16, 20, 21, 24, 34, 36, 40, 41, 45, 46, 49, 56, 63, 73, 86, 96,
                                    100, 113]]
    },
    {
        "first": 7,
        "last": 7,
        "expected_tokens": [['this', 'is', 'a', 'funky', 'string']],
        "expected_offsets_start": [[0, 5, 8, 10, 16]],
        "expected_offsets_limit": [[4, 7, 9, 15, 22]],
        "lower_case": True
    },
]


def check_basic_tokenizer_default(first, last, expected_tokens, expected_offsets_start, expected_offsets_limit,
                                  lower_case=False, keep_whitespace=False,
                                  normalization_form=text.utils.NormalizeForm.NONE, preserve_unused_token=False):
    dataset = ds.TextFileDataset(BASIC_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)

    basic_tokenizer = text.BasicTokenizer(lower_case=lower_case,
                                          keep_whitespace=keep_whitespace,
                                          normalization_form=normalization_form,
                                          preserve_unused_token=preserve_unused_token)

    dataset = dataset.map(operations=basic_tokenizer)
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text']
        logger.info("Out:", token)
        logger.info("Exp:", expected_tokens[count])
        np.testing.assert_array_equal(token, expected_tokens[count])
        count = count + 1


def check_basic_tokenizer_with_offsets(first, last, expected_tokens, expected_offsets_start, expected_offsets_limit,
                                       lower_case=False, keep_whitespace=False,
                                       normalization_form=text.utils.NormalizeForm.NONE, preserve_unused_token=False):
    dataset = ds.TextFileDataset(BASIC_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)

    basic_tokenizer = text.BasicTokenizer(lower_case=lower_case,
                                          keep_whitespace=keep_whitespace,
                                          normalization_form=normalization_form,
                                          preserve_unused_token=preserve_unused_token,
                                          with_offsets=True)

    dataset = dataset.map(operations=basic_tokenizer, input_columns=['text'],
                          output_columns=['token', 'offsets_start', 'offsets_limit'])
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['token']
        logger.info("Out:", token)
        logger.info("Exp:", expected_tokens[count])
        np.testing.assert_array_equal(token, expected_tokens[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1


def test_basictokenizer_operation_04():
    """
    Feature: BasicTokenizer op
    Description: Test BasicTokenizer op with with_offsets parameter in eager mode
    Expectation: Successfully provide offset information for tokens
    """
    # Test BasicTokenizer, eager mode, with_offsets is True
    data = '你好！ Beijing！'
    expected_tokens = np.array(['你', '好', '!', ' ', 'beijing', '!'])
    expected_offsets_start = np.array([0, 3, 6, 7, 8, 15])
    expected_offsets_limit = np.array([3, 6, 7, 8, 15, 16])
    res = text.BasicTokenizer(lower_case=True, keep_whitespace=True, with_offsets=True)(data)
    np.testing.assert_array_equal(res[0], expected_tokens)
    np.testing.assert_array_equal(res[1], expected_offsets_start)
    np.testing.assert_array_equal(res[2], expected_offsets_limit)

    # setting with_offsets to True
    for paras in test_paras:
        check_basic_tokenizer_with_offsets(**paras)

    # with default parameters
    for paras in test_paras:
        check_basic_tokenizer_default(**paras)


def test_basictokenizer_exception_01():
    """
    Feature: BasicTokenizer op
    Description: Test BasicTokenizer op with invalid parameter types
    Expectation: Raise expected exceptions for invalid parameter types
    """
    # Test BasicTokenizer, lower_case is int
    with pytest.raises(TypeError, match='Wrong input type for lower_case, should be boolean.'):
        text.BasicTokenizer(lower_case=0)

    # Test BasicTokenizer, keep_whitespace is int
    with pytest.raises(TypeError, match='Wrong input type for keep_whitespace, should be boolean.'):
        text.BasicTokenizer(keep_whitespace=0)

    # Test BasicTokenizer, normalization_form is int
    with pytest.raises(TypeError, match="Wrong input type for normalization_form, should be enum of 'NormalizeForm'."):
        text.BasicTokenizer(normalization_form=0)

    # Test BasicTokenizer, preserve_unused_token is int
    with pytest.raises(TypeError, match="Wrong input type for preserve_unused_token, should be boolean."):
        text.BasicTokenizer(preserve_unused_token=0)

    # Test BasicTokenizer, with_offsets is int
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean."):
        text.BasicTokenizer(with_offsets=0)

    # Test BasicTokenizer, wrong datatype
    data = np.random.randn(10, 20, 25)
    dataset = ds.NumpySlicesDataset(data, ["col"], shuffle=False)
    op = text.BasicTokenizer()
    with pytest.raises(RuntimeError, match=r"map operation: \[BasicTokenizer\] failed. BasicToke"
                                           r"nizer: the input should be a scalar, but got a tensor with rank: 2"):
        dataset = dataset.map(input_columns=["col"], operations=op)
        for _ in dataset.create_dict_iterator():
            pass

    # Test BasicTokenizer, eager mode
    data = ['Welcome to Beijing!']
    with pytest.raises(RuntimeError, match=r"BasicTokenizer: the input should be a scalar, "
                                           r"but got a tensor with rank: 1"):
        text.BasicTokenizer()(data)
