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
"""text transform - berttokenizer"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC,
                         "text_data/testTextFile/textfile/testTokenizerData/BertTokenizer/1.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/BertTokenizer/2.txt")
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/BertTokenizer/3.txt")
DATA_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/BertTokenizer/4.txt")
DATA_FILE4 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/BertTokenizer/5.txt")
DATA_FILE5 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/BertTokenizer/6.txt")

BERT_TOKENIZER_FILE = "../data/dataset/testTokenizerData/bert_tokenizer.txt"

vocab_bert = [
    "床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低", "思", "故", "乡",
    "繁", "體", "字", "嘿", "哈", "大", "笑", "嘻",
    "i", "am", "mak", "make", "small", "mistake", "##s", "during", "work", "##ing", "hour",
    "😀", "😃", "😄", "😁", "+", "/", "-", "=", "12", "28", "40", "16", " ", "I",
    "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"
]

pad = '<pad>'


def test_berttokenizer_operation_01():
    """
    Feature: BertTokenizer op
    Description: Test BertTokenizer op with different parameters (lower_case, normalization_form, preserve_unused_token)
    Expectation: Successfully tokenize text with BERT-style tokenization
    """
    # test lower_case=False
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=False)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['I', "am", 'mak', '##ing', 'small', 'mistake', '##s', 'during', 'work', '##ing', 'hour', '##s']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test normalization_form=text.utils.NormalizeForm.NFKC
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, normalization_form=text.utils.NormalizeForm.NFKC)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['😀', '嘿', '嘿', '😃', '哈', '哈', '😄', '大', '笑', '😁', '嘻', '嘻'], ['繁', '體', '字']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test preserve_unused_token=True, special tokens
    dataset = ds.TextFileDataset(DATA_FILE3, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=False, preserve_unused_token=True)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['[UNK]', '[CLS]'],
                  ['[UNK]', '[SEP]'],
                  ['[UNK]', '[UNK]'],
                  ['[UNK]', '[PAD]'],
                  ['[UNK]', '[MASK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test preserve_unused_token=True
    dataset = ds.TextFileDataset(DATA_FILE4, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, preserve_unused_token=True)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['12', '+', '/', '-', '28', '=', '40', '/', '-', '16']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test keep_whitespace=True
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=False, preserve_unused_token=True, keep_whitespace=True)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['[UNK]', ' ', '[CLS]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test unknown_token=''
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=False, preserve_unused_token=True, keep_whitespace=True,
                                     unknown_token='')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['unused', ' ', '[CLS]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1


def test_berttokenizer_operation_02():
    """
    Feature: BertTokenizer op
    Description: Test BertTokenizer op with unknown_token and preserve_unused_token settings
    Expectation: Successfully handle unknown tokens and special tokens
    """
    # test unknown_token='[UNK]'
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=False, preserve_unused_token=True, keep_whitespace=True,
                                     unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['[UNK]', ' ', '[CLS]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test preserve_unused_token=False
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=False, preserve_unused_token=False, keep_whitespace=True,
                                     unknown_token='')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    out_text = []
    expect_str = [['unused', ' ', '[', 'CLS', ']']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"]
        out_text.append(data.tolist())
        np.testing.assert_array_equal(data, expect_str[count])
        count = count + 1

    # test with_offsets = True, Chinese text
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, with_offsets=True)
    dataset = dataset.map(input_columns=["text"], output_columns=["token", "offsets_start", "offsets_limit"],
                          operations=tokenizer_op)
    dataset = dataset.project(columns=["token", "offsets_start", "offsets_limit"])
    count = 0
    out_token = []
    expect_str = [['床', '前', '明', '月', '光'],
                  ['疑', '是', '地', '上', '霜'],
                  ['举', '头', '望', '明', '月'],
                  ['低', '头', '思', '故', '乡']]
    expected_offsets_start = [[0, 3, 6, 9, 12],
                              [0, 3, 6, 9, 12],
                              [0, 3, 6, 9, 12],
                              [0, 3, 6, 9, 12]]
    expected_offsets_limit = [[3, 6, 9, 12, 15],
                              [3, 6, 9, 12, 15],
                              [3, 6, 9, 12, 15],
                              [3, 6, 9, 12, 15]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["token"]
        out_token.append(token.tolist())
        np.testing.assert_array_equal(token, expect_str[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1

    # test with_offsets = True, English text
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=True, with_offsets=True)
    dataset = dataset.map(input_columns=["text"], output_columns=["token", "offsets_start", "offsets_limit"],
                          operations=tokenizer_op)
    dataset = dataset.project(columns=["token", "offsets_start", "offsets_limit"])
    count = 0
    out_token = []
    expect_str = [['i', 'am', 'mak', '##ing', 'small', 'mistake', '##s', 'during', 'work', '##ing', 'hour', '##s']]
    expected_offsets_start = [[0, 2, 5, 8, 12, 18, 25, 27, 34, 38, 42, 46]]
    expected_offsets_limit = [[1, 4, 8, 11, 17, 25, 26, 33, 38, 41, 46, 47]]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i["token"]
        out_token.append(token.tolist())
        np.testing.assert_array_equal(token, expect_str[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1

    # test default value
    data = "床前明月光疑是地上霜!"
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    out = tokenizer_op(data)

    tokenizer_op = text.BertTokenizer(vocab=vocab, suffix_indicator='##', max_bytes_per_token=100, unknown_token='[UNK]',
                                     lower_case=False, keep_whitespace=False, preserve_unused_token=True,
                                     with_offsets=False)
    out1 = tokenizer_op(data)
    assert (out == out1).all()


def test_berttokenizer_operation_03():
    """
    Feature: BertTokenizer op
    Description: Test BertTokenizer op in eager mode with different character types
    Expectation: Successfully tokenize in eager mode with expected results
    """
    data = "繁體字嘿哈大笑嘻"
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    res = []
    for i in data:
        op = tokenizer_op(i)
        res.append(op)
    assert res[0] == ['繁']
    assert res[1] == ['體']
    assert res[2] == ['字']

    data = "1234567890"
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    res = []
    for i in data:
        op = tokenizer_op(i)
        res.append(op)
    assert res[0] == ['[UNK]']
    assert res[1] == ['[UNK]']
    assert res[2] == ['[UNK]']

    data = "😀😃😄😁+"
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    res = []
    for i in data:
        op = tokenizer_op(i)
        res.append(op)
    assert res[0] == ['😀']
    assert res[1] == ['😃']
    assert res[2] == ['😄']
    assert res[3] == ['😁']

    data = "[CLS][SEP][UNK]"
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, lower_case=True)
    res = []
    for i in data:
        op = tokenizer_op(i)
        res.append(op)
    assert res[0] == ['[UNK]']
    assert res[1] == ['[UNK]']
    assert res[2] == ['[UNK]']

    data = "繁體 字嘿哈"
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab, keep_whitespace=True)
    res = []
    for i in data:
        op = tokenizer_op(i)
        res.append(op)
    assert res == ['繁', '體', ' ', '字', '嘿', '哈']


vocab_bert = [
    "床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低", "思", "故", "乡",
    "繁", "體", "字", "嘿", "哈", "大", "笑", "嘻",
    "i", "am", "mak", "make", "small", "mistake", "##s", "during", "work", "##ing", "hour",
    "😀", "😃", "😄", "😁", "+", "/", "-", "=", "12", "28", "40", "16", " ", "I",
    "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"
]

pad = '<pad>'

test_paras = [
    # test chinese text
    {
        "first": 1,
        "last": 4,
        "expect_str": [['床', '前', '明', '月', '光'],
                       ['疑', '是', '地', '上', '霜'],
                       ['举', '头', '望', '明', '月'],
                       ['低', '头', '思', '故', '乡']],
        "expected_offsets_start": [[0, 3, 6, 9, 12],
                                   [0, 3, 6, 9, 12],
                                   [0, 3, 6, 9, 12],
                                   [0, 3, 6, 9, 12]],
        "expected_offsets_limit": [[3, 6, 9, 12, 15],
                                   [3, 6, 9, 12, 15],
                                   [3, 6, 9, 12, 15],
                                   [3, 6, 9, 12, 15]],
        "vocab_list": vocab_bert
    },
    # test english text
    {
        "first": 5,
        "last": 5,
        "expect_str": [['i', 'am', 'mak', '##ing', 'small', 'mistake', '##s', 'during', 'work', '##ing', 'hour',
                        '##s']],
        "expected_offsets_start": [[0, 2, 5, 8, 12, 18, 25, 27, 34, 38, 42, 46]],
        "expected_offsets_limit": [[1, 4, 8, 11, 17, 25, 26, 33, 38, 41, 46, 47]],
        "lower_case": True,
        "vocab_list": vocab_bert
    },
    {
        "first": 5,
        "last": 5,
        "expect_str": [['I', "am", 'mak', '##ing', 'small', 'mistake', '##s', 'during', 'work', '##ing', 'hour',
                        '##s']],
        "expected_offsets_start": [[0, 2, 5, 8, 12, 18, 25, 27, 34, 38, 42, 46]],
        "expected_offsets_limit": [[1, 4, 8, 11, 17, 25, 26, 33, 38, 41, 46, 47]],
        "lower_case": False,
        "vocab_list": vocab_bert
    },
    # test emoji tokens
    {
        "first": 6,
        "last": 7,
        "expect_str": [
            ['😀', '嘿', '嘿', '😃', '哈', '哈', '😄', '大', '笑', '😁', '嘻', '嘻'],
            ['繁', '體', '字']],
        "expected_offsets_start": [[0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37], [0, 3, 6]],
        "expected_offsets_limit": [[4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40], [3, 6, 9]],
        "normalization_form": text.utils.NormalizeForm.NFKC,
        "vocab_list": vocab_bert
    },
    # test preserved tokens
    {
        "first": 8,
        "last": 14,
        "expect_str": [
            ['[UNK]', '[CLS]'],
            ['[UNK]', '[SEP]'],
            ['[UNK]', '[UNK]'],
            ['[UNK]', '[PAD]'],
            ['[UNK]', '[MASK]'],
            ['[unused1]'],
            ['[unused10]']
        ],
        "expected_offsets_start": [[0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0], [0]],
        "expected_offsets_limit": [[6, 12], [6, 12], [6, 12], [6, 12], [6, 13], [9], [10]],
        "lower_case": False,
        "vocab_list": vocab_bert,
        "preserve_unused_token": True,
    },
    {
        "first": 8,
        "last": 14,
        "expect_str": [
            ['[UNK]', '[CLS]'],
            ['[UNK]', '[SEP]'],
            ['[UNK]', '[UNK]'],
            ['[UNK]', '[PAD]'],
            ['[UNK]', '[MASK]'],
            ['[unused1]'],
            ['[unused10]']
        ],
        "expected_offsets_start": [[0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0], [0]],
        "expected_offsets_limit": [[6, 12], [6, 12], [6, 12], [6, 12], [6, 13], [9], [10]],
        "lower_case": True,
        "vocab_list": vocab_bert,
        "preserve_unused_token": True,
    },
    # test special symbol
    {
        "first": 15,
        "last": 15,
        "expect_str": [['12', '+', '/', '-', '28', '=', '40', '/', '-', '16']],
        "expected_offsets_start": [[0, 2, 3, 4, 5, 7, 8, 10, 11, 12]],
        "expected_offsets_limit": [[2, 3, 4, 5, 7, 8, 10, 11, 12, 14]],
        "preserve_unused_token": True,
        "vocab_list": vocab_bert
    },
    # test non-default params
    {
        "first": 8,
        "last": 8,
        "expect_str": [['[UNK]', ' ', '[CLS]']],
        "expected_offsets_start": [[0, 6, 7]],
        "expected_offsets_limit": [[6, 7, 12]],
        "lower_case": False,
        "vocab_list": vocab_bert,
        "preserve_unused_token": True,
        "keep_whitespace": True
    },
    {
        "first": 8,
        "last": 8,
        "expect_str": [['unused', ' ', '[CLS]']],
        "expected_offsets_start": [[0, 6, 7]],
        "expected_offsets_limit": [[6, 7, 12]],
        "lower_case": False,
        "vocab_list": vocab_bert,
        "preserve_unused_token": True,
        "keep_whitespace": True,
        "unknown_token": ''
    },
    {
        "first": 8,
        "last": 8,
        "expect_str": [['unused', ' ', '[', 'CLS', ']']],
        "expected_offsets_start": [[0, 6, 7, 8, 11]],
        "expected_offsets_limit": [[6, 7, 8, 11, 12]],
        "lower_case": False,
        "vocab_list": vocab_bert,
        "preserve_unused_token": False,
        "keep_whitespace": True,
        "unknown_token": ''
    },
]


def check_bert_tokenizer_default(first, last, expect_str,
                                 expected_offsets_start, expected_offsets_limit,
                                 vocab_list, suffix_indicator='##',
                                 max_bytes_per_token=100, unknown_token='[UNK]',
                                 lower_case=False, keep_whitespace=False,
                                 normalization_form=text.utils.NormalizeForm.NONE,
                                 preserve_unused_token=False):
    dataset = ds.TextFileDataset(BERT_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)
    vocab = text.Vocab.from_list(vocab_list)
    tokenizer_op = text.BertTokenizer(
        vocab=vocab, suffix_indicator=suffix_indicator,
        max_bytes_per_token=max_bytes_per_token, unknown_token=unknown_token,
        lower_case=lower_case, keep_whitespace=keep_whitespace,
        normalization_form=normalization_form,
        preserve_unused_token=preserve_unused_token)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text']
        logger.info("Out:", token)
        logger.info("Exp:", expect_str[count])
        np.testing.assert_array_equal(token, expect_str[count])
        count = count + 1


def check_bert_tokenizer_with_offsets(first, last, expect_str,
                                      expected_offsets_start, expected_offsets_limit,
                                      vocab_list, suffix_indicator='##',
                                      max_bytes_per_token=100, unknown_token='[UNK]',
                                      lower_case=False, keep_whitespace=False,
                                      normalization_form=text.utils.NormalizeForm.NONE,
                                      preserve_unused_token=False):
    dataset = ds.TextFileDataset(BERT_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)
    vocab = text.Vocab.from_list(vocab_list)
    tokenizer_op = text.BertTokenizer(
        vocab=vocab, suffix_indicator=suffix_indicator, max_bytes_per_token=max_bytes_per_token,
        unknown_token=unknown_token, lower_case=lower_case, keep_whitespace=keep_whitespace,
        normalization_form=normalization_form, preserve_unused_token=preserve_unused_token, with_offsets=True)
    dataset = dataset.map(operations=tokenizer_op, input_columns=['text'],
                          output_columns=['token', 'offsets_start', 'offsets_limit'])
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['token']
        logger.info("Out:", token)
        logger.info("Exp:", expect_str[count])
        np.testing.assert_array_equal(token, expect_str[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1


def test_berttokenizer_operation_04():
    """
    Feature: BertTokenizer
    Description: Test BertTokenizer by setting with_offsets to False / True
    Expectation: Output is equal to the expected output
    """

    # setting with_offsets to False
    for paras in test_paras:
        check_bert_tokenizer_default(**paras)

    # setting with_offsets to True
    for paras in test_paras:
        check_bert_tokenizer_with_offsets(**paras)


def test_berttokenizer_exception_01():
    """
    Feature: BertTokenizer op
    Description: Test BertTokenizer op with invalid parameter types
    Expectation: Raise expected exceptions for invalid parameter types
    """
    # test vocab is list
    data = {"张三": 18, "王五": 20}
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    with pytest.raises(TypeError, match=r"Invalid user input. Got \<class 'dict'\>: \{'张三': 18, '王五': 20\}, "
                                        "cannot be converted into tensor"):
        _ = tokenizer_op(data)

    # test suffix_indicator  is not str
    vocab = text.Vocab.from_list(vocab_bert)
    with pytest.raises(TypeError, match="Wrong input type for suffix_indicator, should be string"):
        _ = text.BertTokenizer(vocab=vocab, suffix_indicator=1)

    # test max_bytes_per_token  is not int
    vocab = text.Vocab.from_list(vocab_bert)
    with pytest.raises(TypeError, match="Wrong input type for max_bytes_per_token, should be int"):
        _ = text.BertTokenizer(vocab=vocab, max_bytes_per_token="1")

    # test unknown_token  is not str
    vocab = text.Vocab.from_list(vocab_bert)
    with pytest.raises(TypeError, match="Wrong input type for unknown_token, should be string"):
        _ = text.BertTokenizer(vocab=vocab, unknown_token=True)

    # test lower_case   is not bool
    vocab = text.Vocab.from_list(vocab_bert)
    with pytest.raises(TypeError, match="Wrong input type for lower_case, should be boolean"):
        _ = text.BertTokenizer(vocab=vocab, lower_case="a")

    # test preserve_unused_token  is not bool
    vocab = text.Vocab.from_list(vocab_bert)
    with pytest.raises(TypeError, match="Wrong input type for preserve_unused_token, should be boolean"):
        _ = text.BertTokenizer(vocab=vocab, preserve_unused_token=" ")

    # test with_offsets is not bool
    vocab = text.Vocab.from_list(vocab_bert)
    with pytest.raises(TypeError, match="Wrong input type for with_offsets, should be boolean"):
        _ = text.BertTokenizer(vocab=vocab, with_offsets="Ture")

    # invalid input
    data = {'张三': 18, '王五': 20}
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)

    with pytest.raises(TypeError) as info:
        _ = tokenizer_op(data)
    assert "Invalid user input. Got <class 'dict'>: {'张三': 18, '王五': 20}, cannot be converted into tensor." in str(info)
