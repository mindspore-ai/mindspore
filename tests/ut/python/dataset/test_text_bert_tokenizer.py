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
"""
Testing BertTokenizer op in DE
"""
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.text as text

BERT_TOKENIZER_FILE = "../data/dataset/testTokenizerData/bert_tokenizer.txt"

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
    dict(
        first=1,
        last=4,
        expect_str=[['床', '前', '明', '月', '光'],
                    ['疑', '是', '地', '上', '霜'],
                    ['举', '头', '望', '明', '月'],
                    ['低', '头', '思', '故', '乡']],
        expected_offsets_start=[[0, 3, 6, 9, 12],
                                [0, 3, 6, 9, 12],
                                [0, 3, 6, 9, 12],
                                [0, 3, 6, 9, 12]],
        expected_offsets_limit=[[3, 6, 9, 12, 15],
                                [3, 6, 9, 12, 15],
                                [3, 6, 9, 12, 15],
                                [3, 6, 9, 12, 15]],
        vocab_list=vocab_bert
    ),
    # test english text
    dict(
        first=5,
        last=5,
        expect_str=[['i', 'am', 'mak', '##ing', 'small', 'mistake', '##s', 'during', 'work', '##ing', 'hour', '##s']],
        expected_offsets_start=[[0, 2, 5, 8, 12, 18, 25, 27, 34, 38, 42, 46]],
        expected_offsets_limit=[[1, 4, 8, 11, 17, 25, 26, 33, 38, 41, 46, 47]],
        lower_case=True,
        vocab_list=vocab_bert
    ),
    dict(
        first=5,
        last=5,
        expect_str=[['I', "am", 'mak', '##ing', 'small', 'mistake', '##s', 'during', 'work', '##ing', 'hour', '##s']],
        expected_offsets_start=[[0, 2, 5, 8, 12, 18, 25, 27, 34, 38, 42, 46]],
        expected_offsets_limit=[[1, 4, 8, 11, 17, 25, 26, 33, 38, 41, 46, 47]],
        lower_case=False,
        vocab_list=vocab_bert
    ),
    # test emoji tokens
    dict(
        first=6,
        last=7,
        expect_str=[
            ['😀', '嘿', '嘿', '😃', '哈', '哈', '😄', '大', '笑', '😁', '嘻', '嘻'],
            ['繁', '體', '字']],
        expected_offsets_start=[[0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37], [0, 3, 6]],
        expected_offsets_limit=[[4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40], [3, 6, 9]],
        normalization_form=text.utils.NormalizeForm.NFKC,
        vocab_list=vocab_bert
    ),
    # test preserved tokens
    dict(
        first=8,
        last=14,
        expect_str=[
            ['[UNK]', '[CLS]'],
            ['[UNK]', '[SEP]'],
            ['[UNK]', '[UNK]'],
            ['[UNK]', '[PAD]'],
            ['[UNK]', '[MASK]'],
            ['[unused1]'],
            ['[unused10]']
        ],
        expected_offsets_start=[[0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0], [0]],
        expected_offsets_limit=[[6, 12], [6, 12], [6, 12], [6, 12], [6, 13], [9], [10]],
        lower_case=False,
        vocab_list=vocab_bert,
        preserve_unused_token=True,
    ),
    dict(
        first=8,
        last=14,
        expect_str=[
            ['[UNK]', '[CLS]'],
            ['[UNK]', '[SEP]'],
            ['[UNK]', '[UNK]'],
            ['[UNK]', '[PAD]'],
            ['[UNK]', '[MASK]'],
            ['[unused1]'],
            ['[unused10]']
        ],
        expected_offsets_start=[[0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0], [0]],
        expected_offsets_limit=[[6, 12], [6, 12], [6, 12], [6, 12], [6, 13], [9], [10]],
        lower_case=True,
        vocab_list=vocab_bert,
        preserve_unused_token=True,
    ),
    # test special symbol
    dict(
        first=15,
        last=15,
        expect_str=[['12', '+', '/', '-', '28', '=', '40', '/', '-', '16']],
        expected_offsets_start=[[0, 2, 3, 4, 5, 7, 8, 10, 11, 12]],
        expected_offsets_limit=[[2, 3, 4, 5, 7, 8, 10, 11, 12, 14]],
        preserve_unused_token=True,
        vocab_list=vocab_bert
    ),
    # test non-default params
    dict(
        first=8,
        last=8,
        expect_str=[['[UNK]', ' ', '[CLS]']],
        expected_offsets_start=[[0, 6, 7]],
        expected_offsets_limit=[[6, 7, 12]],
        lower_case=False,
        vocab_list=vocab_bert,
        preserve_unused_token=True,
        keep_whitespace=True
    ),
    dict(
        first=8,
        last=8,
        expect_str=[['unused', ' ', '[CLS]']],
        expected_offsets_start=[[0, 6, 7]],
        expected_offsets_limit=[[6, 7, 12]],
        lower_case=False,
        vocab_list=vocab_bert,
        preserve_unused_token=True,
        keep_whitespace=True,
        unknown_token=''
    ),
    dict(
        first=8,
        last=8,
        expect_str=[['unused', ' ', '[', 'CLS', ']']],
        expected_offsets_start=[[0, 6, 7, 8, 11]],
        expected_offsets_limit=[[6, 7, 8, 11, 12]],
        lower_case=False,
        vocab_list=vocab_bert,
        preserve_unused_token=False,
        keep_whitespace=True,
        unknown_token=''
    ),
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
        token = text.to_str(i['text'])
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
                          output_columns=['token', 'offsets_start', 'offsets_limit'],
                          column_order=['token', 'offsets_start', 'offsets_limit'])
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = text.to_str(i['token'])
        logger.info("Out:", token)
        logger.info("Exp:", expect_str[count])
        np.testing.assert_array_equal(token, expect_str[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1


def test_bert_tokenizer_default():
    """
    Test WordpieceTokenizer when with_offsets=False
    """
    for paras in test_paras:
        check_bert_tokenizer_default(**paras)


def test_bert_tokenizer_with_offsets():
    """
    Test WordpieceTokenizer when with_offsets=True
    """
    for paras in test_paras:
        check_bert_tokenizer_with_offsets(**paras)


def test_bert_tokenizer_callable_invalid_input():
    """
    Test WordpieceTokenizer in eager mode with invalid input
    """
    data = {'张三': 18, '王五': 20}
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)

    with pytest.raises(TypeError) as info:
        _ = tokenizer_op(data)
    assert "Invalid user input. Got <class 'dict'>: {'张三': 18, '王五': 20}, cannot be converted into tensor." in str(info)

if __name__ == '__main__':
    test_bert_tokenizer_callable_invalid_input()
    test_bert_tokenizer_default()
    test_bert_tokenizer_with_offsets()
