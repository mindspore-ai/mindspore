# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Testing BasicTokenizer op in DE
"""
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.text as text

BASIC_TOKENIZER_FILE = "../data/dataset/testTokenizerData/basic_tokenizer.txt"

test_paras = [
    dict(
        first=1,
        last=6,
        expected_tokens=
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
        expected_offsets_start=[[0, 8, 11, 18, 21, 24, 27, 30],
                                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
                                [0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37],
                                [0, 3, 6, 9, 13, 16, 20, 23, 26, 29, 32, 35, 38, 42, 45, 49,
                                 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100],
                                [0, 3, 6, 9, 13, 14, 18, 21, 24, 27, 30, 33, 37, 38, 42, 45, 48, 51,
                                 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 93, 96, 99, 109, 112, 115],
                                [0, 10, 11, 15, 16, 20, 21, 25, 35, 36, 40, 41, 45, 46, 50, 57, 64, 74, 87, 97, 101]],
        expected_offsets_limit=[[7, 10, 18, 21, 24, 27, 30, 33],
                                [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
                                [4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40],
                                [3, 6, 9, 13, 16, 20, 23, 26, 29, 32, 35, 38, 42, 45, 49, 52, 55, 58,
                                 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103],
                                [3, 6, 9, 13, 14, 18, 21, 24, 27, 30, 33, 37, 38, 42, 45, 48, 51, 54,
                                 57, 60, 63, 66, 69, 72, 75, 78, 81, 93, 96, 99, 109, 112, 115, 124],
                                [9, 11, 15, 16, 20, 21, 24, 34, 36, 40, 41, 45, 46, 49, 56, 63, 73, 86, 96, 100, 113]]
    ),
    dict(
        first=7,
        last=7,
        expected_tokens=[['this', 'is', 'a', 'funky', 'string']],
        expected_offsets_start=[[0, 5, 8, 10, 16]],
        expected_offsets_limit=[[4, 7, 9, 15, 22]],
        lower_case=True
    ),
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

def test_basic_tokenizer_with_offsets():
    """
    Feature: BasicTokenizer
    Description: Test BasicTokenizer by setting with_offsets to True
    Expectation: Output is equal to the expected output
    """
    for paras in test_paras:
        check_basic_tokenizer_with_offsets(**paras)


def test_basic_tokenizer_default():
    """
    Feature: BasicTokenizer
    Description: Test BasicTokenizer with default parameters
    Expectation: Output is equal to the expected output
    """
    for paras in test_paras:
        check_basic_tokenizer_default(**paras)


if __name__ == '__main__':
    test_basic_tokenizer_default()
    test_basic_tokenizer_with_offsets()
