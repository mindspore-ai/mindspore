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
Testing WordpieceTokenizer op in DE
"""
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.text as text

WORDPIECE_TOKENIZER_FILE = "../data/dataset/testTokenizerData/wordpiece_tokenizer.txt"

vocab_english = [
    "book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"
]

vocab_chinese = [
    "我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情'
]

vocab_mix = vocab_chinese + vocab_english

test_paras = [
    dict(
        first=1,
        last=10,
        expect_str=[['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                    ['era'], ['[UNK]']],
        expected_offsets_start=[[0], [0, 5], [0], [0], [0], [0, 3], [0], [0], [0], [0]],
        expected_offsets_limit=[[2], [5, 8], [4], [2], [4], [3, 6], [3], [7], [3], [4]],
        vocab_list=vocab_english
    ),
    dict(
        first=1,
        last=10,
        expect_str=[['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                    ['era'], ['what']],
        expected_offsets_start=[[0], [0, 5], [0], [0], [0], [0, 3], [0], [0], [0], [0]],
        expected_offsets_limit=[[2], [5, 8], [4], [2], [4], [3, 6], [3], [7], [3], [4]],
        vocab_list=vocab_english,
        unknown_token=""
    ),
    dict(
        first=1,
        last=10,
        expect_str=[['my'], ['[UNK]'], ['book'], ['is'], ['love'], ['[UNK]'], ['the'], ['[UNK]'], ['era'], ['[UNK]']],
        expected_offsets_start=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
        expected_offsets_limit=[[2], [5], [4], [2], [4], [5], [3], [5], [3], [4]],
        vocab_list=vocab_english,
        max_bytes_per_token=4
    ),
    dict(
        first=11,
        last=25,
        expect_str=[['我'], ['最'], ['喜'], ['欢'], ['的'], ['书'], ['是'], ['霍'], ['乱'], ['时'], ['期'], ['的'], ['爱'], ['情'],
                    ['[UNK]']],
        expected_offsets_start=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
        expected_offsets_limit=[[3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3]],
        vocab_list=vocab_chinese,
    ),
    dict(
        first=25,
        last=25,
        expect_str=[['您']],
        expected_offsets_start=[[0]],
        expected_offsets_limit=[[3]],
        vocab_list=vocab_chinese,
        unknown_token=""
    ),
    dict(
        first=1,
        last=25,
        expect_str=[
            ['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'], ['era'],
            ['[UNK]'],
            ['我'], ['最'], ['喜'], ['欢'], ['的'], ['书'], ['是'], ['霍'], ['乱'], ['时'], ['期'], ['的'], ['爱'], ['情'],
            ['[UNK]']],
        expected_offsets_start=[[0], [0, 5], [0], [0], [0], [0, 3], [0], [0], [0], [0],
                                [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
        expected_offsets_limit=[[2], [5, 8], [4], [2], [4], [3, 6], [3], [7], [3], [4],
                                [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3]],
        vocab_list=vocab_mix,
    ),
]


def check_wordpiece_tokenizer_default(first, last, expect_str, expected_offsets_start, expected_offsets_limit,
                                      vocab_list, unknown_token='[UNK]', max_bytes_per_token=100):
    dataset = ds.TextFileDataset(WORDPIECE_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)
    vocab = text.Vocab.from_list(vocab_list)
    tokenizer_op = text.WordpieceTokenizer(vocab=vocab, unknown_token=unknown_token,
                                           max_bytes_per_token=max_bytes_per_token)
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text']
        logger.info("Out:", token)
        logger.info("Exp:", expect_str[count])
        np.testing.assert_array_equal(token, expect_str[count])
        count = count + 1


def check_wordpiece_tokenizer_with_offsets(first, last, expect_str, expected_offsets_start, expected_offsets_limit,
                                           vocab_list, unknown_token='[UNK]', max_bytes_per_token=100):
    dataset = ds.TextFileDataset(WORDPIECE_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)
    vocab = text.Vocab.from_list(vocab_list)
    tokenizer_op = text.WordpieceTokenizer(vocab=vocab, with_offsets=True, unknown_token=unknown_token,
                                           max_bytes_per_token=max_bytes_per_token)
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


def test_wordpiece_tokenizer_default():
    """
    Feature: WordpieceTokenizer
    Description: Test WordpieceTokenizer with default parameters
    Expectation: Output is equal to the expected output
    """
    for paras in test_paras:
        check_wordpiece_tokenizer_default(**paras)


def test_wordpiece_tokenizer_with_offsets():
    """
    Feature: WordpieceTokenizer
    Description: Test WordpieceTokenizer by setting with_offsets to True
    Expectation: Output is equal to the expected output
    """
    for paras in test_paras:
        check_wordpiece_tokenizer_with_offsets(**paras)


if __name__ == '__main__':
    test_wordpiece_tokenizer_default()
    test_wordpiece_tokenizer_with_offsets()
