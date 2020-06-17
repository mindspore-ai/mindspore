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
Testing BasicTokenizer op in DE
"""
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.text as nlp

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
          '중국', '봉건', '왕조의', '역사에서', '마지막', '두', '왕조였다']]
    ),
    dict(
        first=7,
        last=7,
        expected_tokens=[['this', 'is', 'a', 'funky', 'string']],
        lower_case=True
    ),
]


def check_basic_tokenizer(first, last, expected_tokens, lower_case=False, keep_whitespace=False,
                          normalization_form=nlp.utils.NormalizeForm.NONE, preserve_unused_token=False):
    dataset = ds.TextFileDataset(BASIC_TOKENIZER_FILE, shuffle=False)
    if first > 1:
        dataset = dataset.skip(first - 1)
    if last >= first:
        dataset = dataset.take(last - first + 1)

    basic_tokenizer = nlp.BasicTokenizer(lower_case=lower_case,
                                         keep_whitespace=keep_whitespace,
                                         normalization_form=normalization_form,
                                         preserve_unused_token=preserve_unused_token)

    dataset = dataset.map(operations=basic_tokenizer)
    count = 0
    for i in dataset.create_dict_iterator():
        text = nlp.to_str(i['text'])
        logger.info("Out:", text)
        logger.info("Exp:", expected_tokens[count])
        np.testing.assert_array_equal(text, expected_tokens[count])
        count = count + 1


def test_basic_tokenizer():
    """
    Test BasicTokenizer
    """
    for paras in test_paras:
        check_basic_tokenizer(**paras)


if __name__ == '__main__':
    test_basic_tokenizer()
