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
Testing UnicodeCharTokenizer op in DE
"""
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.transforms.text.c_transforms as nlp
import mindspore.dataset.transforms.text.utils as nlp_util

DATA_FILE = "../data/dataset/testTokenizerData/1.txt"


def split_by_unicode_char(input_strs):
    """
    Split utf-8 strings to unicode characters
    """
    out = []
    for s in input_strs:
        out.append([c for c in s])
    return out


def test_unicode_char_tokenizer():
    """
    Test UnicodeCharTokenizer
    """
    input_strs = ("Welcome to Beijing!", "北京欢迎您！", "我喜欢English!", "  ")
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = nlp.UnicodeCharTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator():
        text = nlp_util.as_text(i['text']).tolist()
        tokens.append(text)
    logger.info("The out tokens is : {}".format(tokens))
    assert split_by_unicode_char(input_strs) == tokens


if __name__ == '__main__':
    test_unicode_char_tokenizer()
