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
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.text as nlp

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
        text = nlp.to_str(i['text']).tolist()
        tokens.append(text)
    logger.info("The out tokens is : {}".format(tokens))
    assert split_by_unicode_char(input_strs) == tokens


def test_whitespace_tokenizer():
    """
    Test WhitespaceTokenizer
    """
    whitespace_strs = [["Welcome", "to", "Beijing!"],
                       ["北京欢迎您！"],
                       ["我喜欢English!"],
                       [""]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = nlp.WhitespaceTokenizer()
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator():
        text = nlp.to_str(i['text']).tolist()
        tokens.append(text)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens


def test_unicode_script_tokenizer():
    """
    Test UnicodeScriptTokenizer when para keep_whitespace=False
    """
    unicode_script_strs = [["Welcome", "to", "Beijing", "!"],
                           ["北京欢迎您", "！"],
                           ["我喜欢", "English", "!"],
                           [""]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = nlp.UnicodeScriptTokenizer(keep_whitespace=False)
    dataset = dataset.map(operations=tokenizer)

    tokens = []
    for i in dataset.create_dict_iterator():
        text = nlp.to_str(i['text']).tolist()
        tokens.append(text)
    logger.info("The out tokens is : {}".format(tokens))
    assert unicode_script_strs == tokens


def test_unicode_script_tokenizer2():
    """
    Test UnicodeScriptTokenizer when para keep_whitespace=True
    """
    unicode_script_strs2 = [["Welcome", " ", "to", " ", "Beijing", "!"],
                            ["北京欢迎您", "！"],
                            ["我喜欢", "English", "!"],
                            ["  "]]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = nlp.UnicodeScriptTokenizer(keep_whitespace=True)
    dataset = dataset.map(operations=tokenizer)
    tokens = []
    for i in dataset.create_dict_iterator():
        text = nlp.to_str(i['text']).tolist()
        tokens.append(text)
    logger.info("The out tokens is :", tokens)
    assert unicode_script_strs2 == tokens


def test_case_fold():
    """
    Test CaseFold
    """
    expect_strs = ["welcome to beijing!", "北京欢迎您!", "我喜欢english!", "  "]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    op = nlp.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator():
        text = nlp.to_str(i['text']).tolist()
        lower_strs.append(text)
    assert lower_strs == expect_strs


def test_normalize_utf8():
    """
    Test NormalizeUTF8
    """

    def normalize(normalize_form):
        dataset = ds.TextFileDataset(NORMALIZE_FILE, shuffle=False)
        normalize = nlp.NormalizeUTF8(normalize_form=normalize_form)
        dataset = dataset.map(operations=normalize)
        out_bytes = []
        out_texts = []
        for i in dataset.create_dict_iterator():
            out_bytes.append(i['text'])
            out_texts.append(nlp.to_str(i['text']).tolist())
        logger.info("The out bytes is : ", out_bytes)
        logger.info("The out texts is: ", out_texts)
        return out_bytes

    expect_normlize_data = [
        # NFC
        [b'\xe1\xb9\xa9', b'\xe1\xb8\x8d\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'\xef\xac\x81', b'2\xe2\x81\xb5', b'\xe1\xba\x9b\xcc\xa3'],
        # NFKC
        [b'\xe1\xb9\xa9', b'\xe1\xb8\x8d\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'fi', b'25', b'\xe1\xb9\xa9'],
        # NFD
        [b's\xcc\xa3\xcc\x87', b'd\xcc\xa3\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'\xef\xac\x81', b'2\xe2\x81\xb5', b'\xc5\xbf\xcc\xa3\xcc\x87'],
        # NFKD
        [b's\xcc\xa3\xcc\x87', b'd\xcc\xa3\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'fi', b'25', b's\xcc\xa3\xcc\x87']
    ]
    assert normalize(nlp.utils.NormalizeForm.NFC) == expect_normlize_data[0]
    assert normalize(nlp.utils.NormalizeForm.NFKC) == expect_normlize_data[1]
    assert normalize(nlp.utils.NormalizeForm.NFD) == expect_normlize_data[2]
    assert normalize(nlp.utils.NormalizeForm.NFKD) == expect_normlize_data[3]


def test_regex_replace():
    """
    Test RegexReplace
    """

    def regex_replace(first, last, expect_str, pattern, replace):
        dataset = ds.TextFileDataset(REGEX_REPLACE_FILE, shuffle=False)
        if first > 1:
            dataset = dataset.skip(first - 1)
        if last >= first:
            dataset = dataset.take(last - first + 1)
        replace_op = nlp.RegexReplace(pattern, replace)
        dataset = dataset.map(operations=replace_op)
        out_text = []
        for i in dataset.create_dict_iterator():
            text = nlp.to_str(i['text']).tolist()
            out_text.append(text)
        logger.info("Out:", out_text)
        logger.info("Exp:", expect_str)
        assert expect_str == out_text

    regex_replace(1, 2, ['H____ W____', "L__'_ G_"], "\\p{Ll}", '_')
    regex_replace(3, 5, ['hello', 'world', '31:beijing'], "^(\\d:|b:)", "")
    regex_replace(6, 6, ["WelcometoChina!"], "\\s+", "")
    regex_replace(7, 8, ['我不想长大', 'WelcometoShenzhen!'], "\\p{Cc}|\\p{Cf}|\\s+", "")


def test_regex_tokenizer():
    """
    Test RegexTokenizer
    """

    def regex_tokenizer(first, last, expect_str, delim_pattern, keep_delim_pattern):
        dataset = ds.TextFileDataset(REGEX_TOKENIZER_FILE, shuffle=False)
        if first > 1:
            dataset = dataset.skip(first - 1)
        if last >= first:
            dataset = dataset.take(last - first + 1)
        tokenizer_op = nlp.RegexTokenizer(delim_pattern, keep_delim_pattern)
        dataset = dataset.map(operations=tokenizer_op)
        out_text = []
        count = 0
        for i in dataset.create_dict_iterator():
            text = nlp.to_str(i['text']).tolist()
            np.testing.assert_array_equal(text, expect_str[count])
            count += 1
            out_text.append(text)
        logger.info("Out:", out_text)
        logger.info("Exp:", expect_str)

    regex_tokenizer(1, 1, [['Welcome', 'to', 'Shenzhen!']], "\\s+", "")
    regex_tokenizer(1, 1, [['Welcome', ' ', 'to', ' ', 'Shenzhen!']], "\\s+", "\\s+")
    regex_tokenizer(2, 2, [['北', '京', '欢', '迎', '您', '!Welcome to Beijing!']], r"\p{Han}", r"\p{Han}")
    regex_tokenizer(3, 3, [['12', '￥+', '36', '￥=?']], r"[\p{P}|\p{S}]+", r"[\p{P}|\p{S}]+")
    regex_tokenizer(3, 3, [['12', '36']], r"[\p{P}|\p{S}]+", "")
    regex_tokenizer(3, 3, [['￥+', '￥=?']], r"[\p{N}]+", "")


if __name__ == '__main__':
    test_unicode_char_tokenizer()
    test_whitespace_tokenizer()
    test_unicode_script_tokenizer()
    test_unicode_script_tokenizer2()
    test_case_fold()
    test_normalize_utf8()
    test_regex_replace()
    test_regex_tokenizer()
