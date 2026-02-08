# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""test text eager"""

import numpy as np
import pytest

import mindspore
from mindspore import log as logger
import mindspore.common.dtype as mstype
from mindspore.dataset import text


def test_sliding_window():
    """
    Feature: SlidingWindow op
    Description: Test SlidingWindow op basic usage
    Expectation: Output is equal to the expected output
    """
    txt = ["Welcome", "to", "Beijing", "!"]
    sliding_window = text.SlidingWindow(width=2)
    txt = sliding_window(txt)
    logger.info("Result: {}".format(txt))

    expected = [['Welcome', 'to'], ['to', 'Beijing'], ['Beijing', '!']]
    np.testing.assert_equal(txt, expected)


def test_to_number():
    """
    Feature: ToNumber op
    Description: Test ToNumber op basic usage
    Expectation: Output is equal to the expected output
    """
    txt = ["123456"]
    to_number = text.ToNumber(mstype.int32)
    txt = to_number(txt)
    logger.info("Result: {}, type: {}".format(txt, type(txt[0])))

    assert txt == 123456


def test_whitespace_tokenizer():
    """
    Feature: WhitespaceTokenizer op
    Description: Test WhitespaceTokenizer op basic usage
    Expectation: Output is equal to the expected output
    """
    txt = "Welcome to Beijing !"
    txt = text.WhitespaceTokenizer()(txt)
    logger.info("Tokenize result: {}".format(txt))

    expected = ['Welcome', 'to', 'Beijing', '!']
    np.testing.assert_equal(txt, expected)


def test_python_tokenizer():
    """
    Feature: PythonTokenizer op
    Description: Test PythonTokenizer op basic usage
    Expectation: Output is equal to the expected output
    """

    # whitespace tokenizer
    def my_tokenizer(line):
        words = line.split()
        if not words:
            return [""]
        return words

    txt1 = np.array("Welcome to Beijing !".encode())
    txt1 = text.PythonTokenizer(my_tokenizer)(txt1)
    logger.info("Tokenize result: {}".format(txt1))

    txt2 = np.array("Welcome to Beijing !")
    txt2 = text.PythonTokenizer(my_tokenizer)(txt2)
    logger.info("Tokenize result: {}".format(txt2))

    expected = ['Welcome', 'to', 'Beijing', '!']
    np.testing.assert_equal(txt1, expected)
    np.testing.assert_equal(txt2, expected)


def test_raising_error_for_bytes():
    """
    Feature: Text operations
    Description: Test applying text operations on bytes input
    Expectation: Raise errors that bytes input is not supported
    """
    txt = b"/x56/x34/x78/x66"

    def process_bytes(operation):
        with pytest.raises(RuntimeError) as e:
            _ = operation(txt)
        assert "type" in str(e.value) and "string" in str(e.value)

    process_bytes(text.BasicTokenizer())
    process_bytes(text.CaseFold())
    process_bytes(text.FilterWikipediaXML())
    process_bytes(text.Ngram([3]))
    process_bytes(text.NormalizeUTF8())
    process_bytes(text.RegexReplace("/x56", "/x78"))
    process_bytes(text.ToNumber(mindspore.dtype.int8))
    process_bytes(text.UnicodeCharTokenizer())
    process_bytes(text.UnicodeScriptTokenizer())
    process_bytes(text.WhitespaceTokenizer())


if __name__ == '__main__':
    test_sliding_window()
    test_to_number()
    test_whitespace_tokenizer()
    test_python_tokenizer()
    test_raising_error_for_bytes()
