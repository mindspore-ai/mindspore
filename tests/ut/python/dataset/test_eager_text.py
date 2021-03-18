# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.dataset.text.transforms as T
import mindspore.common.dtype as mstype
from mindspore import log as logger

def test_sliding_window():
    txt = ["Welcome", "to", "Beijing", "!"]
    sliding_window = T.SlidingWindow(width=2)
    txt = sliding_window(txt)
    logger.info("Result: {}".format(txt))

    expected = [['Welcome', 'to'], ['to', 'Beijing'], ['Beijing', '!']]
    np.testing.assert_equal(txt, expected)


def test_to_number():
    txt = ["123456"]
    to_number = T.ToNumber(mstype.int32)
    txt = to_number(txt)
    logger.info("Result: {}, type: {}".format(txt, type(txt[0])))

    assert txt == 123456


def test_whitespace_tokenizer():
    txt = "Welcome to Beijing !"
    txt = T.WhitespaceTokenizer()(txt)
    logger.info("Tokenize result: {}".format(txt))

    expected = ['Welcome', 'to', 'Beijing', '!']
    np.testing.assert_equal(txt, expected)


def test_python_tokenizer():
    # whitespace tokenizer
    def my_tokenizer(line):
        words = line.split()
        if not words:
            return [""]
        return words
    txt1 = np.array("Welcome to Beijing !".encode())
    txt1 = T.PythonTokenizer(my_tokenizer)(txt1)
    logger.info("Tokenize result: {}".format(txt1))

    txt2 = np.array("Welcome to Beijing !")
    txt2 = T.PythonTokenizer(my_tokenizer)(txt2)
    logger.info("Tokenize result: {}".format(txt2))

    expected = ['Welcome', 'to', 'Beijing', '!']
    np.testing.assert_equal(txt1, expected)
    np.testing.assert_equal(txt2, expected)


if __name__ == '__main__':
    test_sliding_window()
    test_to_number()
    test_whitespace_tokenizer()
    test_python_tokenizer()
