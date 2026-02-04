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
"""text transform - to_bytes"""

import numpy as np
import pytest
from mindspore.dataset import text

chinese = np.array(["今天天气太好了我们一起去外面玩吧",
                    "男默女泪",
                    "江州市长江大桥参加了长江大桥的通车仪式"])

english = np.array(["This is a text file.",
                    "Be happy every day.",
                    "Good luck to everyone."])

words = np.array([["This", "text", "file", "a"],
                  ["Be", "happy", "day", "b"],
                  ["女", "", "everyone", "c"]])

number = np.random.randn(4, 8).astype(np.float32)


def test_to_bytes_operation_01():
    """
    Feature: to_bytes op
    Description: Test to_bytes op with different text types and encodings
    Expectation: Successfully convert strings to bytes
    """
    # Test to_bytes, chinese
    text.utils.to_bytes(chinese)

    # Test to_bytes, english
    text.utils.to_bytes(english)

    # Test to_bytes, english, encoding utf-16
    text.utils.to_bytes(english, encoding='utf-16')


def test_to_bytes_exception_01():
    """
    Feature: to_bytes op
    Description: Test to_bytes op with invalid input types
    Expectation: Raise expected exceptions for non-string inputs
    """
    # Test to_bytes, number
    with pytest.raises(TypeError):
        text.utils.to_bytes(number)

    # Test to_bytes, encoding num
    with pytest.raises(TypeError):
        text.utils.to_bytes(words, encoding=16)
