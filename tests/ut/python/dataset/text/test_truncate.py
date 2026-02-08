# Copyright 2022 Huawei Technologies Co., Ltd
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
Testing Truncate Python API
"""
import numpy as np

from mindspore import log as logger
from mindspore.dataset import text


def test_truncate_max_len_1d():
    """
    Feature: Truncate op
    Description: Test Truncate op using 1D str as the input
    Expectation: Output is equal to the expected output
    """
    truncate = text.Truncate(3)
    input1 = ["1", "2", "3", "4", "5"]
    result1 = truncate(input1)
    expect1 = (["1", "2", "3"])
    assert np.array_equal(result1, expect1)


def test_truncate_max_len_2d():
    """
    Feature: Truncate op
    Description: Test Truncate op using 2D int as the input
    Expectation: Output is equal to the expected output
    """
    truncate = text.Truncate(3)
    input1 = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
    result1 = truncate(input1)
    expect1 = ([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    assert np.array_equal(result1, expect1)


def test_truncate_input_error():
    """
    Feature:Truncate Op
    Description: Test input Error
    Expectation: Throw ValueError, TypeError or RuntimeError exception
    """
    try:
        _ = text.Truncate(-1)
    except ValueError as error:
        logger.info("Got an exception in Truncate: {}".format(str(error)))
        assert "Input max_seq_len is not within the required interval of [1, 2147483647]." in str(
            error)
    try:
        _ = text.Truncate('a')
    except TypeError as error:
        logger.info("Got an exception in Truncate: {}".format(str(error)))
        assert "Argument max_seq_len with value a is not of type [<class 'int'>], but got <class 'str'>." in str(
            error)
    try:
        truncate = text.Truncate(2)
        input1 = [b'1', b'2', b'3', b'4', b'5']
        truncate(input1)
    except RuntimeError as error:
        logger.info("Got an exception in Truncate: {}".format(str(error)))
        assert "Truncate: Truncate: the input tensor should be in type of [bool, int, float, double, string]." in str(
            error)
    try:
        truncate = text.Truncate(2)
        input1 = [[[1, 2, 3]]]
        truncate(input1)
    except RuntimeError as error:
        logger.info("Got an exception in Truncate: {}".format(str(error)))


if __name__ == "__main__":
    test_truncate_max_len_1d()
    test_truncate_max_len_2d()
    test_truncate_input_error()
