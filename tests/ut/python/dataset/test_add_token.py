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
Testing AddToken op
"""
import numpy as np

import mindspore.dataset.text as text


def test_add_token_at_begin():
    """
    Feature: AddToken op
    Description: Test AddToken with begin = True
    Expectation: Output is equal to the expected output
    """
    input_one_dimension = ['a', 'b', 'c', 'd', 'e']
    expected = ['TOKEN', 'a', 'b', 'c', 'd', 'e']
    out = text.AddToken(token='TOKEN', begin=True)
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))


def test_add_token_at_end():
    """
    Feature: AddToken op
    Description: Test AddToken with begin = False
    Expectation: Output is equal to the expected output
    """
    input_one_dimension = ['a', 'b', 'c', 'd', 'e']
    expected = ['a', 'b', 'c', 'd', 'e', 'TOKEN']
    out = text.AddToken(token='TOKEN', begin=False)
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))


def test_add_token_fail():
    """
    Feature: AddToken op
    Description: fail to test AddToken
    Expectation: TypeError is raised as expected
    """
    try:
        _ = text.AddToken(token=1.0, begin=True)
    except TypeError as error:
        assert "Argument token with value 1.0 is not of type [<class 'str'>], but got <class 'float'>." in str(error)
    try:
        _ = text.AddToken(token='TOKEN', begin=12.3)
    except TypeError as error:
        assert "Argument begin with value 12.3 is not of type [<class 'bool'>], but got <class 'float'>." in str(error)


if __name__ == "__main__":
    test_add_token_at_begin()
    test_add_token_at_end()
    test_add_token_fail()
