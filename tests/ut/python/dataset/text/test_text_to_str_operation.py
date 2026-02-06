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
"""text transform - to_str"""

import numpy as np
import pytest
from mindspore.dataset import text
from mindspore import Tensor


def test_to_str_operation_01():
    """
    Feature: to_str op
    Description: Test to_str op with byte arrays
    Expectation: Successfully convert byte arrays to strings
    """
    # no doc
    array = np.array(['4', '5', '6']).astype("S")
    out = text.to_str(array, encoding='ascii')
    np.testing.assert_array_equal(array.astype("U"), out)


def test_to_str_exception_01():
    """
    Feature: to_str op
    Description: Test to_str op with invalid input types
    Expectation: Raise expected exceptions for non-array inputs
    """
    # no doc
    str_list = ['4', '5', '6']
    with pytest.raises(ValueError, match="input should be a NumPy array."):
        text.to_bytes(str_list)

    # no doc
    tensor = Tensor([4, 5, 6])
    with pytest.raises(ValueError, match="input should be a NumPy array."):
        text.to_bytes(tensor)

    # no doc
    array = np.array([1, 2, 3]).astype(np.float32)
    with pytest.raises(TypeError, match="string operation on non-string array"):
        text.to_bytes(array)

    # no doc
    array = np.array(['4', '5', '6']).astype("S")
    with pytest.raises(AttributeError):
        text.to_bytes(array, encoding='ut8')

    # no doc
    array = np.array(['4', '5', '6']).astype("S")
    with pytest.raises(AttributeError):
        text.to_bytes(array, encoding=1)
