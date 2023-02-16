# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Test Tensor check_input_data_type"""
import pytest
import numpy as np

from mindspore import Tensor


def test_convert_to_tensor_by_structured_array():
    """
    Feature: Check the type of input_data for Tensor.
    Description: Convert to Tensor by structured array.
    Expectation: Throw TypeError.
    """
    a = np.array([('x', 1), ('y', 2)], dtype=[('name', '<U10'), ('value', '<i4')])
    with pytest.raises(TypeError) as ex:
        Tensor(a)
    assert "initializing tensor by numpy array failed" in str(ex.value)
    assert "<class 'numpy.void'>" in str(ex.value)


def test_convert_to_tensor_by_object_type_array():
    """
    Feature: Check the type of input_data for Tensor.
    Description: Convert to Tensor by object type.
    Expectation: Throw TypeError.
    """
    a = np.array([[1, 2, 3], [4, Tensor(5), 6], [7, 8, 9]], dtype=object)
    with pytest.raises(TypeError) as ex:
        Tensor(a)
    assert "initializing tensor by numpy array failed" in str(ex.value)
    assert "<class 'mindspore.common.tensor.Tensor'>" in str(ex.value)
