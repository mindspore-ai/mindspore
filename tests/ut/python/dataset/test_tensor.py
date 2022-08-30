# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

import mindspore._c_dataengine as cde


def test_shape():
    """
    Feature: TensorShape op
    Description: Test TensorShape op basic usage
    Expectation: Output is equal to the expected output
    """
    x = [2, 3]
    s = cde.TensorShape(x)
    assert s.as_list() == x
    assert s.is_known()


def test_basic():
    """
    Feature: Tensor
    Description: Test Tensor basic usage
    Expectation: Output is equal to the expected output
    """
    x = np.array([1, 2, 3, 4, 5])
    n = cde.Tensor(x)
    arr = np.array(n, copy=False)
    arr[0] = 0
    x = np.array([0, 2, 3, 4, 5])

    np.testing.assert_array_equal(x, arr)
    assert n.type() == cde.DataType("int64")

    arr2 = n.as_array()
    # decoding only impacts string arrays
    arr3 = n.as_array()
    arr[0] = 2
    x = np.array([2, 2, 3, 4, 5])
    np.testing.assert_array_equal(x, arr2)
    np.testing.assert_array_equal(x, arr3)
    assert n.type() == cde.DataType("int64")
    assert arr.__array_interface__['data'] == arr2.__array_interface__['data']
    assert arr.__array_interface__['data'] == arr3.__array_interface__['data']


def test_strides():
    """
    Feature: Tensor
    Description: Test Tensor basic usage with strides
    Expectation: Output is equal to the expected output
    """
    x = np.array([[1, 2, 3], [4, 5, 6]])
    n1 = cde.Tensor(x[:, 1])
    arr = np.array(n1, copy=False)

    np.testing.assert_array_equal(x[:, 1], arr)

    n2 = cde.Tensor(x.transpose())
    arr = np.array(n2, copy=False)

    np.testing.assert_array_equal(x.transpose(), arr)


if __name__ == '__main__':
    test_shape()
    test_strides()
    test_basic()
