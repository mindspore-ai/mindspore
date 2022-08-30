# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import mindspore.dataset as ds


def test_tensor_empty():
    """
    Feature: Tensor
    Description: Test empty tensor using GeneratorDataset
    Expectation: Output is equal to the expected output
    """

    def gen():
        for _ in range(4):
            yield (np.array([], dtype=np.int64), np.array([1], dtype=np.float64),
                   np.array([], dtype=np.str_).reshape([0, 4]), np.array([], dtype=np.bytes_).reshape([1, 0]))

    data = ds.GeneratorDataset(gen, column_names=["int64", "float64", "str", "bytes"])

    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(np.array([], dtype=np.int64), d[0])
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), d[1])
        np.testing.assert_array_equal(np.array([], dtype=np.str_).reshape([0, 4]), d[2])
        np.testing.assert_array_equal(np.array([], dtype=np.bytes_).reshape([1, 0]), d[3])


def test_tensor_empty_map():
    """
    Feature: Tensor
    Description: Test empty tensor using GeneratorDataset and map it using a function op
    Expectation: Output is equal to the expected output
    """

    def gen():
        for _ in range(4):
            (yield np.array([], dtype=np.int64), np.array([], dtype='S'), np.array([1], dtype=np.float64))

    data = ds.GeneratorDataset(gen, column_names=["col1", "col2", "col3"])

    def func(x, y, z):
        x = np.array([1], dtype=np.int64)
        y = np.array(["Hi"], dtype='S')
        z = np.array([], dtype=np.float64)
        return x, y, z

    data = data.map(operations=func, input_columns=["col1", "col2", "col3"])

    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(np.array([1], dtype=np.int64), d[0])
        np.testing.assert_array_equal(np.array(["Hi"], dtype='S'), d[1])
        np.testing.assert_array_equal(np.array([], dtype=np.float64), d[2])


def test_tensor_empty_batch():
    """
    Feature: Tensor
    Description: Test empty tensor using GeneratorDataset and apply batch op
    Expectation: Output is equal to the expected output
    """

    def gen():
        for _ in range(4):
            (yield np.array([], dtype=np.int64), np.array([], dtype='S').reshape([0, 4]), np.array([1],
                                                                                                   dtype=np.float64))

    data = ds.GeneratorDataset(gen, column_names=["col1", "col2", "col3"]).batch(2)

    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(np.array([], dtype=np.int64).reshape([2, 0]), d[0])
        np.testing.assert_array_equal(np.array([], dtype='S').reshape([2, 0, 4]), d[1])
        np.testing.assert_array_equal(np.array([[1], [1]], dtype=np.float64), d[2])


if __name__ == '__main__':
    test_tensor_empty()
    test_tensor_empty_map()
    test_tensor_empty_batch()
