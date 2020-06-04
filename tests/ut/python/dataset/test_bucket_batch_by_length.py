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

import pytest
import numpy as np
import mindspore.dataset as ds

# generates 1 column [0], [0, 1], ..., [0, ..., n-1]
def generate_sequential(n):
    for i in range(n):
        yield (np.array([j for j in range(i + 1)]),)


# generates 1 column [0], [1], ..., [n-1]
def generate_sequential_same_shape(n):
    for i in range(n):
        yield (np.array([i]),)


# combines generate_sequential_same_shape and generate_sequential
def generate_2_columns(n):
    for i in range(n):
        yield (np.array([i]), np.array([j for j in range(i + 1)]))


def test_bucket_batch_invalid_input():
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    invalid_column_names = [1, 2, 3]

    bucket_boundaries = [1, 2, 3]
    empty_bucket_boundaries = []
    invalid_bucket_boundaries = ["1", "2", "3"]
    negative_bucket_boundaries = [1, 2, -3]
    decreasing_bucket_boundaries = [3, 2, 1]
    non_increasing_bucket_boundaries = [1, 2, 2]

    bucket_batch_sizes = [1, 1, 1, 1]
    invalid_bucket_batch_sizes = ["1", "2", "3", "4"]
    negative_bucket_batch_sizes = [1, 2, 3, -4]

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(invalid_column_names, bucket_boundaries, bucket_batch_sizes)
    assert "column_names should be a list of str" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, empty_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries cannot be empty" in str(info.value)

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(column_names, invalid_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries should be a list of int" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, negative_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries cannot contain any negative numbers" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, decreasing_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries should be strictly increasing" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, non_increasing_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries should be strictly increasing" in str(info.value)

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, invalid_bucket_batch_sizes)
    assert "bucket_batch_sizes should be a list of int" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, negative_bucket_batch_sizes)
    assert "bucket_batch_sizes cannot contain any negative numbers" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, bucket_boundaries)
    assert "bucket_batch_sizes must contain one element more than bucket_boundaries" in str(info.value)


def test_bucket_batch_multi_bucket_no_padding():
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [3, 3, 2, 2]
    element_length_function = (lambda x: x[0] % 4)

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function)

    expected_output = [[[2], [6]],
                       [[3], [7]],
                       [[0], [4], [8]],
                       [[1], [5], [9]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_multi_bucket_with_padding():
    dataset = ds.GeneratorDataset((lambda: generate_sequential(10)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [2, 3, 3, 2]
    element_length_function = (lambda x: len(x) % 4)
    pad_info = {"col1": ([10], 0)}

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info)

    expected_output = [[[0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]],
                       [[0, 1, 2, 3, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 0, 0]],
                       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]],
                       [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_single_bucket_no_padding():
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [1, 1, 5, 1]
    element_length_function = (lambda x: 2)

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function)

    expected_output = [[[0], [1], [2], [3], [4]],
                       [[5], [6], [7], [8], [9]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_single_bucket_with_padding():
    dataset = ds.GeneratorDataset((lambda: generate_sequential(9)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [1, 1, 1, 3]
    element_length_function = (lambda x: 7)
    pad_info = {"col1": ([12], 0)}

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info)

    expected_output = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                       [[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]],
                       [[0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_pad_to_bucket_boundary():
    dataset = ds.GeneratorDataset((lambda: generate_sequential(9)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [3, 6, 15]
    bucket_batch_sizes = [2, 3, 4, 1]
    element_length_function = len
    pad_info = {"col1": ([None], 0)}
    pad_to_bucket_boundary = True

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info, pad_to_bucket_boundary)

    expected_output = [[[0, 0],
                        [0, 1]],
                       [[0, 1, 2, 0, 0],
                        [0, 1, 2, 3, 0],
                        [0, 1, 2, 3, 4]],
                       [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_default_pad():
    dataset = ds.GeneratorDataset((lambda: generate_sequential(15)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [5, 8, 17]
    bucket_batch_sizes = [2, 1, 4, 1]
    element_length_function = len
    pad_info = {"col1": ([None], 0)}

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info)

    expected_output = [[[0, 0],
                        [0, 1]],
                       [[0, 1, 2, 0],
                        [0, 1, 2, 3]],
                       [[0, 1, 2, 3, 4]],
                       [[0, 1, 2, 3, 4, 5]],
                       [[0, 1, 2, 3, 4, 5, 6]],
                       [[0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                       [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]]


    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_drop_remainder():
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(27)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2]
    bucket_batch_sizes = [2, 3, 5]
    element_length_function = (lambda x: x[0] % 3)
    pad_info = None
    pad_to_bucket_boundary = False
    drop_remainder = True

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info, pad_to_bucket_boundary, drop_remainder)

    expected_output = [[[0], [3]],
                       [[1], [4], [7]],
                       [[6], [9]],
                       [[2], [5], [8], [11], [14]],
                       [[12], [15]],
                       [[10], [13], [16]],
                       [[18], [21]],
                       [[19], [22], [25]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_default_length_function():
    dataset = ds.GeneratorDataset((lambda: generate_sequential(9)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [6, 12]
    bucket_batch_sizes = [5, 4, 1]
    element_length_function = None
    pad_info = {}

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info)

    expected_output = [[[0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 2, 0, 0],
                        [0, 1, 2, 3, 0],
                        [0, 1, 2, 3, 4]],
                       [[0, 1, 2, 3, 4, 5, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8]]]

    output = []
    for data in dataset.create_dict_iterator():
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_multi_column():
    dataset = ds.GeneratorDataset((lambda: generate_2_columns(10)), ["same_shape", "variable_shape"])

    column_names = ["same_shape"]
    bucket_boundaries = [6, 12]
    bucket_batch_sizes = [5, 5, 1]
    element_length_function = None
    pad_info = {}

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info)

    same_shape_expected_output = [[[0], [1], [2], [3], [4]],
                                  [[5], [6], [7], [8], [9]]]

    variable_shape_expected_output = [[[0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0],
                                       [0, 1, 2, 0, 0],
                                       [0, 1, 2, 3, 0],
                                       [0, 1, 2, 3, 4]],
                                      [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
                                       [0, 1, 2, 3, 4, 5, 6, 0, 0, 0],
                                       [0, 1, 2, 3, 4, 5, 6, 7, 0, 0],
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]

    same_shape_output = []
    variable_shape_output = []
    for data in dataset.create_dict_iterator():
        same_shape_output.append(data["same_shape"].tolist())
        variable_shape_output.append(data["variable_shape"].tolist())

    assert same_shape_output == same_shape_expected_output
    assert variable_shape_output == variable_shape_expected_output


if __name__ == '__main__':
    test_bucket_batch_invalid_input()
    test_bucket_batch_multi_bucket_no_padding()
    test_bucket_batch_multi_bucket_with_padding()
    test_bucket_batch_single_bucket_no_padding()
    test_bucket_batch_single_bucket_with_padding()
    test_bucket_batch_pad_to_bucket_boundary()
    test_bucket_batch_default_pad()
    test_bucket_batch_drop_remainder()
    test_bucket_batch_default_length_function()
    test_bucket_batch_multi_column()
