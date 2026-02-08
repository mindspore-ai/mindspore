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


def generate_3_columns(n):
    for i in range(n):
        yield (np.array([i]), np.array([i + 1]), np.array([j for j in range(i + 1)]))


def test_bucket_batch_invalid_input():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with invalid inputs
    Expectation: Correct error is raised as expected
    """
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    invalid_column_names = [1, 2, 3]

    bucket_boundaries = [1, 2, 3]
    empty_bucket_boundaries = []
    invalid_bucket_boundaries = ["1", "2", "3"]
    zero_start_bucket_boundaries = [0, 2, 3]
    negative_bucket_boundaries = [1, 2, -3]
    decreasing_bucket_boundaries = [3, 2, 1]
    non_increasing_bucket_boundaries = [1, 2, 2]

    bucket_batch_sizes = [1, 1, 1, 1]
    invalid_bucket_batch_sizes = ["1", "2", "3", "4"]
    negative_bucket_batch_sizes = [1, 2, 3, -4]
    zero_bucket_batch_sizes = [0, 1, 2, 3]

    invalid_type_pad_to_bucket_boundary = ""
    invalid_type_drop_remainder = ""

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(invalid_column_names, bucket_boundaries, bucket_batch_sizes)
    assert "Argument column_names[0] with value 1 is not of type [<class 'str'>]" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, empty_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries cannot be empty" in str(info.value)

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(column_names, invalid_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries should be a list of int" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, zero_start_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries must only contain positive numbers." in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, negative_bucket_boundaries, bucket_batch_sizes)
    assert "bucket_boundaries must only contain positive numbers." in str(info.value)

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
    assert "bucket_batch_sizes should be a list of positive numbers" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, zero_bucket_batch_sizes)
    assert "bucket_batch_sizes should be a list of positive numbers" in str(info.value)

    with pytest.raises(ValueError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, bucket_boundaries)
    assert "bucket_batch_sizes must contain one element more than bucket_boundaries" in str(info.value)

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, bucket_batch_sizes,
                                           None, None, invalid_type_pad_to_bucket_boundary)
    assert "Argument pad_to_bucket_boundary with value \"\" is not of type [<class \'bool\'>]" in str(info.value)

    with pytest.raises(TypeError) as info:
        _ = dataset.bucket_batch_by_length(column_names, bucket_boundaries, bucket_batch_sizes,
                                           None, None, False, invalid_type_drop_remainder)
    assert "Argument drop_remainder with value \"\" is not of type [<class 'bool'>]" in str(info.value)


def test_bucket_batch_multi_bucket_no_padding():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with multi buckets but without padding
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_multi_bucket_no_padding_repeat():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with multi buckets but without padding, followed by repeat op
    Expectation: Output is equal to the expected output
    """
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [3, 3, 2, 2]
    element_length_function = (lambda x: x[0] % 4)

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function)
    dataset = dataset.repeat(2)

    expected_output = [[[2], [6]],
                       [[3], [7]],
                       [[0], [4], [8]],
                       [[1], [5], [9]],
                       [[2], [6]],
                       [[3], [7]],
                       [[0], [4], [8]],
                       [[1], [5], [9]]]

    output = []
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_multi_bucket_with_padding():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with multi buckets and padding
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_single_bucket_no_padding():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with single bucket but without padding
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_single_bucket_with_padding():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with single bucket and padding
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_pad_to_bucket_boundary():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with pad_to_bucket_boundary
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_default_pad():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with default pad
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


@pytest.mark.parametrize("drop_remainder", (True, False))
def test_bucket_batch_drop_remainder(drop_remainder):
    """
    Feature: bucket_batch_by_length
    Description: Test bucket_batch_by_length with drop_remainder
    Expectation: Output is as expected
    """
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(27)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2]
    bucket_batch_sizes = [2, 3, 5]
    element_length_function = (lambda x: x[0] % 3)
    pad_info = None
    pad_to_bucket_boundary = False

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
    if not drop_remainder:
        # append the remainder
        expected_output += [[[24]], [[17], [20], [23], [26]]]

    # iterate at least 2 epochs to check if the remainder is dropped
    num_epochs = 2
    iterator = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    for _ in range(num_epochs):
        output = []
        for data in iterator:
            output.append(data["col1"].tolist())
        assert output == expected_output


def test_bucket_batch_default_length_function():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with default element_length_function
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output.append(data["col1"].tolist())

    assert output == expected_output


def test_bucket_batch_multi_column():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with multi columns
    Expectation: Output is equal to the expected output
    """
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
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        same_shape_output.append(data["same_shape"].tolist())
        variable_shape_output.append(data["variable_shape"].tolist())

    assert same_shape_output == same_shape_expected_output
    assert variable_shape_output == variable_shape_expected_output


def test_bucket_batch_three_columns():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with three columns
    Expectation: Output is equal to the expected output
    """
    dataset = ds.GeneratorDataset((lambda: generate_3_columns(10)), ["same_shape", "same_shape2", "variable_shape"])

    column_names = ["same_shape2"]
    bucket_boundaries = [6, 12]
    bucket_batch_sizes = [5, 5, 1]
    element_length_function = (lambda x: x[0] % 3)
    pad_info = {}

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             pad_info)

    same_shape_expected_output = [[[0], [1], [2], [3], [4]],
                                  [[5], [6], [7], [8], [9]]]
    same_shape2_expected_output = [[[1], [2], [3], [4], [5]],
                                   [[6], [7], [8], [9], [10]]]
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
    same_shape2_output = []
    variable_shape_output = []
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        same_shape_output.append(data["same_shape"].tolist())
        same_shape2_output.append(data["same_shape2"].tolist())
        variable_shape_output.append(data["variable_shape"].tolist())

    assert same_shape_output == same_shape_expected_output
    assert same_shape2_output == same_shape2_expected_output
    assert variable_shape_output == variable_shape_expected_output


def test_bucket_batch_get_dataset_size():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op followed by get_dataset_size op
    Expectation: Output is equal to the expected output
    """
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [3, 3, 2, 2]
    element_length_function = (lambda x: x[0] % 4)

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function)

    data_size = dataset.get_dataset_size()

    num_rows = 0
    for _ in dataset.create_dict_iterator(num_epochs=1):
        num_rows += 1

    assert data_size == num_rows


def test_bucket_batch_with_string():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with string
    Expectation: Output is equal to the expected output
    """
    input_data = ['ab',
                  'abcdefg',
                  ['abcd'],
                  'efgh',
                  ['cd'],
                  [['hijklmn'], ['ef']],
                  'g',
                  ['ijkln']]
    class RandomAccessDataset:
        def __init__(self, input_data):
            self._data = input_data
            self._label = np.zeros((len(self._data), 1))
        def __getitem__(self, index):
            return self._data[index], self._label[index]
        def __len__(self):
            return len(self._data)

    dataset = ds.GeneratorDataset(RandomAccessDataset(input_data), column_names=["data", "label"])

    column_names = ["data"]
    bucket_boundaries = [1, 2]
    bucket_batch_sizes = [4, 3, 1]

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes)

    expected_output = [np.sort(np.array([[['hijklmn'], ['ef']]]), axis=0),
                       np.sort(np.array([['abcd'], ['cd'], ['ijkln']]), axis=0),
                       np.sort(np.array(['ab', 'abcdefg', 'efgh', 'g']), axis=0)]

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        exist = False
        for item in expected_output:
            if data['data'].shape == item.shape and (np.sort(data['data'], axis=0) == item).all():
                exist = True
        assert exist


def test_bucket_batch_with_scalar():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with scalar
    Expectation: Output is equal to the expected output
    """
    input_data = [np.array([1]),
                  np.array(0),
                  np.array(0),
                  np.array(0),
                  np.array([9, 10, 11, 12, 13]),
                  np.array([2]),
                  np.array([3, 4, 5]),
                  np.array([6, 7, 8])]

    class RandomAccessDataset:
        def __init__(self, input_data):
            self._data = input_data
            self._label = np.zeros((len(self._data), 1))
        def __getitem__(self, index):
            return self._data[index], self._label[index]
        def __len__(self):
            return len(self._data)

    dataset = ds.GeneratorDataset(RandomAccessDataset(input_data), column_names=["data", "label"])

    column_names = ["data"]
    bucket_boundaries = [1, 2, 4]
    bucket_batch_sizes = [3, 2, 4, 1]

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes)

    expected_output = [np.sort(np.array([[1], [2]])),
                       np.sort(np.array([[6, 7, 8], [3, 4, 5]]), axis=0),
                       np.sort(np.array([[9, 10, 11, 12, 13]]), axis=0),
                       np.array([0, 0, 0])]

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        exist = False
        for item in expected_output:
            print(data['data'], item, data['data'].shape, item.shape)
            if data['data'].shape == item.shape and (np.sort(data['data'], axis=0) == item).all():
                exist = True
                break
        assert exist


def test_bucket_batch_invalid_column():
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op with invalid column
    Expectation: Error is raised as expected
    """
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["invalid_column"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [3, 3, 2, 2]
    element_length_function = (lambda x: x[0] % 4)

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function)

    with pytest.raises(RuntimeError) as info:
        num_rows = 0
        for _ in dataset.create_dict_iterator(num_epochs=1):
            num_rows += 1

        assert "BucketBatchByLength: Couldn't find the specified column in the dataset" in str(info.value)


@pytest.mark.parametrize("drop_remainder", (True, False))
def test_bucket_batch_with_pull_mode(drop_remainder):
    """
    Feature: bucket_batch_by_length op
    Description: Test bucket_batch_by_length op to support pull mode
    Expectation: Output is equal to the expected output
    """
    original_debug_mode = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)
    dataset = ds.GeneratorDataset((lambda: generate_sequential_same_shape(10)), ["col1"])

    column_names = ["col1"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [3, 3, 3, 3]
    element_length_function = (lambda x: x[0] % 4)

    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
                                             bucket_batch_sizes, element_length_function,
                                             drop_remainder=drop_remainder)

    expected_output = [[[0], [4], [8]],
                       [[1], [5], [9]]]
    if not drop_remainder:
        # append the remainder
        expected_output += [[[2], [6]], [[3], [7]]]

    # iterate at least 2 epochs to check if the remainder is dropped
    num_epochs = 2
    iterator = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    for _ in range(num_epochs):
        output = []
        for data in iterator:
            output.append(data["col1"].tolist())
        assert output == expected_output

    ds.config.set_debug_mode(original_debug_mode)


if __name__ == '__main__':
    test_bucket_batch_invalid_input()
    test_bucket_batch_multi_bucket_no_padding()
    test_bucket_batch_multi_bucket_no_padding_repeat()
    test_bucket_batch_multi_bucket_with_padding()
    test_bucket_batch_single_bucket_no_padding()
    test_bucket_batch_single_bucket_with_padding()
    test_bucket_batch_pad_to_bucket_boundary()
    test_bucket_batch_default_pad()
    test_bucket_batch_drop_remainder(drop_remainder=True)
    test_bucket_batch_default_length_function()
    test_bucket_batch_multi_column()
    test_bucket_batch_three_columns()
    test_bucket_batch_get_dataset_size()
    test_bucket_batch_invalid_column()
    test_bucket_batch_with_pull_mode(drop_remainder=True)
    test_bucket_batch_with_string()
    test_bucket_batch_with_scalar()
