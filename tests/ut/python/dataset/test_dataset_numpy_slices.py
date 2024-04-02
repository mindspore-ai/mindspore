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
import sys

import numpy as np
import pandas as pd
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger


def test_numpy_slices_list_1():
    """
    Feature: NumpySlicesDataset
    Description: Check if 1D list is sliced successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Slicing a 1D list.")

    np_data = [1, 2, 3]
    dataset = ds.NumpySlicesDataset(np_data, shuffle=False)

    for i, data in enumerate(dataset):
        assert data[0].asnumpy() == np_data[i]


def test_numpy_slices_list_2():
    """
    Feature: NumpySlicesDataset
    Description: Check if 2D list is sliced into 1D list successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Slicing a 2D list into 1D list.")

    np_data = [[1, 2], [3, 4]]
    dataset = ds.NumpySlicesDataset(np_data, column_names=["col1"], shuffle=False)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), np_data[i]).all()


def test_numpy_slices_list_3():
    """
    Feature: NumpySlicesDataset
    Description: Check if first dimension of 2D list is sliced successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Slicing list in the first dimension.")

    np_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    dataset = ds.NumpySlicesDataset(np_data, column_names=["col1"], shuffle=False)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), np_data[i]).all()


def test_numpy_slices_numpy():
    """
    Feature: NumpySlicesDataset
    Description: Check NumPy structure data after slicing 2D list
    Expectation: The dataset is processed as expected
    """
    logger.info("Test NumPy structure data.")

    np_data = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
    dataset = ds.NumpySlicesDataset(np_data, column_names=["col1"], shuffle=False)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), np_data[i]).all()


def test_numpy_slices_list_append():
    """
    Feature: NumpySlicesDataset
    Description: Check reading data of image list
    Expectation: The dataset is processed as expected
    """
    logger.info("Test reading data of image list.")

    DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    resize_height, resize_width = 2, 2

    data1 = ds.TFRecordDataset(DATA_DIR)
    resize_op = vision.Resize((resize_height, resize_width))
    data1 = data1.map(
        operations=[vision.Decode(), resize_op], input_columns=["image"])

    res = []
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(data["image"])

    dataset = ds.NumpySlicesDataset(res, column_names=["col1"], shuffle=False)

    for i, data in enumerate(dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert np.equal(data, res[i]).all()


def test_numpy_slices_dict_1():
    """
    Feature: NumpySlicesDataset
    Description: Check if dictionary dataset is sliced successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Dictionary structure data.")

    np_data = {"a": [1, 2], "b": [3, 4]}
    dataset = ds.NumpySlicesDataset(np_data, shuffle=False)
    res = [[1, 3], [2, 4]]

    for i, data in enumerate(dataset):
        assert data[0].asnumpy() == res[i][0]
        assert data[1].asnumpy() == res[i][1]


def test_numpy_slices_tuple_1():
    """
    Feature: NumpySlicesDataset
    Description: Check if tuple dataset is sliced successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test slicing a list of tuple.")

    np_data = [([1, 2], [3, 4]), ([11, 12], [13, 14]), ([21, 22], [23, 24])]
    dataset = ds.NumpySlicesDataset(np_data, shuffle=False)

    for i, data in enumerate(dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert np.equal(data, np_data[i]).all()

    assert sum([1 for _ in dataset]) == 3


def test_numpy_slices_tuple_2():
    """
    Feature: NumpySlicesDataset
    Description: Check if tuple dataset is sliced successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test slicing a tuple of list.")

    np_data = ([1, 2], [3, 4], [5, 6])
    expected = [[1, 3, 5], [2, 4, 6]]
    dataset = ds.NumpySlicesDataset(np_data, shuffle=False)

    for i, data in enumerate(dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert np.equal(data, expected[i]).all()

    assert sum([1 for _ in dataset]) == 2


def test_numpy_slices_tuple_3():
    """
    Feature: NumpySlicesDataset
    Description: Check if tuple dataset with different dimension is read successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test reading different dimension of tuple data.")
    features, labels = np.random.sample((5, 2)), np.random.sample((5, 1))
    data = (features, labels)

    dataset = ds.NumpySlicesDataset(
        data, column_names=["col1", "col2"], shuffle=False)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), features[i]).all()
        assert data[1].asnumpy() == labels[i]


def test_numpy_slices_csv_value():
    """
    Feature: NumpySlicesDataset
    Description: Check if csv dataset is loaded successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test loading value of csv file.")
    csv_file = "../data/dataset/testNumpySlicesDataset/heart.csv"

    df = pd.read_csv(csv_file)
    target = df.pop("target")
    df.pop("state")
    np_data = (df.values, target.values)

    dataset = ds.NumpySlicesDataset(np_data, column_names=[
        "col1", "col2"], shuffle=False)

    for i, data in enumerate(dataset):
        assert np.equal(np_data[0][i], data[0].asnumpy()).all()
        assert np.equal(np_data[1][i], data[1].asnumpy()).all()


def test_numpy_slices_csv_dict():
    """
    Feature: NumpySlicesDataset
    Description: Check if csv dataset is loaded as dictionary successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test loading csv file as dict.")

    csv_file = "../data/dataset/testNumpySlicesDataset/heart.csv"
    df = pd.read_csv(csv_file)
    df.pop("state")
    res = df.values

    dataset = ds.NumpySlicesDataset(dict(df), shuffle=False)

    for i, data in enumerate(dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert np.equal(data, res[i]).all()


def test_numpy_slices_num_samplers():
    """
    Feature: NumpySlicesDataset
    Description: Check if num_samplers argument works successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test num_samplers.")

    np_data = [[1, 2], [3, 4], [5, 6], [7, 8],
               [9, 10], [11, 12], [13, 14], [15, 16]]
    dataset = ds.NumpySlicesDataset(np_data, shuffle=False, num_samples=2)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), np_data[i]).all()

    assert sum([1 for _ in dataset]) == 2


def test_numpy_slices_distributed_sampler():
    """
    Feature: NumpySlicesDataset
    Description: Check if distributed sampler works successfully
    Expectation: The dataset is processed as expected
    """
    logger.info("Test distributed sampler.")

    np_data = [[1, 2], [3, 4], [5, 6], [7, 8],
               [9, 10], [11, 12], [13, 14], [15, 16]]
    dataset = ds.NumpySlicesDataset(
        np_data, shuffle=False, shard_id=0, num_shards=4)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), np_data[i * 4]).all()

    assert sum([1 for _ in dataset]) == 2


def test_numpy_slices_distributed_shard_limit():
    """
    Feature: NumpySlicesDataset
    Description: Check if slicing 1D list with shard beyond limit raises error
    Expectation: Error is raised as expected
    """
    logger.info("Test Slicing a 1D list.")

    np_data = [1, 2, 3]
    num = sys.maxsize
    with pytest.raises(ValueError) as err:
        ds.NumpySlicesDataset(np_data, num_shards=num,
                              shard_id=0, shuffle=False)
    assert "Input num_shards is not within the required interval of [1, 2147483647]." in str(
        err.value)


def test_numpy_slices_distributed_zero_shard():
    """
    Feature: NumpySlicesDataset
    Description: Check if slicing 1D list with 0 shard raises error
    Expectation: Error is raised as expected
    """
    logger.info("Test Slicing a 1D list.")

    np_data = [1, 2, 3]
    with pytest.raises(ValueError) as err:
        ds.NumpySlicesDataset(np_data, num_shards=0, shard_id=0, shuffle=False)
    assert "Input num_shards is not within the required interval of [1, 2147483647]." in str(
        err.value)


def test_numpy_slices_sequential_sampler():
    """
    Feature: NumpySlicesDataset
    Description: Check numpy_slices_dataset with SequentialSampler and repeat
    Expectation: The dataset is processed as expected
    """
    logger.info("Test numpy_slices_dataset with SequentialSampler and repeat.")

    np_data = [[1, 2], [3, 4], [5, 6], [7, 8],
               [9, 10], [11, 12], [13, 14], [15, 16]]
    dataset = ds.NumpySlicesDataset(
        np_data, sampler=ds.SequentialSampler()).repeat(2)

    for i, data in enumerate(dataset):
        assert np.equal(data[0].asnumpy(), np_data[i % 8]).all()


def test_numpy_slices_invalid_column_names_type():
    """
    Feature: NumpySlicesDataset
    Description: Check if incorrect column_names in input raises error
    Expectation: Error is raised as expected
    """
    logger.info("Test incorrect column_names input")
    np_data = [1, 2, 3]

    with pytest.raises(TypeError) as err:
        ds.NumpySlicesDataset(np_data, column_names=[1], shuffle=False)
    assert "Argument column_names[0] with value 1 is not of type [<class 'str'>]" in str(
        err.value)


def test_numpy_slices_invalid_column_names_string():
    """
    Feature: NumpySlicesDataset
    Description: Check if incorrect column_names with string in input raises error
    Expectation: Error is raised as expected
    """
    logger.info("Test incorrect column_names input")
    np_data = [1, 2, 3]

    with pytest.raises(ValueError) as err:
        ds.NumpySlicesDataset(np_data, column_names=[""], shuffle=False)
    assert "column_names[0] should not be empty" in str(err.value)


def test_numpy_slices_invalid_empty_column_names():
    """
    Feature: NumpySlicesDataset
    Description: Check if empty column_names in input raises error
    Expectation: Error is raised as expected
    """
    logger.info("Test incorrect column_names input")
    np_data = [1, 2, 3]

    with pytest.raises(ValueError) as err:
        ds.NumpySlicesDataset(np_data, column_names=[], shuffle=False)
    assert "column_names should not be empty" in str(err.value)


def test_numpy_slices_invalid_empty_data_column():
    """
    Feature: NumpySlicesDataset
    Description: Check if no column_names in input raises error
    Expectation: Error is raised as expected
    """
    logger.info("Test incorrect column_names input")
    np_data = []

    with pytest.raises(ValueError) as err:
        ds.NumpySlicesDataset(np_data, shuffle=False)
    assert "Argument data cannot be empty" in str(err.value)


def test_numpy_slice_empty_output_shape():
    """
    Feature: NumpySlicesDataset
    Description: Check if NumpySlicesDataset produces an empty output_shapes
    Expectation: The dataset is processed as expected
    """
    logger.info("running test_numpy_slice_empty_output_shape")
    dataset = ds.NumpySlicesDataset([[[1, 2], [3, 4]]], column_names=["col1"])
    dataset = dataset.batch(batch_size=3, drop_remainder=True)
    assert dataset.output_shapes() == []


def test_numpy_slice_with_diff_shape():
    """
    Feature: NumpySlicesDataset
    Description: Check if NumpySlicesDataset produces diff shape data
    Expectation: The dataset is processed success
    """
    data1 = np.array([1], dtype=np.uint8)
    data2 = np.array([5, 6], dtype=np.uint8)
    data3 = np.array([9, 10, 11], dtype=np.uint8)
    data4 = np.array([13, 14, 15, 16], dtype=np.uint8)

    data = [data1, data2, data3, data4]

    label = [1, 2, 3, 4]

    dataset = ds.NumpySlicesDataset((data, label), ["data", "label"], num_shards=4, shard_id=2, shuffle=False)

    for item in dataset.create_dict_iterator(output_numpy=True):
        assert (item["data"] == data3).all()


def test_numpy_slice_with_diff_shape_dict():
    """
    Feature: NumpySlicesDataset
    Description: Check if NumpySlicesDataset produces diff shape data by dict
    Expectation: The dataset is processed success
    """
    data1 = np.array([1], dtype=np.uint8)
    data2 = np.array([5, 6], dtype=np.uint8)
    data3 = np.array([9, 10, 11], dtype=np.uint8)
    data4 = np.array([13, 14, 15, 16], dtype=np.uint8)

    data = [data1, data2, data3, data4]

    label = [1, 2, 3, 4]

    dict_data = {"data": data, "label": label}

    dataset = ds.NumpySlicesDataset(dict_data, num_shards=4, shard_id=2, shuffle=False)

    for item in dataset.create_dict_iterator(output_numpy=True):
        assert (item["data"] == data3).all()


def test_numpy_slices_dynamic_list():
    """
    Feature: NumpySlicesDataset
    Description: Test a list of samples with different shapes
    Expectation: The data is processed as expected
    """

    data = [[], [1], [2, 3], [4, 5, 6], [7, 8, 9, 10]]
    dataset = ds.NumpySlicesDataset(data, column_names=["data"], shuffle=False)

    for i, sample in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(sample["data"], np.array(data[i]))


if __name__ == "__main__":
    test_numpy_slices_list_1()
    test_numpy_slices_list_2()
    test_numpy_slices_list_3()
    test_numpy_slices_list_append()
    test_numpy_slices_dict_1()
    test_numpy_slices_tuple_1()
    test_numpy_slices_tuple_2()
    test_numpy_slices_tuple_3()
    test_numpy_slices_csv_value()
    test_numpy_slices_csv_dict()
    test_numpy_slices_num_samplers()
    test_numpy_slices_distributed_sampler()
    test_numpy_slices_distributed_shard_limit()
    test_numpy_slices_distributed_zero_shard()
    test_numpy_slices_sequential_sampler()
    test_numpy_slices_invalid_column_names_type()
    test_numpy_slices_invalid_column_names_string()
    test_numpy_slices_invalid_empty_column_names()
    test_numpy_slices_invalid_empty_data_column()
    test_numpy_slice_empty_output_shape()
    test_numpy_slice_with_diff_shape()
    test_numpy_slice_with_diff_shape_dict()
