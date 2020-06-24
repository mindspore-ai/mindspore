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
import numpy as np
import mindspore.dataset as de
from mindspore import log as logger
import mindspore.dataset.transforms.vision.c_transforms as vision
import pandas as pd


def test_numpy_slices_list_1():
    logger.info("Test Slicing a 1D list.")

    np_data = [1, 2, 3]
    ds = de.NumpySlicesDataset(np_data, shuffle=False)

    for i, data in enumerate(ds):
        assert data[0] == np_data[i]


def test_numpy_slices_list_2():
    logger.info("Test Slicing a 2D list into 1D list.")

    np_data = [[1, 2], [3, 4]]
    ds = de.NumpySlicesDataset(np_data, column_names=["col1"], shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data[0], np_data[i]).all()


def test_numpy_slices_list_3():
    logger.info("Test Slicing list in the first dimension.")

    np_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    ds = de.NumpySlicesDataset(np_data, column_names=["col1"], shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data[0], np_data[i]).all()


def test_numpy_slices_list_append():
    logger.info("Test reading data of image list.")

    DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    resize_height, resize_width = 2, 2

    data1 = de.TFRecordDataset(DATA_DIR)
    resize_op = vision.Resize((resize_height, resize_width))
    data1 = data1.map(input_columns=["image"], operations=[vision.Decode(True), resize_op])

    res = []
    for data in data1.create_dict_iterator():
        res.append(data["image"])

    ds = de.NumpySlicesDataset(res, column_names=["col1"], shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data, res[i]).all()


def test_numpy_slices_dict_1():
    logger.info("Test Dictionary structure data.")

    np_data = {"a": [1, 2], "b": [3, 4]}
    ds = de.NumpySlicesDataset(np_data, shuffle=False)
    res = [[1, 3], [2, 4]]

    for i, data in enumerate(ds):
        assert data[0] == res[i][0]
        assert data[1] == res[i][1]


def test_numpy_slices_tuple_1():
    logger.info("Test slicing a list of tuple.")

    np_data = [([1, 2], [3, 4]), ([11, 12], [13, 14]), ([21, 22], [23, 24])]
    ds = de.NumpySlicesDataset(np_data, shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data, np_data[i]).all()

    assert sum([1 for _ in ds]) == 3


def test_numpy_slices_tuple_2():
    logger.info("Test slicing a tuple of list.")

    np_data = ([1, 2], [3, 4], [5, 6])
    expected = [[1, 3, 5], [2, 4, 6]]
    ds = de.NumpySlicesDataset(np_data, shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data, expected[i]).all()

    assert sum([1 for _ in ds]) == 2


def test_numpy_slices_tuple_3():
    logger.info("Test reading different dimension of tuple data.")
    features, labels = np.random.sample((5, 2)), np.random.sample((5, 1))
    data = (features, labels)

    ds = de.NumpySlicesDataset(data, column_names=["col1", "col2"], shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data[0], features[i]).all()
        assert data[1] == labels[i]


def test_numpy_slices_csv_value():
    logger.info("Test loading value of csv file.")
    csv_file = "../data/dataset/testNumpySlicesDataset/heart.csv"

    df = pd.read_csv(csv_file)
    target = df.pop("target")
    df.pop("state")
    np_data = (df.values, target.values)

    ds = de.NumpySlicesDataset(np_data, column_names=["col1", "col2"], shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(np_data[0][i], data[0]).all()
        assert np.equal(np_data[1][i], data[1]).all()


def test_numpy_slices_csv_dict():
    logger.info("Test loading csv file as dict.")

    csv_file = "../data/dataset/testNumpySlicesDataset/heart.csv"
    df = pd.read_csv(csv_file)
    df.pop("state")
    res = df.values

    ds = de.NumpySlicesDataset(dict(df), shuffle=False)

    for i, data in enumerate(ds):
        assert np.equal(data, res[i]).all()


def test_numpy_slices_num_samplers():
    logger.info("Test num_samplers.")

    np_data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    ds = de.NumpySlicesDataset(np_data, shuffle=False, num_samples=2)

    for i, data in enumerate(ds):
        assert np.equal(data[0], np_data[i]).all()

    assert sum([1 for _ in ds]) == 2


def test_numpy_slices_distributed_sampler():
    logger.info("Test distributed sampler.")

    np_data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    ds = de.NumpySlicesDataset(np_data, shuffle=False, shard_id=0, num_shards=4)

    for i, data in enumerate(ds):
        assert np.equal(data[0], np_data[i * 4]).all()

    assert sum([1 for _ in ds]) == 2


def test_numpy_slices_sequential_sampler():

    logger.info("Test numpy_slices_dataset with SequentialSampler and repeat.")

    np_data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    ds = de.NumpySlicesDataset(np_data, sampler=de.SequentialSampler()).repeat(2)

    for i, data in enumerate(ds):
        assert np.equal(data[0], np_data[i % 8]).all()


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
    test_numpy_slices_sequential_sampler()
