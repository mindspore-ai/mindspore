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
"""
Testing the OneHot Op
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as data_trans
import mindspore.dataset.transforms.py_transforms as py_trans
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from ..dataset.util import dataset_equal_with_function

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def one_hot(index, depth):
    """
    Apply the one_hot
    """
    arr = np.zeros([1, depth], dtype=np.int32)
    arr[0, index] = 1
    return arr


def test_one_hot():
    """
    Feature: OneHot Op
    Description: Test C++ op with One Hot Encoding
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("test_one_hot")
    depth = 10

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    one_hot_op = data_trans.OneHot(num_classes=depth)
    data1 = data1.map(operations=one_hot_op, input_columns=["label"])
    data1 = data1.project(["label"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["label"], shuffle=False)

    assert dataset_equal_with_function(data1, data2, 0, one_hot, depth)


def test_one_hot_post_aug():
    """
    Feature: OneHot Op
    Description: Test C++ op with One Hot Encoding after Multiple Data Augmentation Operations
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("test_one_hot_post_aug")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    # Define data augmentation parameters
    rescale = 1.0 / 255.0
    shift = 0.0
    resize_height, resize_width = 224, 224

    # Define map operations
    decode_op = c_vision.Decode()
    rescale_op = c_vision.Rescale(rescale, shift)
    resize_op = c_vision.Resize((resize_height, resize_width))

    # Apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=rescale_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])

    # Apply one-hot encoding on labels
    depth = 4
    one_hot_encode = data_trans.OneHot(depth)
    data1 = data1.map(operations=one_hot_encode, input_columns=["label"])

    # Apply datasets ops
    buffer_size = 100
    seed = 10
    batch_size = 2
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)
    data1 = data1.batch(batch_size, drop_remainder=True)

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        logger.info("image is: {}".format(item["image"]))
        logger.info("label is: {}".format(item["label"]))
        num_iter += 1

    assert num_iter == 1


def test_one_hot_success():
    """
    Feature: OneHot Op
    Description: Test Python op, with generated label using np.array(index)
    Expectation: Dataset pipeline runs successfully and results are verified
    """

    class GetDatasetGenerator:
        def __init__(self):
            np.random.seed(58)
            self.__data = np.random.sample((5, 2))
            self.__label = []
            for index in range(5):
                self.__label.append(np.array(index))

        def __getitem__(self, index):
            return (self.__data[index], self.__label[index])

        def __len__(self):
            return len(self.__data)

    dataset = ds.GeneratorDataset(GetDatasetGenerator(), ["data", "label"], shuffle=False)

    one_hot_encode = py_trans.OneHotOp(10)
    trans = py_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert item["label"][index] == 1.0


def test_one_hot_success2():
    """
    Feature: OneHot Op
    Description: Test Python op, with generated label using np.array([index])
    Expectation: Dataset pipeline runs successfully and results are verified
    """

    class GetDatasetGenerator:
        def __init__(self):
            np.random.seed(58)
            self.__data = np.random.sample((5, 2))
            self.__label = []
            for index in range(5):
                self.__label.append(np.array([index]))

        def __getitem__(self, index):
            return (self.__data[index], self.__label[index])

        def __len__(self):
            return len(self.__data)

    dataset = ds.GeneratorDataset(GetDatasetGenerator(), ["data", "label"], shuffle=False)

    one_hot_encode = py_trans.OneHotOp(10)
    trans = py_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        logger.info(item)
        assert item["label"][0][index] == 1.0


def test_one_hot_success3():
    """
    Feature: OneHot Op
    Description: Test Python op, with multi-dimension generated label
    Expectation: Dataset pipeline runs successfully and results are verified
    """

    class GetDatasetGenerator:
        def __init__(self):
            np.random.seed(58)
            self.__data = np.random.sample((5, 2))
            self.__label = []
            for _ in range(5):
                value = np.ones([10, 1], dtype=np.int32)
                for i in range(10):
                    value[i][0] = i
                self.__label.append(value)

        def __getitem__(self, index):
            return (self.__data[index], self.__label[index])

        def __len__(self):
            return len(self.__data)

    dataset = ds.GeneratorDataset(GetDatasetGenerator(), ["data", "label"], shuffle=False)

    one_hot_encode = py_trans.OneHotOp(10)
    trans = py_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(item)
        for i in range(10):
            assert item["label"][i][0][i] == 1.0


def test_one_hot_type_error():
    """
    Feature: OneHot Op
    Description: Test Python op with invalid float input type
    Expectation: Invalid input is detected
    """

    class GetDatasetGenerator:
        def __init__(self):
            np.random.seed(58)
            self.__data = np.random.sample((5, 2))
            self.__label = []
            for index in range(5):
                self.__label.append(np.array(float(index)))

        def __getitem__(self, index):
            return (self.__data[index], self.__label[index])

        def __len__(self):
            return len(self.__data)

    dataset = ds.GeneratorDataset(GetDatasetGenerator(), ["data", "label"], shuffle=False)

    one_hot_encode = py_trans.OneHotOp(10)
    trans = py_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    try:
        for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
            assert item["label"][index] == 1.0
    except RuntimeError as e:
        assert "the input numpy type should be int" in str(e)


if __name__ == "__main__":
    test_one_hot()
    test_one_hot_post_aug()
    test_one_hot_success()
    test_one_hot_success2()
    test_one_hot_success3()
    test_one_hot_type_error()
