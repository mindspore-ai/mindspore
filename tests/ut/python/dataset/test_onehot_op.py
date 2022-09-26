# Copyright 2019 Huawei Technologies Co., Ltd
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
Testing OneHot Op in Dataset
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import dataset_equal_with_function

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
DATA_DIR_IMAGENET = "../data/dataset/testImageNetData/train"


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
    Description: Test C++ op with One Hot Encoding after Multiple Data Augmentation Operators
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("test_one_hot_post_aug")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    # Define data augmentation parameters
    rescale = 1.0 / 255.0
    shift = 0.0
    resize_height, resize_width = 224, 224

    # Define map operations
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(rescale, shift)
    resize_op = vision.Resize((resize_height, resize_width))

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

    one_hot_encode = data_trans.OneHot(10)
    trans = data_trans.Compose([one_hot_encode])
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

    one_hot_encode = data_trans.OneHot(10)
    trans = data_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        logger.info(item)
        assert item["label"][index] == 1.0


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

    one_hot_encode = data_trans.OneHot(10)
    trans = data_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(item)
        for i in range(10):
            assert item["label"][i][i] == 1.0


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

    one_hot_encode = data_trans.OneHot(10)
    trans = data_trans.Compose([one_hot_encode])
    dataset = dataset.map(operations=trans, input_columns=["label"])

    try:
        for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
            assert item["label"][index] == 1.0
    except RuntimeError as e:
        assert "OneHot only support input of int type, but got:float64" in str(e)


def test_one_hot_smoothing_rate():
    """
    Feature: OneHot op
    Description: Test smoothing_rate parameter
    Expectation: The dataset is processed as expected
    """
    logger.info("Test one hot encoding op")

    # define map operations
    dataset = ds.ImageFolderDataset(DATA_DIR_IMAGENET, num_samples=20)
    num_classes = 2
    epsilon_para = 0.1

    op = data_trans.OneHot(num_classes=num_classes, smoothing_rate=epsilon_para)
    dataset = dataset.map(operations=op, input_columns=["label"])

    golden_label = np.ones(num_classes) * epsilon_para / num_classes
    golden_label[1] = 1 - epsilon_para / num_classes

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        label = data["label"]
        logger.info("label is {}".format(label))
        logger.info("golden_label is {}".format(golden_label))
        assert label.all() == golden_label.all()


def test_one_hot_smoothing_rate_error_input():
    """
    Feature: OneHot op
    Description: Test smoothing_rate with invalid input
    Expectation: Error is raised as expected
    """

    def test_config(my_smoothing_rate):
        with pytest.raises(ValueError) as info:
            data1 = ds.ImageFolderDataset(DATA_DIR_IMAGENET, num_samples=20)
            op = data_trans.OneHot(num_classes=10, smoothing_rate=my_smoothing_rate)
            data1 = data1.map(operations=op, input_columns=["label"])
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                pass
        error_msg = "Input smoothing_rate is not within the required interval of [0.0, 1.0]."
        assert error_msg in str(info.value)

    # Test out-of-bound values for OneHot's smoothing_rate parameter
    test_config(-0.1)
    test_config(1.1)


if __name__ == "__main__":
    test_one_hot()
    test_one_hot_post_aug()
    test_one_hot_success()
    test_one_hot_success2()
    test_one_hot_success3()
    test_one_hot_type_error()
    test_one_hot_smoothing_rate()
    test_one_hot_smoothing_rate_error_input()
