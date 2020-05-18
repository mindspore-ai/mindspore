# Copyright 2019 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""test mnist to mindrecord tool"""
import cv2
import gzip
import numpy as np
import os

from mindspore import log as logger
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import MnistToMR

MNIST_DIR = "../data/mindrecord/testMnistData"
FILE_NAME = "mnist"
PARTITION_NUM = 4
IMAGE_SIZE = 28
NUM_CHANNELS = 1


def read(train_name, test_name):
    """test file reader"""
    count = 0
    reader = FileReader(train_name)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 2
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 20
    reader.close()

    count = 0
    reader = FileReader(test_name)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 2
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 10
    reader.close()


def test_mnist_to_mindrecord():
    """test transform mnist dataset to mindrecord."""
    mnist_transformer = MnistToMR(MNIST_DIR, FILE_NAME)
    mnist_transformer.transform()
    assert os.path.exists("mnist_train.mindrecord")
    assert os.path.exists("mnist_test.mindrecord")

    read("mnist_train.mindrecord", "mnist_test.mindrecord")

    os.remove("{}".format("mnist_train.mindrecord"))
    os.remove("{}.db".format("mnist_train.mindrecord"))
    os.remove("{}".format("mnist_test.mindrecord"))
    os.remove("{}.db".format("mnist_test.mindrecord"))


def test_mnist_to_mindrecord_compare_data():
    """test transform mnist dataset to mindrecord and compare data."""
    mnist_transformer = MnistToMR(MNIST_DIR, FILE_NAME)
    mnist_transformer.transform()
    assert os.path.exists("mnist_train.mindrecord")
    assert os.path.exists("mnist_test.mindrecord")

    train_name, test_name = "mnist_train.mindrecord", "mnist_test.mindrecord"

    def _extract_images(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels]."""
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(
                IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(
                num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            return data

    def _extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            return labels

    train_data_filename_ = os.path.join(MNIST_DIR,
                                        'train-images-idx3-ubyte.gz')
    train_labels_filename_ = os.path.join(MNIST_DIR,
                                          'train-labels-idx1-ubyte.gz')
    test_data_filename_ = os.path.join(MNIST_DIR,
                                       't10k-images-idx3-ubyte.gz')
    test_labels_filename_ = os.path.join(MNIST_DIR,
                                         't10k-labels-idx1-ubyte.gz')
    train_data = _extract_images(train_data_filename_, 20)
    train_labels = _extract_labels(train_labels_filename_, 20)
    test_data = _extract_images(test_data_filename_, 10)
    test_labels = _extract_labels(test_labels_filename_, 10)

    reader = FileReader(train_name)
    for x, data, label in zip(reader.get_next(), train_data, train_labels):
        _, img = cv2.imencode(".jpeg", data)
        assert np.array(x['data']) == img.tobytes()
        assert np.array(x['label']) == label
    reader.close()

    reader = FileReader(test_name)
    for x, data, label in zip(reader.get_next(), test_data, test_labels):
        _, img = cv2.imencode(".jpeg", data)
        assert np.array(x['data']) == img.tobytes()
        assert np.array(x['label']) == label
    reader.close()

    os.remove("{}".format("mnist_train.mindrecord"))
    os.remove("{}.db".format("mnist_train.mindrecord"))
    os.remove("{}".format("mnist_test.mindrecord"))
    os.remove("{}.db".format("mnist_test.mindrecord"))


def test_mnist_to_mindrecord_multi_partition():
    """test transform mnist dataset to multiple mindrecord files."""
    mnist_transformer = MnistToMR(MNIST_DIR, FILE_NAME, PARTITION_NUM)
    mnist_transformer.transform()

    read("mnist_train.mindrecord0", "mnist_test.mindrecord0")

    for i in range(PARTITION_NUM):
        os.remove("{}".format("mnist_train.mindrecord" + str(i)))
        os.remove("{}.db".format("mnist_train.mindrecord" + str(i)))
        os.remove("{}".format("mnist_test.mindrecord" + str(i)))
        os.remove("{}.db".format("mnist_test.mindrecord" + str(i)))
