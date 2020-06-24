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
import os

import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger

# Data for CIFAR and MNIST are not part of build tree
# They need to be downloaded directly
# prep_data.py can be executed or code below
# import sys
# sys.path.insert(0,"../../data")
# import prep_data
# prep_data.download_all_for_test("../../data")
DATA_DIR_10 = "../data/dataset/testCifar10Data"
DATA_DIR_100 = "../data/dataset/testCifar100Data"


def load_cifar(path):
    raw = np.empty(0, dtype=np.uint8)
    for file_name in os.listdir(path):
        if file_name.endswith(".bin"):
            with open(os.path.join(path, file_name), mode='rb') as file:
                raw = np.append(raw, np.fromfile(file, dtype=np.uint8), axis=0)
    raw = raw.reshape(-1, 3073)
    labels = raw[:, 0]
    images = raw[:, 1:]
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    return images, labels


def test_case_dataset_cifar10():
    """
    dataset parameter
    """
    logger.info("Test dataset parameter")
    # apply dataset operations
    data1 = ds.Cifar10Dataset(DATA_DIR_10, 100)

    num_iter = 0
    for _ in data1.create_dict_iterator():
        # in this example, each dictionary has keys "image" and "label"
        num_iter += 1
    assert num_iter == 100


def test_case_dataset_cifar100():
    """
    dataset parameter
    """
    logger.info("Test dataset parameter")
    # apply dataset operations
    data1 = ds.Cifar100Dataset(DATA_DIR_100, 100)

    num_iter = 0
    for _ in data1.create_dict_iterator():
        # in this example, each dictionary has keys "image" and "label"
        num_iter += 1
    assert num_iter == 100


def test_reading_cifar10():
    """
    Validate CIFAR10 image readings
    """
    data1 = ds.Cifar10Dataset(DATA_DIR_10, 100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_10)
    for i, d in enumerate(data1.create_dict_iterator()):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["label"], labels[i])


if __name__ == '__main__':
    test_case_dataset_cifar10()
    test_case_dataset_cifar100()
    test_reading_cifar10()
