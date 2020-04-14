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
import mindspore.dataset.transforms.vision.c_transforms as vision
from util import save_and_check

import mindspore.dataset as ds
import numpy as np
from mindspore import log as logger

DATA_DIR_TF = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR_TF = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
COLUMNS_TF = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
              "col_sint16", "col_sint32", "col_sint64"]
GENERATE_GOLDEN = False

# Data for CIFAR and MNIST are not part of build tree
# They need to be downloaded directly
# prep_data.py can be exuted or code below
# import sys
# sys.path.insert(0,"../../data")
# import prep_data
# prep_data.download_all_for_test("../../data")
IMG_DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
IMG_SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

DATA_DIR_TF2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR_TF2 = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_tf_repeat_01():
    """
    a simple repeat operation.
    """
    logger.info("Test Simple Repeat")
    # define parameters
    repeat_count = 2
    parameters = {"params": {'repeat_count': repeat_count}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.repeat(repeat_count)

    filename = "repeat_result.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_tf_repeat_02():
    """
    a simple repeat operation to tes infinite
    """
    logger.info("Test Infinite Repeat")
    # define parameters
    repeat_count = -1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.repeat(repeat_count)

    itr = 0
    for _ in data1:
        itr = itr + 1
        if itr == 100:
            break
    assert itr == 100


def test_tf_repeat_03():
    '''repeat and batch '''
    data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)

    batch_size = 32
    resize_height, resize_width = 32, 32
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), interpolation=ds.transforms.vision.Inter.LINEAR)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=resize_op)
    data1 = data1.repeat(22)
    data1 = data1.batch(batch_size, drop_remainder=True)

    num_iter = 0
    for item in data1.create_dict_iterator():
        num_iter += 1
    logger.info("Number of tf data in data1: {}".format(num_iter))
    assert num_iter == 2


def generator():
    for i in range(3):
        yield np.array([i]),


def test_nested_repeat1():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(3)

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3 * 3


def test_nested_repeat2():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(1)
    data = data.repeat(1)

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 3


def test_nested_repeat3():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(1)
    data = data.repeat(2)

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3


def test_nested_repeat4():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(1)

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3


def test_nested_repeat5():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.batch(3)
    data = data.repeat(2)
    data = data.repeat(3)

    for i, d in enumerate(data):
        assert np.array_equal(d[0], np.asarray([[0], [1], [2]]))

    assert sum([1 for _ in data]) == 6


def test_nested_repeat6():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.batch(3)
    data = data.repeat(3)

    for i, d in enumerate(data):
        assert np.array_equal(d[0], np.asarray([[0], [1], [2]]))

    assert sum([1 for _ in data]) == 6


def test_nested_repeat7():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(3)
    data = data.batch(3)

    for i, d in enumerate(data):
        assert np.array_equal(d[0], np.asarray([[0], [1], [2]]))

    assert sum([1 for _ in data]) == 6


def test_nested_repeat8():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.batch(2, drop_remainder=False)
    data = data.repeat(2)
    data = data.repeat(3)

    for i, d in enumerate(data):
        if i % 2 == 0:
            assert np.array_equal(d[0], np.asarray([[0], [1]]))
        else:
            assert np.array_equal(d[0], np.asarray([[2]]))

    assert sum([1 for _ in data]) == 6 * 2


def test_nested_repeat9():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat()
    data = data.repeat(3)

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]
        if i == 10:
            break


def test_nested_repeat10():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(3)
    data = data.repeat()

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]
        if i == 10:
            break


def test_nested_repeat11():
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(3)
    data = data.repeat(4)
    data = data.repeat(5)

    for i, d in enumerate(data):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3 * 4 * 5 * 3


if __name__ == "__main__":
    logger.info("--------test tf repeat 01---------")
    # test_repeat_01()

    logger.info("--------test tf repeat 02---------")
    # test_repeat_02()

    logger.info("--------test tf repeat 03---------")
    test_tf_repeat_03()
