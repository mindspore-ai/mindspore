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
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as F
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from mindspore import log as logger
import numpy as np


# In generator dataset: Number of rows is 3, its value is 0, 1, 2
def generator():
    for i in range(3):
        yield np.array([i]),


# In generator_10 dataset: Number of rows is 7, its value is 3, 4, 5 ... 10
def generator_10():
    for i in range(3, 10):
        yield np.array([i]),

# In generator_20 dataset: Number of rows is 10, its value is 10, 11, 12 ... 20
def generator_20():
    for i in range(10, 20):
        yield np.array([i]),


def test_concat_01():
    """
    Test concat: test concat 2 datasets that have the same column name and data type
    """
    logger.info("test_concat_01")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1 + data2

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert i == d[0][0]

    assert sum([1 for _ in data3]) == 10


def test_concat_02():
    """
    Test concat: test concat 2 datasets using concat operation not "+" operation
    """
    logger.info("test_concat_02")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1.concat(data2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert i == d[0][0]

    assert sum([1 for _ in data3]) == 10


def test_concat_03():
    """
    Test concat: test concat dataset that has different column
    """
    logger.info("test_concat_03")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col2"])

    data3 = data1 + data2

    try:
        for i, d in enumerate(data3):
            pass
        assert False
    except RuntimeError:
        pass


def test_concat_04():
    """
    Test concat: test concat dataset that has different rank
    """
    logger.info("test_concat_04")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col2"])
    data2 = data2.batch(3)

    data3 = data1 + data2

    try:
        for i, d in enumerate(data3):
            pass
        assert False
    except RuntimeError:
        pass


def test_concat_05():
    """
    Test concat: test concat dataset that has different data type
    """
    logger.info("test_concat_05")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    type_cast_op = C.TypeCast(mstype.float32)
    data1 = data1.map(input_columns=["col1"], operations=type_cast_op)

    data3 = data1 + data2

    try:
        for i, d in enumerate(data3):
            pass
        assert False
    except RuntimeError:
        pass


def test_concat_06():
    """
    Test concat: test concat muti datasets in one time
    """
    logger.info("test_concat_06")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])
    data3 = ds.GeneratorDataset(generator_20, ["col1"])

    dataset = data1 + data2 + data3

    # Here i refers to index, d refers to data element
    for i, d in enumerate(dataset):
        logger.info("data: %i", d[0][0])
        assert i == d[0][0]

    assert sum([1 for _ in dataset]) == 20


def test_concat_07():
    """
    Test concat: test concat one dataset with multi datasets (datasets list)
    """
    logger.info("test_concat_07")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])
    data3 = ds.GeneratorDataset(generator_20, ["col1"])

    dataset = [data2] + [data3]
    data4 = data1 + dataset

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data4):
        logger.info("data: %i", d[0][0])
        assert i == d[0][0]

    assert sum([1 for _ in data4]) == 20


def test_concat_08():
    """
    Test concat: test concat 2 datasets, and then repeat
    """
    logger.info("test_concat_08")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1 + data2
    data3 = data3.repeat(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert i % 10 == d[0][0]

    assert sum([1 for _ in data3]) == 20


def test_concat_09():
    """
    Test concat: test concat 2 datasets, both of them have been repeat before
    """
    logger.info("test_concat_09")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data1 = data1.repeat(2)
    data2 = data2.repeat(2)
    data3 = data1 + data2

    res = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9]
    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert res[i] == d[0][0]

    assert sum([1 for _ in data3]) == 20


def test_concat_10():
    """
    Test concat: test concat 2 datasets, one of them have repeat before
    """
    logger.info("test_concat_10")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data1 = data1.repeat(2)
    data3 = data1 + data2

    res = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert res[i] == d[0][0]

    assert sum([1 for _ in data3]) == 13


def test_concat_11():
    """
    Test concat: test dataset batch then concat
    """
    logger.info("test_concat_11")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_20, ["col1"])

    data1 = data1.batch(3)
    data2 = data2.batch(5)

    data3 = data1 + data2
    res = [0, 10, 15, 20]

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert res[i] == d[0][0]

    assert sum([1 for _ in data3]) == 3


def test_concat_12():
    """
    Test concat: test dataset concat then shuffle
    """
    logger.info("test_concat_12")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data1.set_dataset_size(3)
    data2.set_dataset_size(7)

    data3 = data1 + data2
    res = [8, 6, 2, 5, 0, 4, 9, 3, 7, 1]

    ds.config.set_seed(1)
    assert data3.get_dataset_size() == 10
    data3 = data3.shuffle(buffer_size=10)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert res[i] == d[0][0]

    assert sum([1 for _ in data3]) == 10


def test_concat_13():
    """
    Test concat: test dataset batch then shuffle and concat
    """
    logger.info("test_concat_13")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_20, ["col1"])

    data1.set_dataset_size(3)
    data2.set_dataset_size(10)

    data1 = data1.batch(3)
    data2 = data2.batch(5)

    data3 = data1 + data2
    res = [15, 0, 10]

    ds.config.set_seed(1)
    assert data3.get_dataset_size() == 3

    data3 = data3.shuffle(buffer_size=int(data3.get_dataset_size()))

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3):
        logger.info("data: %i", d[0][0])
        assert res[i] == d[0][0]

    assert sum([1 for _ in data3]) == 3


def test_concat_14():
    """
    Test concat: create dataset with different dataset folder, and do diffrent operation then concat
    """
    logger.info("test_concat_14")
    DATA_DIR = "../data/dataset/testPK/data"
    DATA_DIR2 = "../data/dataset/testImageNetData/train/"

    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_samples=3)
    data2 = ds.ImageFolderDatasetV2(DATA_DIR2, num_samples=2)

    transforms1 = F.ComposeOp([F.Decode(),
                               F.Resize((224,224)),
                               F.ToTensor()])

    data1 = data1.map(input_columns=["image"], operations=transforms1())
    data2 = data2.map(input_columns=["image"], operations=transforms1())
    data3 = data1 + data2

    expected, output = [], []
    for d in data1:
        expected.append(d[0])
    for d in data2:
        expected.append(d[0])
    for d in data3:
        output.append(d[0])

    assert len(expected) == len(output)
    np.array_equal(np.array(output), np.array(expected))

    assert sum([1 for _ in data3]) == 5
    assert data3.get_dataset_size() == 5


def test_concat_15():
    """
    Test concat: create dataset with different format of dataset file, and then concat
    """
    logger.info("test_concat_15")
    DATA_DIR = "../data/dataset/testPK/data"
    DATA_DIR2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]

    data1 = ds.ImageFolderDatasetV2(DATA_DIR)
    data2 = ds.TFRecordDataset(DATA_DIR2, columns_list=["image"])

    data1 = data1.project(["image"])
    data3 = data1 + data2

    assert sum([1 for _ in data3]) == 47


if __name__ == "__main__":
    test_concat_01()
    test_concat_02()
    test_concat_03()
    test_concat_04()
    test_concat_05()
    test_concat_06()
    test_concat_07()
    test_concat_08()
    test_concat_09()
    test_concat_10()
    test_concat_11()
    test_concat_12()
    test_concat_13()
    test_concat_14()
    test_concat_15()