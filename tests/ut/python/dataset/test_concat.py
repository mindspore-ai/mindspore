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
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as F
from mindspore import log as logger
from util import config_get_set_seed


# In generator dataset: Number of rows is 3; its values are 0, 1, 2
def generator():
    for i in range(3):
        yield (np.array([i]),)


# In generator_10 dataset: Number of rows is 7; its values are 3, 4, 5 ... 9
def generator_10():
    for i in range(3, 10):
        yield (np.array([i]),)


# In generator_20 dataset: Number of rows is 10; its values are 10, 11, 12 ... 19
def generator_20():
    for i in range(10, 20):
        yield (np.array([i]),)


# In generator_29 dataset: Number of rows is 9; its values are 20, 21, 22 ... 28
def generator_29():
    for i in range(20, 29):
        yield (np.array([i]),)


def test_concat_01():
    """
    Feature: Concat op
    Description: Test Concat op with 2 datasets that have the same column name and data type
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_01")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1 + data2

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert i == t[0][0]

    assert sum([1 for _ in data3]) == 10


def test_concat_02():
    """
    Feature: Concat op
    Description: Test Concat op with 2 datasets using concat operation not "+" operation
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_02")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1.concat(data2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert i == t[0][0]

    assert sum([1 for _ in data3]) == 10


def test_concat_03():
    """
    Feature: Concat op
    Description: Test Concat op with dataset that has different column
    Expectation: Error is raised as expected
    """
    logger.info("test_concat_03")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col2"])

    data3 = data1 + data2

    try:
        for _, _ in enumerate(data3):
            pass
        assert False
    except RuntimeError:
        pass


def test_concat_04():
    """
    Feature: Concat op
    Description: Test Concat op with dataset that has different rank
    Expectation: Error is raised as expected
    """
    logger.info("test_concat_04")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col2"])
    data2 = data2.batch(3)

    data3 = data1 + data2

    try:
        for _, _ in enumerate(data3):
            pass
        assert False
    except RuntimeError:
        pass


def test_concat_05():
    """
    Feature: Concat op
    Description: Test Concat op with dataset that has different data type
    Expectation: Error is raised as expected
    """
    logger.info("test_concat_05")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    type_cast_op = C.TypeCast(mstype.float32)
    data1 = data1.map(operations=type_cast_op, input_columns=["col1"])

    data3 = data1 + data2

    try:
        for _, _ in enumerate(data3):
            pass
        assert False
    except RuntimeError:
        pass


def test_concat_06():
    """
    Feature: Concat op
    Description: Test Concat op with multiple datasets in one time
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_06")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])
    data3 = ds.GeneratorDataset(generator_20, ["col1"])

    dataset = data1 + data2 + data3

    # Here i refers to index, d refers to data element
    for i, d in enumerate(dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert i == t[0][0]

    assert sum([1 for _ in dataset]) == 20


def test_concat_07():
    """
    Feature: Concat op
    Description: Test Concat op one dataset with multiple datasets (datasets list)
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_07")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])
    data3 = ds.GeneratorDataset(generator_20, ["col1"])

    dataset = [data2] + [data3]
    data4 = data1 + dataset

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data4.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert i == t[0][0]

    assert sum([1 for _ in data4]) == 20


def test_concat_08():
    """
    Feature: Concat op
    Description: Test Concat op with 2 datasets and then repeat
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_08")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1 + data2
    data3 = data3.repeat(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert i % 10 == t[0][0]

    assert sum([1 for _ in data3]) == 20


def test_concat_09():
    """
    Feature: Concat op
    Description: Test Concat op with 2 datasets where both of them have been repeat before
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_09")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data1 = data1.repeat(2)
    data2 = data2.repeat(2)
    data3 = data1 + data2

    res = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9]
    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert res[i] == t[0][0]

    assert sum([1 for _ in data3]) == 20


def test_concat_10():
    """
    Feature: Concat op
    Description: Test Concat op with 2 datasets but one of them have repeat before
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_10")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data1 = data1.repeat(2)
    data3 = data1 + data2

    res = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert res[i] == t[0][0]

    assert sum([1 for _ in data3]) == 13


def test_concat_11():
    """
    Feature: Concat op
    Description: Test dataset batch then concat
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_11")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_20, ["col1"])

    data1 = data1.batch(3)
    data2 = data2.batch(5)

    data3 = data1 + data2
    res = [0, 10, 15, 20]

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert res[i] == t[0][0]

    assert sum([1 for _ in data3]) == 3


def test_concat_12():
    """
    Feature: Concat op
    Description: Test dataset concat then shuffle
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_12")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = data1 + data2
    res = [8, 6, 2, 5, 0, 4, 9, 3, 7, 1]

    original_seed = config_get_set_seed(1)
    assert data3.get_dataset_size() == 10
    data3 = data3.shuffle(buffer_size=10)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert res[i] == t[0][0]

    assert sum([1 for _ in data3]) == 10
    ds.config.set_seed(original_seed)


def test_concat_13():
    """
    Feature: Concat op
    Description: Test dataset batch then shuffle and concat
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_13")
    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_20, ["col1"])

    data1 = data1.batch(3)
    data2 = data2.batch(5)

    data3 = data1 + data2
    res = [15, 0, 10]

    original_seed = config_get_set_seed(1)
    assert data3.get_dataset_size() == 3

    data3 = data3.shuffle(buffer_size=int(data3.get_dataset_size()))

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        t = d
        logger.info("data: %i", t[0][0])
        assert res[i] == t[0][0]

    assert sum([1 for _ in data3]) == 3
    ds.config.set_seed(original_seed)


def test_concat_14():
    """
    Feature: Concat op
    Description: Test Concat op on two different source datasets with different dataset operations
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_14")
    data_dir = "../data/dataset/testPK/data"
    data_dir2 = "../data/dataset/testImageNetData/train/"

    data1 = ds.ImageFolderDataset(data_dir, num_samples=3)
    data2 = ds.ImageFolderDataset(data_dir2, num_samples=2)

    transforms1 = C.Compose([F.Decode(True),
                             F.Resize((224, 224)),
                             F.ToTensor()])

    data1 = data1.map(operations=transforms1, input_columns=["image"])
    data2 = data2.map(operations=transforms1, input_columns=["image"])
    data3 = data1 + data2

    expected, output = [], []
    for d in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        expected.append(d[0])
    for d in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
        expected.append(d[0])
    for d in data3.create_tuple_iterator(num_epochs=1, output_numpy=True):
        output.append(d[0])

    assert len(expected) == len(output)
    np.array_equal(np.array(output), np.array(expected))

    assert sum([1 for _ in data3]) == 5
    assert data3.get_dataset_size() == 5


def test_concat_15():
    """
    Feature: Concat op
    Description: Create dataset with different format of dataset file, and then concat
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_15")
    data_dir = "../data/dataset/testPK/data"
    data_dir2 = [
        "../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]

    data1 = ds.ImageFolderDataset(data_dir)
    data2 = ds.TFRecordDataset(data_dir2, columns_list=["image"])

    data1 = data1.project(["image"])
    data3 = data1 + data2

    assert sum([1 for _ in data3]) == 47


def test_concat_16():
    """
    Feature: Concat op
    Description: Test get_dataset_size on nested concats
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_16")
    data_dir = "../data/dataset/testPK/data"
    data_dir2 = [
        "../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]

    data1 = ds.ImageFolderDataset(data_dir)
    data2 = ds.TFRecordDataset(data_dir2, columns_list=["image"])

    data3 = ds.GeneratorDataset(generator, ["col1"])
    data4 = ds.GeneratorDataset(generator_10, ["col1"])

    data5 = data1 + data2
    data6 = data3 + data4
    data7 = data5 + data6

    original_seed = config_get_set_seed(1)

    # 57 is the total size of all 4 leaf datasets
    assert data7.get_dataset_size() == 57
    ds.config.set_seed(original_seed)


def test_concat_17():
    """
    Feature: Concat op
    Description: Test get_dataset_size on nested concats (with sampler)
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_17")

    data1 = ds.GeneratorDataset(generator, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])

    data3 = ds.GeneratorDataset(generator_20, ["col1"])
    data4 = ds.GeneratorDataset(generator_29, ["col1"])

    data5 = data1 + data2
    data6 = data3 + data4
    data7 = data5 + data6

    original_seed = config_get_set_seed(1)
    shard_num = 10
    counter = 0

    for i in range(shard_num):
        distributed_sampler = ds.DistributedSampler(
            num_shards=shard_num, shard_id=i, shuffle=False, num_samples=None)
        data7.use_sampler(distributed_sampler)
        iter_counter = 0
        for _ in data7.create_dict_iterator(num_epochs=1, output_numpy=True):
            counter += 1
            iter_counter += 1
        assert data7.get_dataset_size() == iter_counter

    # 29 is the total size of all 4 leaf datasets
    assert counter == 29
    ds.config.set_seed(original_seed)


def test_concat_18():
    """
    Feature: Concat op
    Description: Test random select of ConcatDataset with sampler
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_18")

    original_seed = config_get_set_seed(3)

    def define_generators():
        # 0, 1, 2 (3 samples)
        data1 = ds.GeneratorDataset(generator, ["col1"])
        # 3, 4, 5 ... 9 (7 samples)
        data2 = ds.GeneratorDataset(generator_10, ["col1"])
        # 10, 11, 12 ... 19 (10 samples)
        data3 = ds.GeneratorDataset(generator_20, ["col1"])
        # 20, 21, 22 ... 28 (9 samples)
        data4 = ds.GeneratorDataset(generator_29, ["col1"])
        return data1, data2, data3, data4

    # check first 3 samples comes from data1 and data2
    data1, data2, _, _ = define_generators()
    check_flag = [False, False]
    data12 = data1 + data2
    data12.use_sampler(ds.RandomSampler())
    for i, d in enumerate(data12.create_tuple_iterator(output_numpy=True)):
        if d[0] <= 2:
            check_flag[0] = True
        else:
            check_flag[1] = True
        if i > 2:
            break
    assert check_flag == [True, True]

    # check first 10 samples comes from data1 ~ data4
    data1, data2, data3, data4 = define_generators()
    check_flag = [False, False, False, False]
    data1234 = data1.concat([data2, data3, data4])
    data1234.use_sampler(ds.RandomSampler())
    for i, d in enumerate(data1234.create_tuple_iterator(output_numpy=True)):
        if d[0] <= 2:
            check_flag[0] = True
        elif d[0] <= 9:
            check_flag[1] = True
        elif d[0] <= 19:
            check_flag[2] = True
        elif d[0] <= 28:
            check_flag[3] = True
        if i > 9:
            break
    assert check_flag == [True, True, True, True]

    # check global shuffle on shared source
    class DS:
        def __init__(self, i, j):
            self.data = [i for i in range(i, j)]

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    data1 = ds.GeneratorDataset(DS(0, 20), "data1", shuffle=True, num_shards=2, shard_id=0)
    data2 = ds.GeneratorDataset(DS(20, 25), "data1", shuffle=True, num_shards=2, shard_id=1)
    data3 = data1 + data2
    data3.use_sampler(ds.RandomSampler())

    check_flag = [False, False]
    for i, d in enumerate(data1234.create_tuple_iterator(output_numpy=True)):
        if d[0] <= 2:
            check_flag[0] = True
        elif d[0] <= 19:
            check_flag[1] = True
        if i > 6:
            break
    assert check_flag == [True, True]

    ds.config.set_seed(original_seed)


def test_concat_19():
    """
    Feature: Concat op
    Description: Test random select of ConcatDataset in debug mode
    Expectation: Output passes the equality test
    """
    logger.info("test_concat_19")

    original_seed = config_get_set_seed(2)

    class DS:
        def __init__(self, i, j):
            self.data = [i for i in range(i, j)]

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    ds1 = ds.GeneratorDataset(DS(0, 20), "data1", shuffle=True)
    ds2 = ds.GeneratorDataset(DS(20, 25), "data1", shuffle=True)
    ds3 = ds1.concat([ds2])
    ds3.use_sampler(ds.RandomSampler())
    ds3 = ds3.map(lambda x: x + 1)

    # check data distribution in debug mode
    ds.config.set_debug_mode(True)
    less_than_20 = 0
    greater_euqal_to_20 = 0
    for i, data in enumerate(ds3.create_tuple_iterator(output_numpy=True, num_epochs=1)):
        if data[0] < 20:
            less_than_20 += 1
        else:
            greater_euqal_to_20 += 1
        if i >= 15:
            break
    result1 = [less_than_20, greater_euqal_to_20]

    # check data distribution in pipeline mode
    ds.config.set_debug_mode(False)
    less_than_20 = 0
    greater_euqal_to_20 = 0
    for i, data in enumerate(ds3.create_tuple_iterator(output_numpy=True, num_epochs=1)):
        if data[0] < 20:
            less_than_20 += 1
        else:
            greater_euqal_to_20 += 1
        if i >= 15:
            break
    result2 = [less_than_20, greater_euqal_to_20]

    assert result1 == result2
    ds.config.set_seed(original_seed)


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
    test_concat_16()
    test_concat_17()
    test_concat_18()
    test_concat_19()
