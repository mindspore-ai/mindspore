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
import copy
import time

import pytest
import numpy as np

import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.engine.iterators as it
from mindspore import log as logger
from mindspore import Tensor
import mindspore.ops as ops


# Generate 1d int numpy array from 0 - 63
def generator_1d():
    for i in range(64):
        yield (np.array([i]),)


class DatasetGenerator:
    def __init__(self):
        pass

    def __getitem__(self, item):
        return (np.array([item]),)

    def __len__(self):
        return 10


class DatasetGeneratorLarge:
    def __init__(self):
        self.data = np.array(range(4000))

    def __getitem__(self, item):
        return (self.data + item, self.data * 10)

    def __len__(self):
        return 10


class DatasetGeneratorMixed:
    def __init__(self):
        pass

    def __getitem__(self, item):
        flatten = ops.Flatten()
        x = Tensor(np.ones(shape=[2, 3]), mindspore.float32)
        output = flatten(x)
        return (output.asnumpy(),)

    def __len__(self):
        return 10


def test_generator_0():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


# Generate md int numpy array from [[0, 1], [2, 3]] to [[63, 64], [65, 66]]
def generator_md():
    for i in range(64):
        yield (np.array([[i, i + 1], [i + 2, i + 3]]),)


def test_generator_1():
    """
    Feature: GeneratorDataset
    Description: Test MD Generator with shape [2, 2]
    Expectation: The dataset is processed as expected
    """
    logger.info("Test MD Generator : 0 - 63, with shape [2, 2]")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_md, ["data"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


# Generate two columns, the first column is from Generator1D, the second column is from GeneratorMD
def generator_mc(maxid=64):
    for i in range(maxid):
        yield (np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]]))


def test_generator_2():
    """
    Feature: GeneratorDataset
    Description: Test multi column Generator
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc, ["col0", "col1"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["col0"], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_3():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with repeat(4)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 + Repeat(4)")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    data1 = data1.repeat(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1
        if i == 64:
            i = 0


def test_generator_4():
    """
    Feature: GeneratorDataset
    Description: Test fixed size 1D Generator with batch(4)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 + batch(4)")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4


def generator_with_type(t):
    for i in range(64):
        yield (np.array([i], dtype=t),)


def type_tester(t):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type(t)), ["data"])

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4


def test_generator_5():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator on different data types
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator on all data types")

    types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]

    for t in types:
        type_tester(t)


def type_tester_with_type_check(t, c):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type(t)), ["data"], column_types=[c])

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4


def test_generator_6():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator on different data types with type check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator on all data types with type check")

    np_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
                np.float64]
    de_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32,
                mstype.uint64, mstype.float32, mstype.float64]

    for i, _ in enumerate(np_types):
        type_tester_with_type_check(np_types[i], de_types[i])


def generator_with_type_2c(t):
    for i in range(64):
        yield (np.array([i], dtype=t), np.array([i], dtype=t))


def type_tester_with_type_check_2c(t, c):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type_2c(t)), ["data0", "data1"], column_types=c)

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data0"], golden)
        i = i + 4


def test_generator_7():
    """
    Feature: GeneratorDataset
    Description: Test 2 column Generator on different data type with type check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 2 column Generator on all data types with type check")

    np_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
                np.float64]
    de_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32,
                mstype.uint64, mstype.float32, mstype.float64]

    for i, _ in enumerate(np_types):
        type_tester_with_type_check_2c(np_types[i], [None, de_types[i]])


def test_generator_8():
    """
    Feature: GeneratorDataset
    Description: Test multi column Generator with few mapops
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator with mapops to check the order too")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: x * 3), input_columns="col0", output_columns="out0",
                      num_parallel_workers=2)
    data1 = data1.map(operations=(lambda x: (x * 7, x)), input_columns="col1", output_columns=["out1", "out2"],
                      num_parallel_workers=2, column_order=["out0", "out1", "out2"])
    data1 = data1.map(operations=(lambda x: x + 1), input_columns="out2", output_columns="out2",
                      num_parallel_workers=2)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i * 3])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 7, (i + 1) * 7], [(i + 2) * 7, (i + 3) * 7]])
        np.testing.assert_array_equal(item["out1"], golden)
        golden = np.array([[i + 1, i + 2], [i + 3, i + 4]])
        np.testing.assert_array_equal(item["out2"], golden)
        i = i + 1


def test_generator_9():
    """
    Feature: GeneratorDataset
    Description: Test map column order when len(input_columns) == len(output_columns)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when len(input_columns) == len(output_columns).")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["image", "label"])
    data2 = ds.GeneratorDataset(generator_mc(2048), ["label", "image"])
    data1 = data1.map(operations=(lambda x: x * 3), input_columns="label",
                      num_parallel_workers=4)
    data2 = data2.map(operations=(lambda x: x * 3), input_columns="label",
                      num_parallel_workers=4)

    # Expected column order is not changed.
    # data1 = data[0] is "image" and data[1] is "label"
    # data2 = data[0] is "label" and data[1] is "image"
    i = 0
    for data1, data2 in zip(data1, data2):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data1[0].asnumpy(), golden)
        golden = np.array([[i * 3, (i + 1) * 3], [(i + 2) * 3, (i + 3) * 3]])
        np.testing.assert_array_equal(data1[1].asnumpy(), golden)

        golden = np.array([i * 3])
        np.testing.assert_array_equal(data2[0].asnumpy(), golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(data2[1].asnumpy(), golden)
        i = i + 1


def test_generator_10():
    """
    Feature: GeneratorDataset
    Description: Test map column order when len(input_columns) != len(output_columns)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when len(input_columns) != len(output_columns).")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns="col1", output_columns=["out1", "out2"],
                      column_order=['col0', 'out1', 'out2'], num_parallel_workers=2)

    # Expected column order is |col0|out1|out2|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        golden = np.array([i])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        golden = np.array([[i * 5, (i + 1) * 5], [(i + 2) * 5, (i + 3) * 5]])
        np.testing.assert_array_equal(item[2], golden)
        i = i + 1


def test_generator_11():
    """
    Feature: GeneratorDataset
    Description: Test map column order len(input_columns) != len(output_columns), column_order drops some columns
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when len(input_columns) != len(output_columns), "
                "and column_order drops some columns.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns="col1", output_columns=["out1", "out2"],
                      column_order=['out1', 'out2'], num_parallel_workers=2)

    # Expected column order is |out1|out2|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        # len should be 2 because col0 is dropped (not included in column_order)
        assert len(item) == 2
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i * 5, (i + 1) * 5], [(i + 2) * 5, (i + 3) * 5]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1


def test_generator_12():
    """
    Feature: GeneratorDataset
    Description: Test map column order when input_columns and output_columns are None
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns and output_columns are None.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x * 5)), num_parallel_workers=2)

    # Expected column order is |col0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1

    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x * 5)), column_order=["col1", "col0"], num_parallel_workers=2)

    # Expected column order is |col0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[1], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1


def test_generator_13():
    """
    Feature: GeneratorDataset
    Description: Test map column order when input_columns is None
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns is None.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x * 5)), output_columns=["out0"], num_parallel_workers=2)

    # Expected column order is |out0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # len should be 2 because col0 is dropped (not included in column_order)
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_14():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator MP with CPP sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator MP : 0 - 63")
    # Sometimes there are some ITERATORS left in ITERATORS_LIST when run all UTs together,
    # and cause core dump and blocking in this UT. Add cleanup() here to fix it.
    it._cleanup()  # pylint: disable=W0212

    # Reduce memory needed by reducing queue size
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    source = [(np.array([x]),) for x in range(256)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler(),
                              num_parallel_workers=4, max_rowsize=1).repeat(2)
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1
        if i == 256:
            i = 0

    ds.config.set_prefetch_size(prefetch_original)


def test_generator_15():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator MP with Python sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator MP : 0 - 63")

    ## Reduce memory needed by reducing queue size
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    sampler = [x for x in range(256)]
    source = [(np.array([x]),) for x in range(256)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=sampler,
                              num_parallel_workers=4, max_rowsize=1).repeat(1)
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1
        if i == 256:
            i = 0

    ds.config.set_prefetch_size(prefetch_original)


def test_generator_16():
    """
    Feature: GeneratorDataset
    Description: Test multi column generator Mp with CPP sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator")

    source = [(np.array([x]), np.array([x + 1])) for x in range(256)]
    # apply dataset operations
    data1 = ds.GeneratorDataset(source, ["col0", "col1"], sampler=ds.SequentialSampler())

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["col0"], golden)
        golden = np.array([i + 1])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_17():
    """
    Feature: GeneratorDataset
    Description: Test multi column generator Mp with CPP sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator")

    sampler = [x for x in range(256)]
    source = [(np.array([x]), np.array([x + 1])) for x in range(256)]
    # apply dataset operations
    data1 = ds.GeneratorDataset(source, ["col0", "col1"], sampler=sampler)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["col0"], golden)
        golden = np.array([i + 1])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_18():
    """
    Feature: GeneratorDataset
    Description: Test multiprocessing flag (same as test 13 with python_multiprocessing=True flag)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns is None.")

    # Reduce shm usage by disabling this optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"], python_multiprocessing=True)
    data1 = data1.map(operations=(lambda x: (x * 5)), output_columns=["out0"], num_parallel_workers=2,
                      python_multiprocessing=True)

    # Expected column order is |out0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # len should be 2 because col0 is dropped (not included in column_order)
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item["out0"], golden)

    ds.config.set_enable_shared_mem(mem_original)


def test_generator_19():
    """
    Feature: GeneratorDataset
    Description: Test multiprocessing 2 different large columns
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns is None.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(DatasetGeneratorLarge(), ["col0", "col1"], python_multiprocessing=True, shuffle=False)

    # Expected column order is |out0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array(range(4000)) + i
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array(range(4000)) * 10
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1


class RandomAccessDataset:
    def __init__(self):
        self.__data = np.random.sample((5, 1))

    def __getitem__(self, item):
        return self.__data[item]

    def __len__(self):
        return 5


class RandomAccessDatasetWithoutLen:
    def __init__(self):
        self.__data = np.random.sample((5, 1))

    def __getitem__(self, item):
        return self.__data[item]


class IterableDataset:
    def __init__(self):
        self.count = 0
        self.max = 10

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max:
            raise StopIteration
        self.count += 1
        return (np.array(self.count),)


def test_generator_20():
    """
    Feature: GeneratorDataset
    Description: Test mappable and unmappable dataset as source for GeneratorDataset
    Expectation: The dataset is processed as expected
    """
    logger.info("Test mappable and unmappable dataset as source for GeneratorDataset.")

    # Mappable dataset
    data1 = ds.GeneratorDataset(RandomAccessDataset(), ["col0"])
    dataset_size1 = data1.get_dataset_size()
    assert dataset_size1 == 5

    # Mappable dataset without __len__
    data2 = ds.GeneratorDataset(RandomAccessDatasetWithoutLen(), ["col0"])
    try:
        data2.get_dataset_size()
    except RuntimeError as e:
        assert "'__len__' method is required" in str(e)

    # Unmappable dataset
    data3 = ds.GeneratorDataset(IterableDataset(), ["col0"])
    dataset_size3 = data3.get_dataset_size()
    assert dataset_size3 == 10


def test_generator_error_1():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with no data type of the converted NumPy array
    Expectation: Error is raised as expected
    """

    def generator_np():
        for i in range(64):
            yield (np.array([{i}]),)

    with pytest.raises(RuntimeError) as info:
        data1 = ds.GeneratorDataset(generator_np, ["data"])
        for _ in data1:
            pass
    assert "Data type of 1th item of the input or its converted Numpy array is expected" in str(info.value)


def test_generator_error_2():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with no data type of 1th item of the input
    Expectation: Error is raised as expected
    """

    def generator_np():
        for i in range(64):
            yield ({i},)

    with pytest.raises(RuntimeError) as info:
        data1 = ds.GeneratorDataset(generator_np, ["data"])
        for _ in data1:
            pass
    assert "Data type of 1th item of the input or its converted Numpy array is expected" in str(info.value)


def test_generator_error_3():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset when len(input_columns) != len(output_columns) and column_order is not specified
    Expectation: Error is raised as expected
    """
    with pytest.raises(ValueError) as info:
        # apply dataset operations
        data1 = ds.GeneratorDataset(generator_mc(2048), ["label", "image"])
        data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns=["label"], output_columns=["out1", "out2"],
                          num_parallel_workers=2)

        for _ in data1:
            pass
    assert "When length of input_columns and output_columns are not equal, column_order must be specified." in \
           str(info.value)


def test_generator_error_4():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset when number of columns in map does not match output_columns
    Expectation: Error is raised as expected
    """
    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.GeneratorDataset(generator_mc(2048), ["label", "image"])
        data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns=["label"],
                          num_parallel_workers=2)

        for _ in data1:
            pass
    assert "the number of columns returned in 'map' operations should match the number of 'output_columns'" \
           in str(info.value)


def test_generator_sequential_sampler():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler())
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1


def test_generator_random_sampler():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with random sampler
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    ds1 = ds.GeneratorDataset(source, ["data"], shuffle=True)
    for _ in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        pass


def test_generator_distributed_sampler():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with distributed sampler
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    for sid in range(8):
        ds1 = ds.GeneratorDataset(source, ["data"], shuffle=False, num_shards=8, shard_id=sid)
        i = sid
        for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(data["data"], golden)
            i = i + 8


def test_generator_num_samples():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with num_samples parameter
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    num_samples = 32
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler(num_samples=num_samples))
    ds2 = ds.GeneratorDataset(source, ["data"], sampler=[i for i in range(32)], num_samples=num_samples)
    ds3 = ds.GeneratorDataset(generator_1d, ["data"], num_samples=num_samples)

    count = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == num_samples

    count = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == num_samples

    count = 0
    for _ in ds3.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == num_samples


def test_generator_num_samples_underflow():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with underflowed num_samples parameter
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    num_samples = 256
    ds2 = ds.GeneratorDataset(source, ["data"], sampler=[i for i in range(64)], num_samples=num_samples)
    ds3 = ds.GeneratorDataset(generator_1d, ["data"], num_samples=num_samples)

    count = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == 64

    count = 0
    for _ in ds3.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == 64


def type_tester_with_type_check_2c_schema(t, c):
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with type check 2c
    Expectation: The dataset is processed as expected
    """
    logger.info("Test with Type {}".format(t.__name__))

    schema = ds.Schema()
    schema.add_column("data0", c[0])
    schema.add_column("data1", c[1])

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type_2c(t)), schema=schema)

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data0"], golden)
        i = i + 4


def test_generator_schema():
    """
    Feature: GeneratorDataset
    Description: Test 2 column Generator on different data type with type check with schema input
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 2 column Generator on all data types with type check")

    np_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
                np.float64]
    de_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32,
                mstype.uint64, mstype.float32, mstype.float64]

    for i, _ in enumerate(np_types):
        type_tester_with_type_check_2c_schema(np_types[i], [de_types[i], de_types[i]])


def test_generator_dataset_size_0():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset get_dataset_size by iterator method
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 get_dataset_size")

    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_1():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset get_dataset_size by __len__ method
    Expectation: The dataset is processed as expected
    """
    logger.info("Test DatasetGenerator get_dataset_size")

    dataset_generator = DatasetGenerator()
    data1 = ds.GeneratorDataset(dataset_generator, ["data"])

    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_2():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with repeat get_dataset_size
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator + repeat get_dataset_size")

    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)

    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_3():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with batch get_dataset_size
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator + batch get_dataset_size")

    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.batch(4)

    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_rows += 1
    assert data_size == num_rows


def test_generator_dataset_size_4():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with num_shards get_dataset_size
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 + num_shards get_dataset_size")

    dataset_generator = DatasetGenerator()
    data1 = ds.GeneratorDataset(dataset_generator, ["data"], num_shards=3, shard_id=0)
    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_5():
    """
    Feature: GeneratorDataset
    Description: Test get_dataset_size after create_dict_iterator
    Expectation: The dataset is processed as expected
    """
    logger.info("Test get_dataset_size after create_dict_iterator")

    dataset_generator = DatasetGenerator()
    data1 = ds.GeneratorDataset(dataset_generator, ["data"], num_shards=3, shard_id=0)

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        num_rows = num_rows + 1
    data_size = data1.get_dataset_size()
    assert data_size == num_rows


def manual_test_generator_keyboard_interrupt():
    """
    Feature: GeneratorDataset
    Description: Test keyboard_interrupt
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator MP : 0 - 63")

    class MyDS():
        def __getitem__(self, item):
            while True:
                pass

        def __len__(self):
            return 1024

    ds1 = ds.GeneratorDataset(MyDS(), ["data"], num_parallel_workers=4).repeat(2)
    for _ in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        pass


def test_explicit_deepcopy():
    """
    Feature: NumPyDataset
    Description: Test explicit_deepcopy
    Expectation: The dataset is processed as expected
    """
    logger.info("Test explicit_deepcopy")

    ds1 = ds.NumpySlicesDataset([1, 2], shuffle=False)
    ds2 = copy.deepcopy(ds1)
    for d1, d2 in zip(ds1, ds2):
        assert d1 == d2


def test_func_generator_dataset_005():
    """
    Feature: GeneratorDataset
    Description: Test Generator's class __getitem__
    Expectation: The dataset is processed as expected
    """
    result = [np.random.randn(242, 242, 242), np.random.randn(42, 24, 442)]

    class MyData():
        def __init__(self, input_para):
            self.data = input_para

        def __getitem__(self, item):
            return (Tensor(self.data[0]), Tensor(self.data[1]))

        def __len__(self):
            return 2

    column_names = ["col1", "col2"]
    dataset = ds.GeneratorDataset(MyData(result), column_names)
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert "col1" in str(data.keys())
        assert (data["col1"] == result[0]).all()
        assert (data["col2"] == result[1]).all()
        i += 1
    assert i == 2


def test_func_generator_dataset_with_zip_source():
    """
    Feature: Verify the source is zip
    Description: The source input is zip
    Expectation: Success
    """

    def synthetic_data(w, b, num_examples):
        """生成 y = Xw + b + 噪声。"""
        X = np.random.normal(0, 1, (num_examples, len(w)))
        y = np.matmul(X, w) + b
        y += np.random.normal(0, 0.01, y.shape)
        return X.astype(np.float32), y.reshape((-1, 1)).astype(np.float32)

    true_w = np.array([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 10)

    def load_array(data_arrays, column_names, batch_size, is_train=True):
        """构造一个MindSpore数据迭代器。"""
        dataset = ds.GeneratorDataset(data_arrays, column_names, shuffle=is_train)
        dataset = dataset.batch(batch_size)
        return dataset

    batch_size = 2
    dataset = load_array(zip(features, labels), ['features', 'labels'], batch_size)

    count = 0
    epochs = 10
    dataset_iter = dataset.create_dict_iterator(num_epochs=epochs, output_numpy=True)
    for _ in range(epochs):
        for _ in dataset_iter:
            count += 1
    assert count == 50


def test_generator_mixed_operator():
    """
    Feature: Test adding computing operator into user defined dataset
    Description: Will decrease num_parallel_worker into 1
    Expectation: Success
    """
    logger.info("Test adding computing operator into user defined dataset.")

    # create dataset
    data1 = ds.GeneratorDataset(DatasetGeneratorMixed(), ["col0"], shuffle=False, python_multiprocessing=False)
    assert data1.num_parallel_workers == 1

    for _ in data1.create_tuple_iterator(num_epochs=1):
        pass


def test_generator_single_input_0():
    """
    Feature: Test single int input
    Description: Input int
    Expectation: Success
    """

    def generator_int():
        for i in range(64):
            yield i

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [i for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [i for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_0(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array(i)
            np.testing.assert_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_0(generator_int)
    assert_generator_single_input_0(RandomAccessDatasetInner())
    assert_generator_single_input_0(SequentialAccessDataset())


def test_generator_single_input_1():
    """
    Feature: Test single float input
    Description: Input float
    Expectation: Success
    """

    def generator_float():
        for i in range(64):
            yield i * 0.1

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [i for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item] * 0.1

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [i for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index] * 0.1
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_1(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0.0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array(i)
            np.testing.assert_almost_equal(item["data"], golden)
            i = i + 0.1

    assert_generator_single_input_1(generator_float)
    assert_generator_single_input_1(RandomAccessDatasetInner())
    assert_generator_single_input_1(SequentialAccessDataset())


def test_generator_single_input_2():
    """
    Feature: Test single str input
    Description: Input str
    Expectation: Success
    """

    def generator_str():
        for i in range(64):
            yield chr(ord('a') + i)

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [i for i in range(64)]

        def __getitem__(self, item):
            return chr(ord('a') + self.__data[item])

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [i for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = chr(ord('a') + self.__data[self.__index])
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_2(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            s = chr(ord('a') + i)
            golden = np.array(s)
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_2(generator_str)
    assert_generator_single_input_2(RandomAccessDatasetInner())
    assert_generator_single_input_2(SequentialAccessDataset())


def test_generator_single_input_3():
    """
    Feature: Test single bytes input
    Description: Input bytes
    Expectation: Success
    """

    def generator_bytes():
        for i in range(64):
            yield bytes('a' * i, encoding='UTF-8')

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [bytes('a' * i, encoding='UTF-8') for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [bytes('a' * i, encoding='UTF-8') for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_3(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            b = bytes('a' * i, encoding='UTF-8')
            golden = np.array(b)
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_3(generator_bytes)
    assert_generator_single_input_3(RandomAccessDatasetInner())
    assert_generator_single_input_3(SequentialAccessDataset())


def test_generator_single_input_4():
    """
    Feature: Test single Tensor input
    Description: Input Tensor
    Expectation: Success
    """

    def generator_tensor():
        for i in range(64):
            yield Tensor(i)

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [Tensor(i) for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [Tensor(i) for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_4(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
            golden = Tensor(i)
            assert item["data"] == golden
            i = i + 1

    assert_generator_single_input_4(generator_tensor)
    assert_generator_single_input_4(RandomAccessDatasetInner())
    assert_generator_single_input_4(SequentialAccessDataset())


def test_generator_single_input_5():
    """
    Feature: Test single np.array input
    Description: Input np.array
    Expectation: Success
    """

    def generator_np():
        for i in range(64):
            yield np.ones(i)

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [np.ones(i) for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [np.ones(i) for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_5(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.ones(i)
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_5(generator_np)
    assert_generator_single_input_5(RandomAccessDatasetInner())
    assert_generator_single_input_5(SequentialAccessDataset())


def test_generator_single_input_6():
    """
    Feature: Test single np.array input whose dtype is object
    Description: Input np.array
    Expectation: Throw exception
    """

    def generator_nested_np():
        for i in range(64):
            yield np.array([[i, i + 1], [i, i + 1, i + 2]])

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [np.array([[i, i + 1], [i, i + 1, i + 2]]) for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDatasetInner:
        def __init__(self):
            self.__data = [np.array([[i, i + 1], [i, i + 1, i + 2]]) for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_6(data):
        # apply dataset operations

        with pytest.raises(RuntimeError) as info:
            data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
                pass
        assert " Data type of the input or its converted Numpy array is expected" in str(info.value)

    assert_generator_single_input_6(generator_nested_np)
    assert_generator_single_input_6(RandomAccessDatasetInner())
    assert_generator_single_input_6(SequentialAccessDatasetInner())


def test_generator_with_single_numpy():
    """
    Feature: Test GeneratorDataset with single numpy and multi columns when use __getitem__
    Description: Single numpy, tuple numpy with single columns and multi columns
    Expectation: Success
    """

    class get_dataset_generator:
        def __init__(self, value):
            np.random.seed(58)
            self.__value = value

        def __getitem__(self, index):
            return self.__value

        def __len__(self):
            return 20

    def test_generator_one_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert (data["data"] == value).all()
            count += 1
        assert count == 20

    # test user define one column
    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_one_column(numpy_1)
    test_generator_one_column(numpy_2)
    test_generator_one_column(numpy_3)
    test_generator_one_column(numpy_4)
    test_generator_one_column(numpy_5)
    test_generator_one_column(numpy_6)
    test_generator_one_column(numpy_7)
    test_generator_one_column(numpy_8)
    test_generator_one_column(numpy_9)
    test_generator_one_column(numpy_10)

    tuple_1 = (numpy_7,)
    dataset_generator = get_dataset_generator(tuple_1)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == tuple_1[0]).all()
        count += 1
    assert count == 20

    tuple_2 = (numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_4 = (numpy_4, numpy_5, numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_4)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:4" in str(info.value)

    # test user define two column
    def test_generator_two_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:2 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_two_column(numpy_1)
    test_generator_two_column(numpy_2)
    test_generator_two_column(numpy_3)
    test_generator_two_column(numpy_4)
    test_generator_two_column(numpy_5)
    test_generator_two_column(numpy_6)
    test_generator_two_column(numpy_7)
    test_generator_two_column(numpy_8)
    test_generator_two_column(numpy_9)
    test_generator_two_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_two_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    dataset_generator = get_dataset_generator(tuple_2)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_2).all()
        assert (data["label"] == numpy_3).all()
        count += 1
    assert count == 20

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:2 and number of returned NumPy array is:3" in str(info.value)

    # test user define three column
    def test_generator_three_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False,
                                      num_parallel_workers=number, python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:3 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_three_column(numpy_1)
    test_generator_three_column(numpy_2)
    test_generator_three_column(numpy_3)
    test_generator_three_column(numpy_4)
    test_generator_three_column(numpy_5)
    test_generator_three_column(numpy_6)
    test_generator_three_column(numpy_7)
    test_generator_three_column(numpy_8)
    test_generator_three_column(numpy_9)
    test_generator_three_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_three_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:3 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    dataset_generator = get_dataset_generator(tuple_3)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_4).all()
        assert (data["label"] == numpy_5).all()
        assert (data["label2"] == numpy_6).all()
        count += 1
    assert count == 20


def test_generator_with_single_numpy_with_next():
    """
    Feature: Test GeneratorDataset with single numpy and multi columns when use __next__
    Description: Single numpy, tuple numpy with single columns and multi columns
    Expectation: Success
    """

    class get_dataset_generator:
        def __init__(self, value):
            np.random.seed(58)
            self.__value = value
            self.__index = 0

        def __next__(self):
            if self.__index >= 20:
                raise StopIteration

            self.__index += 1
            return self.__value

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 20

    def test_generator_one_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert (data["data"] == value).all()
            count += 1
        assert count == 20

    # test user define one column
    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_one_column(numpy_1)
    test_generator_one_column(numpy_2)
    test_generator_one_column(numpy_3)
    test_generator_one_column(numpy_4)
    test_generator_one_column(numpy_5)
    test_generator_one_column(numpy_6)
    test_generator_one_column(numpy_7)
    test_generator_one_column(numpy_8)
    test_generator_one_column(numpy_9)
    test_generator_one_column(numpy_10)

    tuple_1 = (numpy_7,)
    dataset_generator = get_dataset_generator(tuple_1)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == tuple_1[0]).all()
        count += 1
    assert count == 20

    tuple_2 = (numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_1, numpy_2)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_4 = (numpy_4, numpy_5, numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_4)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:4" in str(info.value)

    # test user define two column
    def test_generator_two_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:2 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_two_column(numpy_1)
    test_generator_two_column(numpy_2)
    test_generator_two_column(numpy_3)
    test_generator_two_column(numpy_4)
    test_generator_two_column(numpy_5)
    test_generator_two_column(numpy_6)
    test_generator_two_column(numpy_7)
    test_generator_two_column(numpy_8)
    test_generator_two_column(numpy_9)
    test_generator_two_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_two_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    dataset_generator = get_dataset_generator(tuple_2)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_2).all()
        assert (data["label"] == numpy_3).all()
        count += 1
    assert count == 20

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:2 and number of returned NumPy array is:3" in str(info.value)

    # test user define three column
    def test_generator_three_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False,
                                      num_parallel_workers=number, python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:3 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_three_column(numpy_1)
    test_generator_three_column(numpy_2)
    test_generator_three_column(numpy_3)
    test_generator_three_column(numpy_4)
    test_generator_three_column(numpy_5)
    test_generator_three_column(numpy_6)
    test_generator_three_column(numpy_7)
    test_generator_three_column(numpy_8)
    test_generator_three_column(numpy_9)
    test_generator_three_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_three_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:3 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    dataset_generator = get_dataset_generator(tuple_3)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_4).all()
        assert (data["label"] == numpy_5).all()
        assert (data["label2"] == numpy_6).all()
        count += 1
    assert count == 20


def test_generator_with_single_numpy_with_yield():
    """
    Feature: Test GeneratorDataset with single numpy and multi columns when use yield
    Description: Single numpy, tuple numpy with single columns and multi columns
    Expectation: Success
    """

    def get_dataset_generator(value):
        for _ in range(20):
            yield value

    def test_generator_one_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert (data["data"] == value).all()
            count += 1
        assert count == 20

    # test user define one column
    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_one_column(numpy_1)
    test_generator_one_column(numpy_2)
    test_generator_one_column(numpy_3)
    test_generator_one_column(numpy_4)
    test_generator_one_column(numpy_5)
    test_generator_one_column(numpy_6)
    test_generator_one_column(numpy_7)
    test_generator_one_column(numpy_8)
    test_generator_one_column(numpy_9)
    test_generator_one_column(numpy_10)

    tuple_1 = (numpy_7,)
    dataset_generator = get_dataset_generator(tuple_1)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == tuple_1[0]).all()
        count += 1
    assert count == 20

    tuple_2 = (numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_1, numpy_2)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_4 = (numpy_4, numpy_5, numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_4)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:4" in str(info.value)

    # test user define two column
    def test_generator_two_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:2 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_two_column(numpy_1)
    test_generator_two_column(numpy_2)
    test_generator_two_column(numpy_3)
    test_generator_two_column(numpy_4)
    test_generator_two_column(numpy_5)
    test_generator_two_column(numpy_6)
    test_generator_two_column(numpy_7)
    test_generator_two_column(numpy_8)
    test_generator_two_column(numpy_9)
    test_generator_two_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_two_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    dataset_generator = get_dataset_generator(tuple_2)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_2).all()
        assert (data["label"] == numpy_3).all()
        count += 1
    assert count == 20

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:2 and number of returned NumPy array is:3" in str(info.value)

    # test user define three column
    def test_generator_three_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False,
                                      num_parallel_workers=number, python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:3 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_three_column(numpy_1)
    test_generator_three_column(numpy_2)
    test_generator_three_column(numpy_3)
    test_generator_three_column(numpy_4)
    test_generator_three_column(numpy_5)
    test_generator_three_column(numpy_6)
    test_generator_three_column(numpy_7)
    test_generator_three_column(numpy_8)
    test_generator_three_column(numpy_9)
    test_generator_three_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_three_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:3 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    dataset_generator = get_dataset_generator(tuple_3)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_4).all()
        assert (data["label"] == numpy_5).all()
        assert (data["label2"] == numpy_6).all()
        count += 1
    assert count == 20


@pytest.mark.skip(reason="only for testing stuck scenario")
def test_generator_traceback():
    """
    Feature: GeneratorDataset
    Description: Generator is too slow then main process will log the stack of the stuck process
    Expectation: The stuck locality can be logged
    """
    class SlowDataset:
        def __init__(self):
            self.data = np.random.randint(0, 255, (100, 28, 28, 3), dtype=np.uint8)

        def __getitem__(self, index):
            if index % 10 == 0:
                time.sleep(600)
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = ds.GeneratorDataset(SlowDataset(), column_names=["image"], num_parallel_workers=8)
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        pass


if __name__ == "__main__":
    test_generator_0()
    test_generator_1()
    test_generator_2()
    test_generator_3()
    test_generator_4()
    test_generator_5()
    test_generator_6()
    test_generator_7()
    test_generator_8()
    test_generator_9()
    test_generator_10()
    test_generator_11()
    test_generator_12()
    test_generator_13()
    test_generator_14()
    test_generator_15()
    test_generator_16()
    test_generator_17()
    test_generator_18()
    test_generator_19()
    test_generator_20()
    test_generator_error_1()
    test_generator_error_2()
    test_generator_error_4()
    test_generator_sequential_sampler()
    test_generator_random_sampler()
    test_generator_distributed_sampler()
    test_generator_num_samples()
    test_generator_num_samples_underflow()
    test_generator_schema()
    test_generator_dataset_size_0()
    test_generator_dataset_size_1()
    test_generator_dataset_size_2()
    test_generator_dataset_size_3()
    test_generator_dataset_size_4()
    test_generator_dataset_size_5()
    test_explicit_deepcopy()
    test_func_generator_dataset_005()
    test_func_generator_dataset_with_zip_source()
    test_generator_mixed_operator()
    test_generator_single_input_0()
    test_generator_single_input_1()
    test_generator_single_input_2()
    test_generator_single_input_3()
    test_generator_single_input_4()
    test_generator_single_input_5()
    test_generator_single_input_6()
    test_generator_with_seed_5489_when_dist()
    test_generator_with_set_seed_when_dist()
    test_generator_with_single_numpy()
    test_generator_with_single_numpy_with_next()
    test_generator_with_single_numpy_with_yield()
    test_generator_traceback()
