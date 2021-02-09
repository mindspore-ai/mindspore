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
import copy
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import log as logger


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


def test_generator_0():
    """
    Test 1D Generator
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
    Test MD Generator
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
    Test multi column generator
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
    Test 1D Generator + repeat(4)
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
    Test fixed size 1D Generator + batch
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
    Test 1D Generator on different data type
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
    Test 1D Generator on different data type with type check
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
    Test 2 column Generator on different data type with type check
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
    Test multi column generator with few mapops
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
    Test map column order when len(input_columns) == len(output_columns).
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
    Test map column order when len(input_columns) != len(output_columns).
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
    Test map column order when len(input_columns) != len(output_columns).
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
    Test map column order when input_columns and output_columns are None.
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
    Test map column order when input_columns is None.
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
    Test 1D Generator MP + CPP sampler
    """
    logger.info("Test 1D Generator MP : 0 - 63")

    source = [(np.array([x]),) for x in range(256)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler(), num_parallel_workers=4).repeat(2)
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1
        if i == 256:
            i = 0


def test_generator_15():
    """
    Test 1D Generator MP + Python sampler
    """
    logger.info("Test 1D Generator MP : 0 - 63")

    sampler = [x for x in range(256)]
    source = [(np.array([x]),) for x in range(256)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=sampler, num_parallel_workers=4).repeat(2)
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1
        if i == 256:
            i = 0


def test_generator_16():
    """
    Test multi column generator Mp + CPP sampler
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
    Test multi column generator Mp + Python sampler
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


def test_generator_error_1():
    def generator_np():
        for i in range(64):
            yield (np.array([{i}]),)

    with pytest.raises(RuntimeError) as info:
        data1 = ds.GeneratorDataset(generator_np, ["data"])
        for _ in data1:
            pass
    assert "Invalid data type" in str(info.value)


def test_generator_error_2():
    def generator_np():
        for i in range(64):
            yield ({i},)

    with pytest.raises(RuntimeError) as info:
        data1 = ds.GeneratorDataset(generator_np, ["data"])
        for _ in data1:
            pass
    print("========", str(info.value))
    assert "Generator should return a tuple of numpy arrays" in str(info.value)


def test_generator_error_3():
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
    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.GeneratorDataset(generator_mc(2048), ["label", "image"])
        data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns=["label"],
                          num_parallel_workers=2)

        for _ in data1:
            pass
    assert "Unexpected error. Result of a tensorOp doesn't match output column names" in str(info.value)


def test_generator_sequential_sampler():
    source = [(np.array([x]),) for x in range(64)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler())
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1


def test_generator_random_sampler():
    source = [(np.array([x]),) for x in range(64)]
    ds1 = ds.GeneratorDataset(source, ["data"], shuffle=True)
    for _ in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        pass


def test_generator_distributed_sampler():
    source = [(np.array([x]),) for x in range(64)]
    for sid in range(8):
        ds1 = ds.GeneratorDataset(source, ["data"], shuffle=False, num_shards=8, shard_id=sid)
        i = sid
        for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(data["data"], golden)
            i = i + 8


def test_generator_num_samples():
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
    Test 2 column Generator on different data type with type check with schema input
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
    Test GeneratorDataset get_dataset_size by iterator method.
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
    Test GeneratorDataset get_dataset_size by __len__ method.
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
    Test GeneratorDataset + repeat get_dataset_size
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
    Test GeneratorDataset + batch get_dataset_size
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
    Test GeneratorDataset + num_shards
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
    Test get_dataset_size after create_dict_iterator
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
    Test keyboard_interrupt
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
    Test explicit_deepcopy
    """
    logger.info("Test explicit_deepcopy")

    ds1 = ds.NumpySlicesDataset([1, 2], shuffle=False)
    ds2 = copy.deepcopy(ds1)
    for d1, d2 in zip(ds1, ds2):
        assert d1 == d2


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
    test_generator_error_1()
    test_generator_error_2()
    test_generator_error_3()
    test_generator_error_4()
    test_generator_sequential_sampler()
    test_generator_distributed_sampler()
    test_generator_random_sampler()
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
