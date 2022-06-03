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
"""
Testing Epoch Control op in DE
"""
import itertools
import numpy as np
import pytest
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def diff_mse(in1, in2):
    """
    diff_mse
    """
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def test_cifar10():
    """
    Feature: Epoch Control op
    Description: Test num_epochs as tuple iterator param for Cifar10Dataset
    Expectation: Output is equal to the expected output
    """
    logger.info("Test dataset parameter")
    data_dir_10 = "../data/dataset/testCifar10Data"
    num_repeat = 2
    batch_size = 32
    limit_dataset = 100
    # apply dataset operations
    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=limit_dataset)
    data1 = data1.repeat(num_repeat)
    data1 = data1.batch(batch_size, True)
    num_epoch = 5
    iter1 = data1.create_tuple_iterator(num_epochs=num_epoch)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            # in this example, each dictionary has keys "image" and "label"
            row_count += 1
        assert row_count == int(limit_dataset * num_repeat / batch_size)
        logger.debug("row_count: ", row_count)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    logger.debug("total epochs: ", epoch_count)
    assert sample_count == int(limit_dataset * num_repeat / batch_size) * num_epoch
    logger.debug("total sample: ", sample_count)


def test_decode_op():
    """
    Feature: Epoch Control op
    Description: Test num_epochs as dict iterator param for dataset which Decode op has been applied onto it
    Expectation: Output is equal to the expected output before iterator is stopped, then correct error is raised
    """
    logger.info("test_decode_op")

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # Serialize and Load dataset requires using vision.Decode instead of vision.Decode().
    data1 = data1.map(operations=[vision.Decode()], input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    num_epoch = 5
    # iter1 will always assume there is a next epoch and never shutdown.
    iter1 = data1.create_dict_iterator(num_epochs=-1, output_numpy=True)
    # iter 2 will stop and shutdown pipeline after num_epoch
    iter2 = data2.create_dict_iterator(num_epoch, output_numpy=True)
    for _ in range(num_epoch):
        i = 0
        for item1, item2 in itertools.zip_longest(iter1, iter2):
            actual = item1["image"]
            expected = cv2.imdecode(item2["image"], cv2.IMREAD_COLOR)
            expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
            assert actual.shape == expected.shape
            diff = actual - expected
            mse = np.sum(np.power(diff, 2))
            assert mse == 0
            i = i + 1
        assert i == 3

    # Users have the option to manually stop the iterator, or rely on garbage collector.
    iter1.stop()
    # Expect a AttributeError since iter1 has been stopped.
    with pytest.raises(AttributeError) as info:
        iter1.__next__()
    assert "object has no attribute '_runtime_context'" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        iter2.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


# Generate 1d int numpy array from 0 - 63
def generator_1d():
    """
    generator
    """
    for i in range(64):
        yield (np.array([i]),)


def test_generator_dict_0():
    """
    Feature: Epoch Control op
    Description: Test dict iterator inside the loop declaration for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    i = 0
    # create the iterator inside the loop declaration
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


def test_generator_dict_1():
    """
    Feature: Epoch Control op
    Description: Test dict iterator outside the epoch for loop for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    for _ in range(10):
        i = 0
        # BAD. Do not create iterator every time inside.
        # Create iterator outside the epoch for loop.
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1
        assert i == 64


def test_generator_dict_2():
    """
    Feature: Epoch Control op
    Description: Test dict iterator with num_epochs=-1 for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output and iterator never shutdown
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = data1.create_dict_iterator(num_epochs=-1)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"].asnumpy(), golden)
            i = i + 1
        assert i == 64

    # iter1 is still alive and running.
    item1 = iter1.__next__()
    assert item1
    # rely on garbage collector to destroy iter1


def test_generator_dict_3():
    """
    Feature: Epoch Control op
    Description: Test dict iterator with num_epochs=-1 followed by stop for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output before stop, then error is raised
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = data1.create_dict_iterator(num_epochs=-1)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"].asnumpy(), golden)
            i = i + 1
        assert i == 64

    iter1.stop()
    # Expect a AttributeError since iter1 has been stopped.
    with pytest.raises(AttributeError) as info:
        iter1.__next__()
    assert "object has no attribute '_runtime_context'" in str(info.value)


def test_generator_dict_4():
    """
    Feature: Epoch Control op
    Description: Test dict iterator by fetching data beyond the specified number of epochs for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    iter1 = data1.create_dict_iterator(num_epochs=10)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"].asnumpy(), golden)
            i = i + 1
        assert i == 64

    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


def test_generator_dict_4_1():
    """
    Feature: Epoch Control op
    Description: Test dict iterator by fetching data beyond the specified number of epochs where num_epochs=1 so
        Epoch Control op will not be injected, using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    # epoch ctrl op will not be injected if num_epochs is 1.
    iter1 = data1.create_dict_iterator(num_epochs=1, output_numpy=True)
    for _ in range(1):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1
        assert i == 64

    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


def test_generator_dict_4_2():
    """
    Feature: Epoch Control op
    Description: Test dict iterator by fetching data beyond the specified number of epochs where num_epochs=1 so
        Epoch Control op will not be injected, after repeat op with num_repeat=1, using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    # repeat will not be injected when num repeat is 1.
    data1 = data1.repeat(1)
    # epoch ctrl op will not be injected if num_epochs is 1.
    iter1 = data1.create_dict_iterator(num_epochs=1, output_numpy=True)
    for _ in range(1):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1
        assert i == 64

    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


def test_generator_dict_5():
    """
    Feature: Epoch Control op
    Description: Test dict iterator by fetching data below (2 loops) then
        beyond the specified number of epochs using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    iter1 = data1.create_dict_iterator(num_epochs=11, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1
        assert i == 64

    # still one more epoch left in the iter1.
    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1
    assert i == 64

    # now iter1 has been exhausted, c++ pipeline has been shut down.
    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


# Test tuple iterator

def test_generator_tuple_0():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator inside the loop declaration for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    i = 0
    # create the iterator inside the loop declaration
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1


def test_generator_tuple_1():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator outside the epoch for loop for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    for _ in range(10):
        i = 0
        # BAD. Do not create iterator every time inside.
        # Create iterator outside the epoch for loop.
        for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64


def test_generator_tuple_2():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator with num_epochs=-1 for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output and iterator never shutdown
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = data1.create_tuple_iterator(num_epochs=-1, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64

    # iter1 is still alive and running.
    item1 = iter1.__next__()
    assert item1
    # rely on garbage collector to destroy iter1


def test_generator_tuple_3():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator with num_epochs=-1 followed by stop for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output before stop, then error is raised
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = data1.create_tuple_iterator(num_epochs=-1, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64

    iter1.stop()
    # Expect a AttributeError since iter1 has been stopped.
    with pytest.raises(AttributeError) as info:
        iter1.__next__()
    assert "object has no attribute '_runtime_context'" in str(info.value)


def test_generator_tuple_4():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by fetching data beyond the specified num_epochs for 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    iter1 = data1.create_tuple_iterator(num_epochs=10, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64

    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


def test_generator_tuple_5():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by fetching data below (2 loops) then
        beyond the specified number of epochs using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    iter1 = data1.create_tuple_iterator(num_epochs=11, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64

    # still one more epoch left in the iter1.
    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
    assert i == 64

    # now iter1 has been exhausted, c++ pipeline has been shut down.
    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


# Test with repeat
def test_generator_tuple_repeat_1():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by applying Repeat op first, next fetching data below (2 loops) then
        beyond the specified number of epochs using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)
    iter1 = data1.create_tuple_iterator(num_epochs=11, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2

    # still one more epoch left in the iter1.
    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i % 64])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
    assert i == 64 * 2

    # now iter1 has been exhausted, c++ pipeline has been shut down.
    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


# Test with repeat
def test_generator_tuple_repeat_repeat_1():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by applying Repeat op first twice, next fetching data below (2 loops) then
        beyond the specified number of epochs using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output when fetching data under the specified num_epochs,
        then error is raised due to EOF buffer encountered
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)
    data1 = data1.repeat(3)
    iter1 = data1.create_tuple_iterator(num_epochs=11, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2 * 3

    # still one more epoch left in the iter1.
    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i % 64])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
    assert i == 64 * 2 * 3

    # now iter1 has been exhausted, c++ pipeline has been shut down.
    with pytest.raises(RuntimeError) as info:
        iter1.__next__()
    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    assert err_msg in str(info.value)


def test_generator_tuple_repeat_repeat_2():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator with num_epochs=-1 by applying Repeat op first twice, next
        stop op is called on the iterator using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output before stop is called, then error is raised
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)
    data1 = data1.repeat(3)
    # iter1 will always assume there is a next epoch and never shutdown
    iter1 = data1.create_tuple_iterator(num_epochs=-1, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2 * 3

    iter1.stop()
    # Expect a AttributeError since iter1 has been stopped.
    with pytest.raises(AttributeError) as info:
        iter1.__next__()
    assert "object has no attribute '_runtime_context'" in str(info.value)


def test_generator_tuple_repeat_repeat_3():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by applying Repeat op first twice, then do 2 loops
        that the sum of iteration is equal to the specified num_epochs using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)
    data1 = data1.repeat(3)
    iter1 = data1.create_tuple_iterator(num_epochs=15, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2 * 3

    for _ in range(5):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2 * 3

    # rely on garbage collector to destroy iter1


def test_generator_tuple_infinite_repeat_repeat_1():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by applying infinite Repeat then Repeat with specified num_repeat,
        then iterate using iterator using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat()
    data1 = data1.repeat(3)
    iter1 = data1.create_tuple_iterator(num_epochs=11, output_numpy=True)

    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i % 64])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
        if i == 100:
            break

    # rely on garbage collector to destroy iter1


def test_generator_tuple_infinite_repeat_repeat_2():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by applying Repeat with specified num_repeat then infinite Repeat,
        then iterate using iterator using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(3)
    data1 = data1.repeat()
    iter1 = data1.create_tuple_iterator(num_epochs=11, output_numpy=True)

    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i % 64])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
        if i == 100:
            break

    # rely on garbage collector to destroy iter1


def test_generator_tuple_infinite_repeat_repeat_3():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator by applying infinite Repeat first twice,
        then iterate using iterator using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat()
    data1 = data1.repeat()
    iter1 = data1.create_tuple_iterator(num_epochs=11, output_numpy=True)

    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i % 64])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
        if i == 100:
            break

    # rely on garbage collector to destroy iter1


def test_generator_tuple_infinite_repeat_repeat_4():
    """
    Feature: Epoch Control op
    Description: Test tuple iterator with num_epochs=1 by applying infinite Repeat first twice,
        then iterate using iterator using 1D GeneratorDataset 0-63
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat()
    data1 = data1.repeat()
    iter1 = data1.create_tuple_iterator(num_epochs=1, output_numpy=True)

    i = 0
    for item in iter1:  # each data is a dictionary
        golden = np.array([i % 64])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1
        if i == 100:
            break

    # rely on garbage collector to destroy iter1


def test_generator_reusedataset():
    """
    Feature: Epoch Control op
    Description: Test iterator and other op (Repeat/Batch) on 1D GeneratorDataset 0-63 which previously
        has been applied with iterator and other op (Repeat/Batch)
    Expectation: Output is equal to the expected output
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)
    iter1 = data1.create_tuple_iterator(num_epochs=10, output_numpy=True)
    for _ in range(10):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2

    data1 = data1.repeat(3)
    iter1 = data1.create_tuple_iterator(num_epochs=5, output_numpy=True)
    for _ in range(5):
        i = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([i % 64])
            np.testing.assert_array_equal(item[0], golden)
            i = i + 1
        assert i == 64 * 2 * 3

    data1 = data1.batch(2)
    iter1 = data1.create_dict_iterator(num_epochs=5, output_numpy=True)
    for _ in range(5):
        i = 0
        sample = 0
        for item in iter1:  # each data is a dictionary
            golden = np.array([[i % 64], [(i + 1) % 64]])
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 2
            sample = sample + 1
        assert sample == 64 * 3

    # rely on garbage collector to destroy iter1
