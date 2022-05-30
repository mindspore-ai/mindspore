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
Test Repeat Op
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import save_and_check_dict

DATA_DIR_TF = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR_TF = "../data/dataset/testTFTestAllTypes/datasetSchema.json"

DATA_DIR_TF2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR_TF2 = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_tf_repeat_01():
    """
    Feature: Repeat op
    Description: Test repeat op under simple case
    Expectation: Output is the same as expected output
    """
    logger.info("Test Simple Repeat")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.repeat(repeat_count)

    filename = "repeat_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_tf_repeat_02():
    """
    Feature: Repeat op
    Description: Test repeat op with infinite count
    Expectation: Runs successfully
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
    """
    Feature: Repeat op
    Description: Test repeat op then batch op
    Expectation: Output is the same as expected output
    """
    logger.info("Test Repeat then Batch")
    data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)

    batch_size = 32
    resize_height, resize_width = 32, 32
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), interpolation=ds.transforms.vision.Inter.LINEAR)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.repeat(22)
    data1 = data1.batch(batch_size, drop_remainder=True)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    logger.info("Number of tf data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_tf_repeat_04():
    """
    Feature: Repeat op
    Description: Test repeat op under simple case with column list
    Expectation: Output is the same as expected output
    """
    logger.info("Test Simple Repeat Column List")
    # define parameters
    repeat_count = 2
    columns_list = ["col_sint64", "col_sint32"]
    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, columns_list=columns_list, shuffle=False)
    data1 = data1.repeat(repeat_count)

    filename = "repeat_list_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def generator():
    for i in range(3):
        (yield np.array([i]),)


def test_nested_repeat1():
    """
    Feature: Repeat op
    Description: Test nested repeat with count > 1
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat1")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(3)

    for i, d in enumerate(data.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3 * 3


def test_nested_repeat2():
    """
    Feature: Repeat op
    Description: Test nested repeat with count = 1
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat2")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(1)
    data = data.repeat(1)

    for i, d in enumerate(data.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 3


def test_nested_repeat3():
    """
    Feature: Repeat op
    Description: Test nested repeat with first repeat with count = 1 and second repeat with count > 1
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat3")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(1)
    data = data.repeat(2)

    for i, d in enumerate(data.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3


def test_nested_repeat4():
    """
    Feature: Repeat op
    Description: Test nested repeat with first repeat with count > 1 and second input with count = 1
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat4")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(1)

    for i, d in enumerate(data.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert i % 3 == d[0][0]

    assert sum([1 for _ in data]) == 2 * 3


def test_nested_repeat5():
    """
    Feature: Repeat op
    Description: Test nested repeat after a batch operation
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat5")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.batch(3)
    data = data.repeat(2)
    data = data.repeat(3)

    for _, d in enumerate(data):
        np.testing.assert_array_equal(d[0].asnumpy(), np.asarray([[0], [1], [2]]))

    assert sum([1 for _ in data]) == 6


def test_nested_repeat6():
    """
    Feature: Repeat op
    Description: Test nested repeat with batch op (repeat -> batch -> repeat)
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat6")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.batch(3)
    data = data.repeat(3)

    for _, d in enumerate(data):
        np.testing.assert_array_equal(d[0].asnumpy(), np.asarray([[0], [1], [2]]))

    assert sum([1 for _ in data]) == 6


def test_nested_repeat7():
    """
    Feature: Repeat op
    Description: Test nested repeat followed by a batch op
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat7")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(3)
    data = data.batch(3)

    for _, d in enumerate(data):
        np.testing.assert_array_equal(d[0].asnumpy(), np.asarray([[0], [1], [2]]))

    assert sum([1 for _ in data]) == 6


def test_nested_repeat8():
    """
    Feature: Repeat op
    Description: Test nested repeat after a batch operation with drop_remainder=False
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat8")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.batch(2, drop_remainder=False)
    data = data.repeat(2)
    data = data.repeat(3)

    for i, d in enumerate(data):
        if i % 2 == 0:
            np.testing.assert_array_equal(d[0].asnumpy(), np.asarray([[0], [1]]))
        else:
            np.testing.assert_array_equal(d[0].asnumpy(), np.asarray([[2]]))

    assert sum([1 for _ in data]) == 6 * 2


def test_nested_repeat9():
    """
    Feature: Repeat op
    Description: Test nested repeat with first repeat with no count
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat9")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat()
    data = data.repeat(3)

    for i, d in enumerate(data):
        assert i % 3 == d[0].asnumpy()[0]
        if i == 10:
            break


def test_nested_repeat10():
    """
    Feature: Repeat op
    Description: Test nested repeat with second repeat with no count
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat10")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(3)
    data = data.repeat()

    for i, d in enumerate(data):
        assert i % 3 == d[0].asnumpy()[0]
        if i == 10:
            break


def test_nested_repeat11():
    """
    Feature: Repeat op
    Description: Test nested repeat (4 repeat ops)
    Expectation: Output is the same as expected output
    """
    logger.info("test_nested_repeat11")
    data = ds.GeneratorDataset(generator, ["data"])
    data = data.repeat(2)
    data = data.repeat(3)
    data = data.repeat(4)
    data = data.repeat(5)

    for i, d in enumerate(data):
        assert i % 3 == d[0].asnumpy()[0]

    assert sum([1 for _ in data]) == 2 * 3 * 4 * 5 * 3


def test_repeat_count1():
    """
    Feature: Repeat op
    Description: Test repeat after multiple operations, then followed by batch op
    Expectation: Output is the same as expected output
    """
    data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is {}".format(data1_size))
    batch_size = 2
    repeat_count = 4
    resize_height, resize_width = 32, 32
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), interpolation=ds.transforms.vision.Inter.LINEAR)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.repeat(repeat_count)
    data1 = data1.batch(batch_size, drop_remainder=False)
    dataset_size = data1.get_dataset_size()
    logger.info("dataset repeat then batch's size is {}".format(dataset_size))
    num1_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num1_iter += 1

    assert data1_size == 3
    assert dataset_size == num1_iter == 6


def test_repeat_count2():
    """
    Feature: Repeat op
    Description: Test repeat after multiple operations and a batch op
    Expectation: Output is the same as expected output
    """
    data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is {}".format(data1_size))
    batch_size = 2
    repeat_count = 4
    resize_height, resize_width = 32, 32
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), interpolation=ds.transforms.vision.Inter.LINEAR)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.batch(batch_size, drop_remainder=False)
    data1 = data1.repeat(repeat_count)
    dataset_size = data1.get_dataset_size()
    logger.info("dataset batch then repeat's size is {}".format(dataset_size))
    num1_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num1_iter += 1

    assert data1_size == 3
    assert dataset_size == num1_iter == 8


def test_repeat_count0():
    """
    Feature: Repeat op
    Description: Test repeat with invalid count = 0
    Expectation: Error is raised as expected
    """
    logger.info("Test Repeat with invalid count 0")
    with pytest.raises(ValueError) as info:
        data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)
        data1.repeat(0)
    assert "count" in str(info.value)


def test_repeat_countneg2():
    """
    Feature: Repeat op
    Description: Test nested repeat with invalid count = -2
    Expectation: Error is raised as expected
    """
    logger.info("Test Repeat with invalid count -2")
    with pytest.raises(ValueError) as info:
        data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)
        data1.repeat(-2)
    assert "count" in str(info.value)


if __name__ == "__main__":
    test_tf_repeat_01()
    test_tf_repeat_02()
    test_tf_repeat_03()
    test_tf_repeat_04()
    test_nested_repeat1()
    test_nested_repeat2()
    test_nested_repeat3()
    test_nested_repeat4()
    test_nested_repeat5()
    test_nested_repeat6()
    test_nested_repeat7()
    test_nested_repeat8()
    test_nested_repeat9()
    test_nested_repeat10()
    test_nested_repeat11()
    test_repeat_count1()
    test_repeat_count2()
    test_repeat_count0()
    test_repeat_countneg2()
