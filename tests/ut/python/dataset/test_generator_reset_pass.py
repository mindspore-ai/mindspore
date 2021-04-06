# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.dataset as ds
from mindspore import log as logger

# Generate 2 rows of data (1, 2)
def generator_1to2():
    for i in np.array([1, 2]):
        yield (np.array(i),)

# Generate 3 rows of data (10, 11, 12)
def generator_10to12():
    for i in np.array([10, 11, 12]):
        yield (np.array(i),)

# Generate 3 rows of data (22, 23, 24)
def generator_22to24():
    for i in np.array([22, 23, 24]):
        yield (np.array(i),)

def test_simple_repeat():

    # Since number of epoch is 1, the GeneratorPass logic will not add the reset logic.
    logger.info("test_simple_repeat")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1to2, ["data"])
    branch1 = data1.repeat(2)
    branch1 = branch1.skip(1)    # Skip the first row

    output = np.array([0])
    for item in branch1.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = np.append(output, item["data"])

    golden = np.array([0, 2, 1, 2])

    np.testing.assert_array_equal(output, golden)

def test_generator_reset_1():
    """
    Test (Generator -> Repeat) + (Generator -> Repeat) + (Generator -> Repeat)
    """
    logger.info("test_generator_reset_1")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1to2, ["data"])
    branch1 = data1.repeat(4)
    data2 = ds.GeneratorDataset(generator_10to12, ["data"])
    branch2 = data2.repeat(2)
    branch2 = branch2.take(10)   # Meaningless operation, just want to insert an op in between
    data3 = ds.GeneratorDataset(generator_22to24, ["data"])
    branch3 = data3.repeat(3)
    branch3 = branch3.skip(1)    # Skip the first row

    concat1 = branch1 + branch2
    concat2 = concat1 + branch3

    output = np.array([0])
    for item in concat2.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = np.append(output, item["data"])

    golden = np.array([0, 1, 2, 1, 2, 1, 2, 1, 2, 10, 11, 12, 10, 11, 12, 23, 24, 22, 23, 24, 22, 23, 24])

    np.testing.assert_array_equal(output, golden)

def test_generator_reset_2():
    """
    Test ((Generator -> Repeat) + (Generator -> Repeat) -> Repeat) + (Generator)
    """
    logger.info("test_generator_reset_2")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1to2, ["data"])
    data1 = data1.skip(1)
    branch1 = data1.repeat(3)
    data2 = ds.GeneratorDataset(generator_10to12, ["data"])
    branch2 = data2.repeat(2)
    branch2 = branch2.take(10)   # Meaningless operation, just want to insert an op in between
    data3 = ds.GeneratorDataset(generator_22to24, ["data"])
    branch3 = data3.skip(2)    # Skip the first row

    concat1 = branch1 + branch2
    concat2 = concat1.repeat(2).take(11) + branch3

    output = np.array([0])
    for item in concat2.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = np.append(output, item["data"])

    golden = np.array([0, 2, 2, 2, 10, 11, 12, 10, 11, 12, 2, 2, 24])

    np.testing.assert_array_equal(output, golden)

def test_generator_reset_3():
    """
    Test (Generator -> Repeat -> Repeat) + ((Generator -> Repeat) + (Generator)) -> Repeat) -> EpochCtrl
    """
    logger.info("test_generator_reset_3")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1to2, ["data"])
    branch1 = data1.repeat(2)
    branch1 = branch1.skip(1)
    branch1 = branch1.take(2)
    branch1 = branch1.repeat(2)
    data2 = ds.GeneratorDataset(generator_10to12, ["data"])
    branch2 = data2.repeat(2)
    data3 = ds.GeneratorDataset(generator_22to24, ["data"])
    branch3 = data3.take(2)
    branch3 = branch3

    concat1 = branch2 + branch3
    concat2 = branch1 + concat1.repeat(3).skip(5).take(15)

    itr = concat2.create_dict_iterator(output_numpy=True)

    num_epochs = 5
    output = np.array([0])
    golden = np.array([0])
    expected = np.array([2, 1, 2, 1, 12, 22, 23, 10, 11, 12, 10, 11, 12, 22, 23, 10, 11, 12, 10])
    for _ in range(num_epochs):
        golden = np.append(golden, expected)
        for item in itr:
            output = np.append(output, item["data"])

    np.testing.assert_array_equal(output, golden)

    itr.stop()

def test_generator_reset_4():
    """
    Test Generator -> Repeat -> Repeat
    """
    logger.info("test_generator_reset_4")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1to2, ["data"])
    branch1 = data1.repeat(4).repeat(2)

    output = np.array([0])
    for item in branch1.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = np.append(output, item["data"])

    golden = np.array([0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])

    np.testing.assert_array_equal(output, golden)

def test_generator_reset_5():
    """
    Test Generator -> Repeat -> Repeat -> EpochCtrl
    """
    logger.info("test_generator_reset_5")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1to2, ["data"])
    branch1 = data1.repeat(3).take(3).repeat(2)

    num_epochs = 2
    output = np.array([0])
    itr = branch1.create_dict_iterator(output_numpy=True)

    for _ in range(num_epochs):
        for item in itr:
            output = np.append(output, item["data"])

    golden = np.array([0, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1])

    np.testing.assert_array_equal(output, golden)

    itr.stop()

def test_generator_reset_6():
    """
    Test Generator -> Repeat -> Repeat -> EpochCtrl
    """
    logger.info("test_generator_reset_6")
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_10to12, ["data"])
    branch1 = data1.repeat(2).take(5).repeat(2).skip(2)
    iter1 = branch1.create_dict_iterator(num_epochs=3, output_numpy=True)

    output = np.array([0])
    for _ in range(2):
        for item in iter1:
            output = np.append(output, item["data"])

    golden = np.array([0, 12, 10, 11, 10, 11, 12, 10, 11, 12, 10, 11, 10, 11, 12, 10, 11])

    np.testing.assert_array_equal(output, golden)

    # intentionally not adding itr.stop() to trigger the self-termination when itr is out of scope


if __name__ == '__main__':
    test_generator_reset_1()
    test_generator_reset_2()
    test_generator_reset_3()
    test_generator_reset_4()
    test_generator_reset_5()
    test_generator_reset_6()
    logger.info('\n')
