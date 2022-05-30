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
import pytest
import mindspore.dataset as ds
from mindspore import log as logger


# In generator dataset: Number of rows is 3, its value is 0, 1, 2
def generator():
    for i in range(3):
        yield (np.array([i]),)


# In generator dataset: Number of rows is 10, its value is 0, 1, 2 ... 10
def generator_10():
    for i in range(10):
        yield (np.array([i]),)


def filter_func_ge(data):
    if data > 3:
        return False
    return True


def test_take_01():
    """
    Feature: Take op
    Description: Test take op where originally there are 3 rows and take 1 row. In this case, will not meet EOE and EOF
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_01")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(1)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for _, d in enumerate(data1):
        assert d[0].asnumpy()[0] == 0

    assert sum([1 for _ in data1]) == 2


def test_take_02():
    """
    Feature: Take op
    Description: Test take op where originally there are 3 rows and take 2 rows. In this case, will meet EOE
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_02")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(2)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i % 2 == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 4


def test_take_03():
    """
    Feature: Take op
    Description: Test take op where originally there are 3 rows and take 3 rows. In this case, will meet EOE and EOF
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_03")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(3)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data elements
    for i, d in enumerate(data1):
        assert i % 3 == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 6


def test_take_04():
    """
    Feature: Take op
    Description: Test take op where originally there are 3 rows and take 4 rows (more than the total rows)
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_04")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(4)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i % 3 == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 6


def test_take_05():
    """
    Feature: Take op
    Description: Test take op where there is no repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_05")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 2


def test_take_06():
    """
    Feature: Take op
    Description: Test take op where repeat op is done before take op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_06")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.repeat(2)
    data1 = data1.take(4)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i % 3 == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 4


def test_take_07():
    """
    Feature: Take op
    Description: Test take op where take op is before batch op and have take(N) where N refers to rows num
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_07")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(2)
    data1 = data1.batch(2)
    assert sum([1 for _ in data1]) == 1


def test_take_08():
    """
    Feature: Take op
    Description: Test take op where take op is after batch op and have take(N) where N refers to batches num
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_08")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.batch(2)
    data1 = data1.take(2)
    assert sum([1 for _ in data1]) == 2


def test_take_09():
    """
    Feature: Take op
    Description: Test take op where take count is -1 and read the the whole dataset, take op is after repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_09")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.repeat(2)
    data1 = data1.take(-1)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i % 3 == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 6


def test_take_10():
    """
    Feature: Take op
    Description: Test take op where take count is -1 and read the the whole dataset, take op is before repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_10")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(-1)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i % 3 == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 6


def test_take_11():
    """
    Feature: Take op
    Description: Test take op where batch op is first, followed by repeat op, then take op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_11")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.batch(2)
    data1 = data1.repeat(2)
    data1 = data1.take(-1)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert 2 * (i % 2) == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 4


def test_take_12():
    """
    Feature: Take op
    Description: Test take op where take op is first, followed by batch op, then repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_12")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(2)
    data1 = data1.batch(2)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for _, d in enumerate(data1):
        assert d[0].asnumpy()[0] == 0

    assert sum([1 for _ in data1]) == 2


def test_take_13():
    """
    Feature: Take op
    Description: Test take op where skip op is first, followed by take op, then batch op, finally repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_13")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.skip(2)
    data1 = data1.take(-1)
    data1 = data1.batch(2)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for _, d in enumerate(data1):
        assert d[0].asnumpy()[0] == 2

    assert sum([1 for _ in data1]) == 2


def test_take_14():
    """
    Feature: Take op
    Description: Test take op where take op is first, followed by batch op, then skip op, finally repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_14")
    data1 = ds.GeneratorDataset(generator, ["data"])

    data1 = data1.take(-1)
    data1 = data1.batch(2)
    data1 = data1.skip(1)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for _, d in enumerate(data1):
        assert d[0].asnumpy()[0] == 2

    assert sum([1 for _ in data1]) == 2


def test_take_15():
    """
    Feature: Take op
    Description: Test take op with large amount of data, first take op then skip op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_15")
    data1 = ds.GeneratorDataset(generator_10, ["data"])

    data1 = data1.take(6)
    data1 = data1.skip(2)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert (i + 2) == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 4


def test_take_16():
    """
    Feature: Take op
    Description: Test take op with large amount of data, first skip op then take op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_16")
    data1 = ds.GeneratorDataset(generator_10, ["data"])

    data1 = data1.skip(3)
    data1 = data1.take(5)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert (i + 3) == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 5


def test_take_17():
    """
    Feature: Take op
    Description: Test take op with take op first then filter op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_17")
    data1 = ds.GeneratorDataset(generator_10, ["data"])

    data1 = data1.take(8)
    data1 = data1.filter(predicate=filter_func_ge, num_parallel_workers=4)

    # Here i refers to index, d refers to data element
    for i, d in enumerate(data1):
        assert i == d[0].asnumpy()[0]

    assert sum([1 for _ in data1]) == 4


def test_take_18():
    """
    Feature: Take op
    Description: Test take op with take op first, then filter op, skip op, batch op, and repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_take_18")
    data1 = ds.GeneratorDataset(generator_10, ["data"])

    data1 = data1.take(8)
    data1 = data1.filter(predicate=filter_func_ge, num_parallel_workers=4)
    data1 = data1.skip(2)

    data1 = data1.batch(2)
    data1 = data1.repeat(2)

    # Here i refers to index, d refers to data element
    for _, d in enumerate(data1):
        assert d[0].asnumpy()[0] == 2

    assert sum([1 for _ in data1]) == 2


def test_take_19():
    """
    Feature: Take op
    Description: Test take op where take op is after batch op, meaning take(N) where N refers to batches num
    Expectation: Error is raised as expected
    """
    logger.info("test_take_19")
    with pytest.raises(ValueError) as info:
        data1 = ds.GeneratorDataset(generator, ["data"])

        data1 = data1.batch(2)
        data1 = data1.take(0)
    assert "within the required interval" in str(info.value)

if __name__ == '__main__':
    test_take_01()
    test_take_02()
    test_take_03()
    test_take_04()
    test_take_05()
    test_take_06()
    test_take_07()
    test_take_08()
    test_take_09()
    test_take_10()
    test_take_11()
    test_take_12()
    test_take_13()
    test_take_14()
    test_take_15()
    test_take_16()
    test_take_17()
    test_take_18()
    test_take_19()
    logger.info('== test take operation finished ==')
