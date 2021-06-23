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
import pytest
import mindspore.dataset as ds
from mindspore import log as logger


def generator0():
    for i in range(50, 70):
        yield (np.ones((32, i)), np.zeros((16, i, i, 3)), np.ones((i)))


def test_get_dynamic_min_max_shapes_0():
    logger.info("Test dynamic_min_max_shapes with dynamic shape columns.")

    dataset = ds.GeneratorDataset(generator0, ["data1", "data2", "data3"])

    # config dynamic shape
    dataset.set_dynamic_columns(columns={"data1": [32, None], "data2": [16, None, None, 3], "data3": [None]})

    # get dynamic information
    min_shapes, max_shapes = dataset.dynamic_min_max_shapes()
    dynamic_shapes = dataset.output_shapes()

    # check result
    np.testing.assert_array_equal(min_shapes, [[32, 1], [16, 1, 1, 3], [1]])
    np.testing.assert_array_equal(max_shapes, [[32, 69], [16, 69, 69, 3], [69]])
    np.testing.assert_array_equal(dynamic_shapes, [[32, -1], [16, -1, -1, 3], [-1]])


def generator1():
    for i in range(1, 100):
        yield (np.ones((16, i, 83)), np.array((i)))


def test_get_dynamic_min_max_shapes_1():
    logger.info("Test dynamic_min_max_shapes with dynamic shape column and fix shape column.")

    dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])

    # config dynamic shape
    dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": []})

    # get dynamic information
    dynamic_shapes = dataset.output_shapes()
    min_shapes, max_shapes = dataset.dynamic_min_max_shapes()

    # check result
    # raise a warning to tell user "data2" is not dynamic
    np.testing.assert_array_equal(min_shapes, [[16, 1, 83], []])
    np.testing.assert_array_equal(max_shapes, [[16, 99, 83], []])
    np.testing.assert_array_equal(dynamic_shapes, [[16, -1, 83], []])


def test_get_dynamic_min_max_shapes_2():
    logger.info("Test dynamic_min_max_shapes with setting all columns to dynamic.")

    dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])

    # config all dims have dynamic shape
    dataset.set_dynamic_columns(columns={"data1": [None, None, None]})
    dynamic_shapes = dataset.output_shapes()
    min_shapes, max_shapes = dataset.dynamic_min_max_shapes()

    # check result
    # Although shape[0] of data1 is fix in given data, user think it is dynamic, so shape[0] is dynamic
    np.testing.assert_array_equal(min_shapes, [[1, 1, 1], []])
    np.testing.assert_array_equal(max_shapes, [[16, 99, 83], []])
    np.testing.assert_array_equal(dynamic_shapes, [[-1, -1, -1], []])


def generator2():
    for i in range(80, 100):
        yield (np.ones((16, i, 83)), np.ones((5, 5)))


def test_get_dynamic_min_max_shapes_3():
    logger.info("Test dynamic_min_max_shapes only config dynamic column.")

    dataset = ds.GeneratorDataset(generator2, ["data1", "data2"])

    # only dynamic shape is required to config
    dataset.set_dynamic_columns(columns={"data1": [16, None, 83]})

    # get dynamic information
    dynamic_shapes = dataset.output_shapes()
    min_shapes, max_shapes = dataset.dynamic_min_max_shapes()

    # check result
    # column with fixed shape will also be appended to shapes list
    np.testing.assert_array_equal(min_shapes, [[16, 1, 83], [5, 5]])
    np.testing.assert_array_equal(max_shapes, [[16, 99, 83], [5, 5]])
    np.testing.assert_array_equal(dynamic_shapes, [[16, -1, 83], [5, 5]])


def test_get_dynamic_min_max_shapes_4():
    logger.info("Test dynamic_min_max_shapes with dynamic setting for column with fixed shape.")

    dataset = ds.GeneratorDataset(generator2, ["data1", "data2"])

    # only dynamic shape is required to config
    dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": [None, 5]})

    # get dynamic information
    dynamic_shapes = dataset.output_shapes()
    min_shapes, max_shapes = dataset.dynamic_min_max_shapes()

    # check result
    # column with fixed shape will also be appended to shapes list
    np.testing.assert_array_equal(min_shapes, [[16, 1, 83], [1, 5]])
    np.testing.assert_array_equal(max_shapes, [[16, 99, 83], [5, 5]])
    np.testing.assert_array_equal(dynamic_shapes, [[16, -1, 83], [-1, 5]])


def test_get_dynamic_min_max_shapes_5():
    logger.info("Test dynamic_min_max_shapes with NumpySlicesDataset.")

    np_data = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
        [[13, 14], [15, 16]]
    ]

    dataset = ds.NumpySlicesDataset(np_data, column_names=["col1"])
    dataset.set_dynamic_columns(columns={"col1": [2, None]})

    # get dynamic information
    dynamic_shapes = dataset.output_shapes()
    min_shapes, max_shapes = dataset.dynamic_min_max_shapes()

    # check result
    # column with fixed shape will also be appended to shapes list
    np.testing.assert_array_equal(min_shapes, [[2, 1]])
    np.testing.assert_array_equal(max_shapes, [[2, 2]])
    np.testing.assert_array_equal(dynamic_shapes, [[2, -1]])


def test_get_dynamic_min_max_shapes_6():
    logger.info("Test dynamic_min_max_shapes with unexpected column setting.")

    dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])

    with pytest.raises(TypeError) as info:
        # dynamic column is not in dict
        dataset.set_dynamic_columns(columns=list())
    assert "Pass a dict to set dynamic shape" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        # dynamic column is not set
        dataset.set_dynamic_columns(columns=dict())
        dataset.dynamic_min_max_shapes()
    assert "dynamic_columns is not set, call set_dynamic_columns()" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        # dynamic column is not set
        dataset.set_dynamic_columns(columns={"data2": []})
        dataset.dynamic_min_max_shapes()
    assert "column [data1] has dynamic shape but not set by set_dynamic_columns()" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        # column does not exist
        dataset.set_dynamic_columns(columns={"data3": [16, None, 83]})
        dataset.dynamic_min_max_shapes()
    assert "dynamic column [data3] does not match any column in dataset" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        # unexpected column shape
        dataset.set_dynamic_columns(columns={"data1": [16, 83, None]})
        dataset.dynamic_min_max_shapes()
    assert "shape [16, 83, None] does not match dataset column [data1] with shape [16, 1, 83]" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        # unexpected column shape
        dataset.set_dynamic_columns(columns={"data1": [16, None]})
        dataset.dynamic_min_max_shapes()
    assert "shape [16, None] does not match dataset column [data1] with shape [16, 1, 83]" in str(info.value)


if __name__ == "__main__":
    test_get_dynamic_min_max_shapes_0()
    test_get_dynamic_min_max_shapes_1()
    test_get_dynamic_min_max_shapes_2()
    test_get_dynamic_min_max_shapes_3()
    test_get_dynamic_min_max_shapes_4()
    test_get_dynamic_min_max_shapes_5()
    test_get_dynamic_min_max_shapes_6()
    