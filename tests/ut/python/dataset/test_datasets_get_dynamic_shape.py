# Copyright 2021-2022 Huawei Technologies Co., Ltd
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


def test_output_shapes_0():
    """
    Feature: Test output_shapes
    Description: Test output_shapes with data of generator0
    Expectation: The dataset is processed as expected
    """
    logger.info("Test output_shapes with data of generator0.")

    dataset = ds.GeneratorDataset(generator0, ["data1", "data2", "data3"])

    estimate_dynamic_shapes = dataset.output_shapes(estimate=True)
    dynamic_shapes = dataset.output_shapes()

    # check result
    np.testing.assert_array_equal(dynamic_shapes, [[32, 50], [16, 50, 50, 3], [50]])
    np.testing.assert_array_equal(estimate_dynamic_shapes, [[32, None], [16, None, None, 3], [None]])


def generator1():
    for i in range(1, 100):
        yield (np.ones((16, i, 83)), np.array((i)))


def test_output_shapes_1():
    """
    Feature: Test output_shapes
    Description: Test output_shapes with data of generator1
    Expectation: The dataset is processed as expected
    """
    logger.info("Test output_shapes with data of generator1.")

    dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])

    estimate_dynamic_shapes = dataset.output_shapes(estimate=True)
    dynamic_shapes = dataset.output_shapes()

    # check result
    # raise a warning to tell user "data2" is not dynamic
    np.testing.assert_array_equal(dynamic_shapes, [[16, 1, 83], []])
    np.testing.assert_array_equal(estimate_dynamic_shapes, [[16, None, 83], []])


def generator2():
    for i in range(80, 100):
        yield (np.ones((16, i, 83)), np.ones((5, 5)))


def test_output_shapes_2():
    """
    Feature: Test output_shapes
    Description: Test output_shapes with data of generator2
    Expectation: The dataset is processed as expected
    """
    logger.info("Test output_shapes with data of generator2.")

    dataset = ds.GeneratorDataset(generator2, ["data1", "data2"])

    # new api
    estimate_dynamic_shapes = dataset.output_shapes(estimate=True)

    # old api
    dynamic_shapes = dataset.output_shapes()

    # check result
    # column with fixed shape will also be appended to shapes list
    np.testing.assert_array_equal(dynamic_shapes, [[16, 80, 83], [5, 5]])
    np.testing.assert_array_equal(estimate_dynamic_shapes, [[16, None, 83], [5, 5]])


def test_output_shapes_3():
    """
    Feature: Test output_shapes
    Description: Test output_shapes with NumpySlicesDataset
    Expectation: The dataset is processed as expected
    """
    logger.info("Test output_shapes with NumpySlicesDataset.")

    np_data = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
        [[13, 14], [15, 16]]
    ]

    dataset = ds.NumpySlicesDataset(np_data, column_names=["col1"])

    # new api
    estimate_dynamic_shapes = dataset.output_shapes(estimate=True)

    # old api
    dynamic_shapes = dataset.output_shapes()

    # check result
    # column with fixed shape will also be appended to shapes list
    np.testing.assert_array_equal(dynamic_shapes, [[2, 2]])
    np.testing.assert_array_equal(estimate_dynamic_shapes, [[2, 2]])


class Generator3:
    def __init__(self):
        self.data = [np.array([[1], [2]]), np.array([1, 2])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return 2


def test_output_shapes_exception():
    """
    Feature: output_shapes with new parameter `estimate`
    Description: When the shapes data row are inconsistent, raise error.
    Expectation: Raise runtime error to tell user inconsistent shapes.
    """
    logger.info("Test dynamic_min_max_shapes with inconsistent shape.")

    with pytest.raises(RuntimeError) as info:
        dataset = ds.GeneratorDataset(Generator3(), ["data1"])
        _ = dataset.output_shapes(estimate=True)
    assert "Inconsistent shapes, expect same shape for each data row" in str(info.value)

    with pytest.raises(TypeError) as info:
        dataset = ds.GeneratorDataset(Generator3(), ["data1"])
        _ = dataset.output_shapes(estimate=1)


if __name__ == "__main__":
    test_get_dynamic_min_max_shapes_0()
    test_get_dynamic_min_max_shapes_1()
    test_get_dynamic_min_max_shapes_2()
    test_get_dynamic_min_max_shapes_3()
    test_output_shapes_exception()
