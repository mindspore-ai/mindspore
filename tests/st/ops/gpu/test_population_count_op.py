# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class PopulationCount(Cell):
    def __init__(self):
        super().__init__()
        self.populationcount = P.PopulationCount()

    def construct(self, x):
        return self.populationcount(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_1x2_int8():
    """
    Feature: PopulationCount
    Description: Test int8 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[1, 2]]).astype(np.int8)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[1, 1]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_2x2x2_uint8():
    """
    Feature: PopulationCount
    Description: Test uint8 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[[1, 2], [3, 4]],
                        [[1, 2], [3, 4]]]).astype(np.uint8)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[[1, 1], [2, 1]],
                              [[1, 1], [2, 1]]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_1x2_int16():
    """
    Feature: PopulationCount
    Description: Test int16 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[1, 2]]).astype(np.int16)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[1, 1]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_2x2x2_uint16():
    """
    Feature: PopulationCount
    Description: Test uint16 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[[1, 2], [3, 4]],
                        [[1, 2], [3, 4]]]).astype(np.uint16)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[[1, 1], [2, 1]],
                              [[1, 1], [2, 1]]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_1x2_int32():
    """
    Feature: PopulationCount
    Description: Test int32 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[1, 2]]).astype(np.int32)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[1, 1]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_2x2x2_uint32():
    """
    Feature: PopulationCount
    Description: Test uint32 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[[1, 2], [3, 4]],
                        [[1, 2], [3, 4]]]).astype(np.uint32)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[[1, 1], [2, 1]],
                              [[1, 1], [2, 1]]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_1x2_int64():
    """
    Feature: PopulationCount
    Description: Test int64 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[1, 2]]).astype(np.int64)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[1, 1]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_populationcount_2x2x2_uint64():
    """
    Feature: PopulationCount
    Description: Test uint64 of input
    Expectation: The results are as expected
    """
    input_x = np.array([[[1, 2], [3, 4]],
                        [[1, 2], [3, 4]]]).astype(np.uint64)
    net = PopulationCount()
    output_ms = net(Tensor(input_x))
    expect_output = np.array([[[1, 1], [2, 1]],
                              [[1, 1], [2, 1]]]).astype(np.uint8)
    assert np.allclose(output_ms.asnumpy(), expect_output)
