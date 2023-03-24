# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops.operations import Tile


class TileNet(Cell):
    def __init__(self, numpy_input):
        super(TileNet, self).__init__()
        self.Tile = Tile()

        self.input_parameter = Parameter(initializer(Tensor(numpy_input), numpy_input.shape), name='x')

    @jit
    def construct(self, mul):
        return self.Tile(self.input_parameter, mul)


def ms_tile(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_0 = np.arange(2).reshape((2, 1, 1)).astype(nptype)
    mul_0 = (8, 1, 1)
    input_1 = np.arange(32).reshape((2, 4, 4)).astype(nptype)
    mul_1 = (2, 2, 2)
    input_2 = np.arange(1).reshape((1, 1, 1)).astype(nptype)
    mul_2 = (1, 1, 1)

    tile_net = TileNet(input_0)
    np_expected = np.tile(input_0, mul_0)
    ms_output = tile_net(mul_0).asnumpy()
    np.testing.assert_array_equal(ms_output, np_expected)

    tile_net = TileNet(input_1)
    np_expected = np.tile(input_1, mul_1)
    ms_output = tile_net(mul_1).asnumpy()
    np.testing.assert_array_equal(ms_output, np_expected)

    tile_net = TileNet(input_2)
    np_expected = np.tile(input_2, mul_2)
    ms_output = tile_net(mul_2).asnumpy()
    np.testing.assert_array_equal(ms_output, np_expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_float16():
    ms_tile(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_float32():
    ms_tile(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_float64():
    ms_tile(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_int16():
    ms_tile(np.int16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_int32():
    ms_tile(np.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_int64():
    ms_tile(np.int64)
