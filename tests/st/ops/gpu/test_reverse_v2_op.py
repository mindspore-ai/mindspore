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
# ============================================================================
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P

class ReverseV2Net(nn.Cell):
    def __init__(self, axis):
        super(ReverseV2Net, self).__init__()
        self.reverse_v2 = P.ReverseV2(axis)

    def construct(self, x):
        return self.reverse_v2(x)


def reverse_v2(x_numpy, axis):
    x = Tensor(x_numpy)
    reverse_v2_net = ReverseV2Net(axis)
    output = reverse_v2_net(x).asnumpy()
    expected_output = np.flip(x_numpy, axis)
    np.testing.assert_array_equal(output, expected_output)

def reverse_v2_3d(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_numpy = np.arange(60).reshape(3, 4, 5).astype(nptype)

    reverse_v2(x_numpy, (0,))
    reverse_v2(x_numpy, (1,))
    reverse_v2(x_numpy, (2,))
    reverse_v2(x_numpy, (2, -2))
    reverse_v2(x_numpy, (-3, 1, 2))

def reverse_v2_1d(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_numpy = np.arange(4).astype(nptype)

    reverse_v2(x_numpy, (0,))
    reverse_v2(x_numpy, (-1,))

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_float16():
    reverse_v2_1d(np.float16)
    reverse_v2_3d(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_float32():
    reverse_v2_1d(np.float32)
    reverse_v2_3d(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_float64():
    """
    Feature: test ReverseV2 with using float64.
    Description: input input_x and axis, test the output value
    Expectation: the result match with expect.
    """
    reverse_v2_1d(np.float64)
    reverse_v2_3d(np.float64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_uint8():
    reverse_v2_1d(np.uint8)
    reverse_v2_3d(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_uint16():
    """
    Feature: test ReverseV2 with using uint16.
    Description: input input_x and axis, test the output value
    Expectation: the result match with expect.
    """
    reverse_v2_1d(np.uint16)
    reverse_v2_3d(np.uint16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_int8():
    """
    Feature: test ReverseV2 with using int8.
    Description: input input_x and axis, test the output value
    Expectation: the result match with expect.
    """
    reverse_v2_1d(np.int8)
    reverse_v2_3d(np.int8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_int16():
    reverse_v2_1d(np.int16)
    reverse_v2_3d(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_int32():
    reverse_v2_1d(np.int32)
    reverse_v2_3d(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_int64():
    reverse_v2_1d(np.int64)
    reverse_v2_3d(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_invalid_axis():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.arange(60).reshape(1, 2, 3, 2, 5).astype(np.int32))

    with pytest.raises(ValueError) as info:
        reverse_v2_net = ReverseV2Net((0, 1, 2, 1))
        _ = reverse_v2_net(x)
    assert "'axis' cannot contain duplicate dimensions" in str(info.value)

    with pytest.raises(ValueError) as info:
        reverse_v2_net = ReverseV2Net((-2, -1, 3))
        _ = reverse_v2_net(x)
    assert "'axis' cannot contain duplicate dimensions" in str(info.value)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_tensor_api():
    """
    Feature: ReverseV2 GPU operation
    Description: input axis, test the output value
    Expectation: the values match the predefined values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.int32)
    output = input_x.reverse(axis=[1])
    expected = np.array([[4, 3, 2, 1], [8, 7, 6, 5]]).astype(np.int32)
    assert np.array_equal(output.asnumpy(), expected)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reverse_v2_functional_api():
    """
    Feature: ReverseV2 GPU operation
    Description: input input_x and axis, test the output value
    Expectation: the values match the predefined values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.int32)
    output = ms.ops.reverse(input_x, axis=[1])
    expected = np.array([[4, 3, 2, 1], [8, 7, 6, 5]]).astype(np.int32)
    assert np.array_equal(output.asnumpy(), expected)
