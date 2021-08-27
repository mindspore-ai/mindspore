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
# ============================================================================

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P

def reshape(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    reshape_op = P.Reshape()
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(nptype)
    input_tensor = Tensor(np.array(data))

    new_shape = (2, 6)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (6, 2)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (3, 4)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (4, 3)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (1, 12)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (12, 1)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

def reshape_bool():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    reshape_op = P.Reshape()
    data = np.array([True, True, False, True, False, False, True, False, False, False, False, False])
    input_tensor = Tensor(np.array(data))

    new_shape = (2, 6)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (6, 2)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (3, 4)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (4, 3)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (1, 12)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)

    new_shape = (12, 1)
    output_tensor = reshape_op(input_tensor, new_shape)
    assert new_shape == output_tensor.shape
    np.testing.assert_array_equal(output_tensor.asnumpy().flatten(), data)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape_float():
    reshape(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape_float16():
    reshape(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape_int32():
    reshape(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape_uint8():
    reshape(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape_bool():
    reshape_bool()
