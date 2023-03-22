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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype


class BoolTensorIndexGetItem(nn.Cell):
    def construct(self, x, index):
        return x[index]


class BoolTensorIndexSetItem(nn.Cell):
    def construct(self, x, index, value):
        x[index] = value
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_get_item_x_5_index_5(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.array([1, 2, 3, 4, 5])
    index0 = np.array([True, False, True, False, True])
    # Mindspore
    x = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    get_item_net = BoolTensorIndexGetItem()
    result_ms = get_item_net(x, index)
    # numpy
    x = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    result_np = x[index]
    # allclose
    assert np.allclose(result_ms.asnumpy(), result_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_get_item_x_2x1x448x448_index_2x1x448x448(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.random.randn(2, 1, 448, 448)
    index0 = np.random.randn(2, 1, 448, 448)
    # Mindspore
    x = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    get_item_net = BoolTensorIndexGetItem()
    result_ms = get_item_net(x, index)
    # numpy
    x = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    result_np = x[index]
    # allclose
    assert np.allclose(result_ms.asnumpy(), result_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_get_item_x_2x1x448x448_index_2x1x448(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.random.randn(2, 1, 448, 448)
    index0 = np.random.randn(2, 1, 448)
    # Mindspore
    x = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    get_item_net = BoolTensorIndexGetItem()
    result_ms = get_item_net(x, index)
    # numpy
    x = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    result_np = x[index]
    # allclose
    assert np.allclose(result_ms.asnumpy(), result_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_5_index_5_value_1(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.array([1, 2, 3, 4, 5])
    index0 = np.array([True, False, True, False, True])
    value = 0
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, value)
    # numpy
    x_np = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    x_np[index] = value
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_3x3_index_3_value_1(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.ones((3, 3))
    index0 = np.array([True, False, True])
    value = -1
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, value)
    # numpy
    x_np = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    x_np[index] = value
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_3x3_index_3_value_2x3(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.ones((3, 3))
    index0 = np.array([True, False, True])
    value = np.array([[-1, -1, -1], [-1, -1, -1]])
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    value_ms = Tensor(value, dtype=mstype.float32)
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, value_ms)
    # numpy
    x_np = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    x_np[index] = value
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_2x1x448x448_index_2x1x448x448_value_401408(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.random.randn(2, 1, 448, 448)
    index0 = np.random.randn(2, 1, 448, 448)
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, -1 * x_ms[index])
    # numpy
    x_np = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    x_np[index] *= -1
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_2x1x448x448_index_2x1x448_value_896x448(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.random.randn(2, 1, 448, 448)
    index0 = np.random.randn(2, 1, 448)
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = Tensor(index0, dtype=mstype.float32).astype(mstype.bool_)
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, -1 * x_ms[index])
    # numpy
    x_np = x0.astype(np.float32)
    index = index0.astype(np.bool_)
    x_np[index] *= -1
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_get_item_x_2x3_index_bool2_int1(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.arange(2*3).reshape(2, 3)
    index0 = np.array([True, False])
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = [Tensor(index0, dtype=mstype.float32).astype(mstype.bool_), 0]
    get_item_net = BoolTensorIndexGetItem()
    y_ms = get_item_net(x_ms, index)
    # numpy
    y_np = np.array([0.])
    # allclose
    assert np.allclose(y_ms.asnumpy(), y_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_2x3_index_bool2_int1_value_1(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.arange(2*3).reshape(2, 3)
    index0 = np.array([True, False])
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = [Tensor(index0, dtype=mstype.float32).astype(mstype.bool_), 0]
    value = -1
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, value)
    # numpy
    x_np = np.array([[-1., 1., 2.], [3., 4., 5.]])
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bool_tensor_index_set_item_x_2x3_index_bool2_int1_value_list_3(mode):
    """
    Feature: tensor indexing with index of bool tensor
    Description: Verify the result of bool tensor indexing
    Expectation: success
    """
    ms.set_context(mode=mode)
    x0 = np.arange(2*3).reshape(2, 3)
    index0 = np.array([True, False])
    # Mindspore
    x_ms = Tensor(x0, dtype=mstype.float32)
    index = [Tensor(index0, dtype=mstype.float32).astype(mstype.bool_), 0]
    value = [-1]
    set_item_net = BoolTensorIndexSetItem()
    x_ms = set_item_net(x_ms, index, value)
    # numpy
    x_np = np.array([[-1., 1., 2.], [3., 4., 5.]])
    # allclose
    assert np.allclose(x_ms.asnumpy(), x_np)
