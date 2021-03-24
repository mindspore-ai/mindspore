# Copyright 2020 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import composite as C


class NetBatchDot(nn.Cell):
    def __init__(self, axes):
        super(NetBatchDot, self).__init__()
        self.axes = axes

    def construct(self, x, y):
        return C.batch_dot(x, y, self.axes)


# Implementation with numpy in tensorflow
def _reference_batch_dot(x, y, axes):
    if isinstance(axes, int):
        axes = [axes, axes]
    elif isinstance(axes, tuple):
        axes = list(axes)
    if axes is None:
        if y.ndim == 2:
            axes = [x.ndim - 1, y.ndim - 1]
        else:
            axes = [x.ndim - 1, y.ndim - 2]
    if axes[0] < 0:
        axes[0] += x.ndim
    if axes[1] < 0:
        axes[1] += y.ndim
    result = []
    axes = [axes[0] - 1, axes[1] - 1]
    for xi, yi in zip(x, y):
        result.append(np.tensordot(xi, yi, axes))
    result = np.array(result)
    if result.ndim == 1:
        result = np.expand_dims(result, -1)
    return result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_dot_fp32():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    np.random.seed(12876)

    # case 1
    shape_x1 = (3, 12, 5, 2, 3)
    shape_x2 = (3, 1, 7, 3, 2)
    axes = (-1, -2)
    x1 = np.ones(shape=shape_x1).astype(np.float32)
    x2 = np.ones(shape=shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 2
    shape_x1 = (4, 3, 7, 5)
    shape_x2 = (4, 1, 7, 1)
    axes = 2
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 3
    shape_x1 = (18, 3, 5, 7)
    shape_x2 = (18, 1, 3, 7)
    axes = -1
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 4
    shape_x1 = (2, 11, 3, 9)
    shape_x2 = (2, 7, 9, 3)
    axes = None
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 5
    shape_x1 = (7, 5)
    shape_x2 = (7, 5)
    axes = None
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 6
    shape_x1 = (7, 3, 5)
    shape_x2 = (7, 5)
    axes = None
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 7
    shape_x1 = (7, 5)
    shape_x2 = (7, 5, 3)
    axes = None
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 8
    shape_x1 = (39, 6)
    shape_x2 = (39, 6)
    axes = -1
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)

    assert np.allclose(ms_result_np, tf_result)

    # case 9
    shape_x1 = (21, 2, 3)
    shape_x2 = (21, 3, 2)
    axes = (-1, -2)
    x1 = np.ones(shape=shape_x1).astype(np.float32)
    x2 = np.ones(shape=shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)
    assert np.allclose(ms_result_np, tf_result)

    # case 10
    shape_x1 = (4, 3, 2, 1, 7, 5)
    shape_x2 = (4, 5, 7, 1)
    axes = -2
    x1 = np.ones(shape=shape_x1).astype(np.float32)
    x2 = np.ones(shape=shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetBatchDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    tf_result = _reference_batch_dot(x1, x2, axes)
    assert np.allclose(ms_result_np, tf_result)
