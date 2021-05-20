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
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.scatter_add = P.TensorScatterAdd()

    def construct(self, x, indices, update):
        return self.scatter_add(x, indices, update)


def scatter_net(x, indices, update):
    scatter_add = Net()
    return scatter_add(Tensor(x), Tensor(indices), Tensor(update)).asnumpy()

def numpy_scatter_add(x, indices, update):
    indices = np.expand_dims(indices, -1) if indices.ndim == 1 else indices
    for idx, up in zip(indices, update):
        idx = tuple(idx.tolist())
        x[idx] += up
    return x

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # indices 2-d, each index points to single value
    arr_input = np.arange(21).reshape(3, 7).astype(np.float32)
    arr_indices = np.array([[0, 1], [1, 1], [0, 2], [0, 2], [2, 1]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = numpy_scatter_add(arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    # indices 2-d, each index points to single value
    arr_input = np.arange(24).reshape(4, 2, 3).astype(np.float32)
    arr_indices = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1], [3, 0, 1]]).astype(np.int32)
    arr_update = np.array([-1, -2, -3, -4]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = numpy_scatter_add(arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    # indices 2-d, each index points to a slice, and each value points to a single element in the slice
    arr_input = np.zeros((3, 3)).astype(np.float32)
    arr_indices = np.array([[0], [2], [1]]).astype(np.int32)
    arr_update = np.array([[-1, 4, 3], [-2, 0, 1], [-3, 1, 2]]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = numpy_scatter_add(arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(21).reshape(3, 7).astype(np.float32)
    arr_indices = np.array([[0, 1], [1, 1], [0, 5], [0, 2], [2, 1]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = numpy_scatter_add(arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(24).reshape(4, 2, 3).astype(np.float32)
    arr_indices = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1], [3, 0, 1]]).astype(np.int32)
    arr_update = np.array([-1, -2, -3, -4]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = numpy_scatter_add(arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    # Below are from test_tensor_scatter_update.py
    arr_input = np.arange(25).reshape(5, 5).astype(np.float32)
    arr_indices = np.array([[[0, 0],
                             [1, 1],
                             [2, 2],
                             [3, 3],
                             [4, 4]],
                            [[0, 4],
                             [1, 3],
                             [2, 2],
                             [3, 1],
                             [4, 0]]]).astype(np.int32)
    arr_update = np.array([[11, 22, 33, 44, 55], [66, 77, 33, 99, 100]]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[11, 1, 2, 3, 70],
                         [5, 28, 7, 85, 9],
                         [10, 11, 78, 13, 14],
                         [15, 115, 17, 62, 19],
                         [120, 21, 22, 23, 79]]).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(25).reshape(5, 5).astype(np.float64)
    arr_indices = np.array([[[0, 0],
                             [1, 1],
                             [2, 2],
                             [3, 3],
                             [4, 4]],
                            [[0, 4],
                             [1, 3],
                             [2, 2],
                             [3, 1],
                             [4, 0]]]).astype(np.int64)
    arr_update = np.array([[11, 22, 33, 44, 55], [66, 77, 33, 99, 100]]).astype(np.float64)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[11, 1, 2, 3, 70],
                         [5, 28, 7, 85, 9],
                         [10, 11, 78, 13, 14],
                         [15, 115, 17, 62, 19],
                         [120, 21, 22, 23, 79]]).astype(np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(25).reshape(5, 5).astype(np.int32)
    arr_indices = np.array([[[0, 0],
                             [1, 1],
                             [2, 2],
                             [3, 3],
                             [4, 4]],
                            [[0, 4],
                             [1, 3],
                             [2, 2],
                             [3, 1],
                             [4, 0]]]).astype(np.int32)
    arr_update = np.array([[11, 22, 33, 44, 55], [66, 77, 33, 99, 100]]).astype(np.int32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[11, 1, 2, 3, 70],
                         [5, 28, 7, 85, 9],
                         [10, 11, 78, 13, 14],
                         [15, 115, 17, 62, 19],
                         [120, 21, 22, 23, 79]]).astype(np.int32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)
