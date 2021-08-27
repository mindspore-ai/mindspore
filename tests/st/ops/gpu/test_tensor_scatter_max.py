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
        self.scatter = P.TensorScatterMax()

    def construct(self, x, indices, update):
        return self.scatter(x, indices, update)


def scatter_net(x, indices, update):
    return Net()(Tensor(x), Tensor(indices), Tensor(update)).asnumpy()

def scatter(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    arr_input = np.arange(21).reshape(3, 7).astype(nptype)
    arr_indices = np.array([[0, 1], [1, 1], [0, 5], [0, 2], [2, 1]]).astype(np.int32)
    arr_update = np.array([3.2, -1.1, -4.3, 2.2, -1.0]).astype(nptype)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[0, 3.2, 2.2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12, 13],
                         [14, 15, 16, 17, 18, 19, 20]]).astype(nptype)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(21).reshape(3, 7).astype(nptype)
    arr_indices = np.array([[0, 1], [2, 3], [2, 3], [0, 2], [2, 1], [2, 3], [2, 3], [2, 3]]).astype(np.int32)
    arr_update = np.array([-3.2, -1.1, 63.3, 2.2, 1.0, 2, -61, 66.3]).astype(nptype)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[0, 1, 2.2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12, 13],
                         [14, 15, 16, 66.3, 18, 19, 20]]).astype(nptype)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

def scatter_unsigned(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    arr_input = np.arange(21).reshape(3, 7).astype(nptype)
    arr_indices = np.array([[0, 1], [1, 1], [0, 5], [0, 2], [2, 1]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 5.3, 2.2, 1.0]).astype(nptype)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[0, 3.2, 2.2, 3, 4, 5.3, 6],
                         [7, 8, 9, 10, 11, 12, 13],
                         [14, 15, 16, 17, 18, 19, 20]]).astype(nptype)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(21).reshape(3, 7).astype(nptype)
    arr_indices = np.array([[0, 1], [2, 3], [2, 3], [0, 2], [2, 1], [2, 3], [2, 3], [2, 3]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 63.3, 2.2, 1.0, 2, 61, 66.3]).astype(nptype)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[0, 3.2, 2.2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12, 13],
                         [14, 15, 16, 66.3, 18, 19, 20]]).astype(nptype)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_float16():
    scatter(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_float32():
    scatter(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_uint8():
    scatter_unsigned(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_int8():
    scatter(np.int8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_int32():
    scatter(np.int32)
