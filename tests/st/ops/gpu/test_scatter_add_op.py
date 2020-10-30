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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# all cases tested against dchip

class TestScatterAddNet(nn.Cell):
    def __init__(self, inputx, indices, updates):
        super(TestScatterAddNet, self).__init__()
        self.scatter_add = P.ScatterAdd()
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_add(self.inputx, self.indices, self.updates)
        return out

def scatter_add_net(inputx, indices, updates):
    net = TestScatterAddNet(inputx, indices, updates)
    return net()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_small_float32():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[6., 8., 10.],
                         [12., 14., 16.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_input_less_than_1_float32():
    inputx = Tensor(np.array([[0.214141, 0.415151, 0.51516],
                              [0.876542, 0.451611, 0.55112],
                              [0.111244, 0.633333, 0.34444]]).astype(np.float32))
    indices = Tensor(np.array([[[1, 0, 2],
                                [2, 2, 0]],
                               [[1, 0, 1],
                                [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(np.float32))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[141.21414, 144.41515, 147.51517],
                         [208.87654, 212.45161, 216.55112],
                         [257.11124, 262.63333, 267.34442]], dtype=np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_float16():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float16))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float16))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[6., 8., 10.],
                         [12., 14., 16.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_large_float16():
    inputx = Tensor(np.zeros((2, 3, 4)).astype(np.float16))
    indices = Tensor(np.array([[0, 0], [1, 1]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.float16))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[[138., 140., 142., 144.],
                          [146., 148., 150., 152.],
                          [154., 156., 158., 160.]],
                         [[186., 188., 190., 192.],
                          [194., 196., 198., 200.],
                          [202., 204., 206., 208.]]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_disordered_float16():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.float16)))
    indices = Tensor(np.array([[[0, 1, 2],
                                [2, 1, 0]],
                               [[0, 0, 0],
                                [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.float16))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[464., 468., 472., 476.],
                         [187., 188., 189., 190.],
                         [492., 496., 500., 504.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
