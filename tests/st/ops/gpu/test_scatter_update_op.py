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

class TestScatterUpdateNet(nn.Cell):
    def __init__(self, inputx, indices, updates):
        super(TestScatterUpdateNet, self).__init__()
        self.scatter_update = P.ScatterUpdate()
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_update(self.inputx, self.indices, self.updates)
        return out

def scatter_update_net(inputx, indices, updates):
    net = TestScatterUpdateNet(inputx, indices, updates)
    return net()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_small_float32():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.arange(6).reshape((2, 3)).astype(np.float32))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[0., 1., 2.],
                         [3., 4., 5.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_input_less_than_1_float32():
    inputx = Tensor(np.array([[0.214141, 0.415151, 0.51516],
                              [0.876542, 0.451611, 0.55112],
                              [0.111244, 0.633333, 0.34444]]).astype(np.float32))
    indices = Tensor(np.array([1, 0, 2]).astype(np.int32))
    updates = Tensor(np.arange(34, 43).reshape((3, 3)).astype(np.float32))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[37., 38., 39.],
                         [34., 35., 36.],
                         [40., 41., 42.]], dtype=np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_float16():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float16))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.arange(6).reshape((2, 3)).astype(np.float16))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[0., 1., 2.],
                         [3., 4., 5.]]).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_int32():
    inputx = Tensor(np.zeros((2, 3)).astype(np.int32))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.arange(6).reshape((2, 3)).astype(np.int32))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[0., 1., 2.],
                         [3., 4., 5.]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_large_float16():
    inputx = Tensor(np.zeros((4, 3)).astype(np.float16))
    indices = Tensor(np.array([[2, 1], [0, 3]]).astype(np.int32))
    updates = Tensor(np.arange(63, 75).reshape((2, 2, 3)).astype(np.float16))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[69., 70., 71.],
                         [66., 67., 68.],
                         [63., 64., 65.],
                         [72., 73., 74.]]).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_disordered_float16():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.float16)))
    indices = Tensor(np.array([1, 2]).astype(np.int32))
    updates = Tensor(np.arange(63, 71).reshape((2, 4)).astype(np.float16))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[45., 44., 43., 42.],
                         [63., 64., 65., 66.],
                         [67., 68., 69., 70.]]).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_disordered_int32():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32)))
    indices = Tensor(np.array([1, 2]).astype(np.int32))
    updates = Tensor(np.arange(63, 71).reshape((2, 4)).astype(np.int32))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[45., 44., 43., 42.],
                         [63., 64., 65., 66.],
                         [67., 68., 69., 70.]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_update_large_shape_float16():
    inputx = Tensor(np.arange(96).reshape((4, 2, 3, 4)).astype(np.float16))
    indices = Tensor(np.array([1, 0]).astype(np.int32))
    updates = Tensor(np.flip(np.arange(48).reshape((2, 2, 3, 4)).astype(np.float16)))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[[[23., 22., 21., 20.],
                           [19., 18., 17., 16.],
                           [15., 14., 13., 12.]],
                          [[11., 10., 9., 8.],
                           [7., 6., 5., 4.],
                           [3., 2., 1., 0.]]],
                         [[[47., 46., 45., 44.],
                           [43., 42., 41., 40.],
                           [39., 38., 37., 36.]],
                          [[35., 34., 33., 32.],
                           [31., 30., 29., 28.],
                           [27., 26., 25., 24.]]],
                         [[[48., 49., 50., 51.],
                           [52., 53., 54., 55.],
                           [56., 57., 58., 59.]],
                          [[60., 61., 62., 63.],
                           [64., 65., 66., 67.],
                           [68., 69., 70., 71.]]],
                         [[[72., 73., 74., 75.],
                           [76., 77., 78., 79.],
                           [80., 81., 82., 83.]],
                          [[84., 85., 86., 87.],
                           [88., 89., 90., 91.],
                           [92., 93., 94., 95.]]]]).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
