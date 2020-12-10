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
from mindspore.ops.operations import _inner_ops as inner

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# all cases tested against dchip

class TestScatterAddNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super(TestScatterAddNet, self).__init__()
        self.scatter_add = P.ScatterAdd(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_add(self.inputx, self.indices, self.updates)
        return out

def scatter_add_net(inputx, indices, updates):
    lock = True
    net = TestScatterAddNet(lock, inputx, indices, updates)
    return net()

def scatter_add_use_locking_false_net(inputx, indices, updates):
    lock = False
    net = TestScatterAddNet(lock, inputx, indices, updates)
    return net()

class TestScatterAddDynamicNet(nn.Cell):
    def __init__(self, inputx, indices, updates):
        super(TestScatterAddDynamicNet, self).__init__()
        self.scatter_add = P.ScatterAdd()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        indices = self.test_dynamic(self.indices)
        updates = self.test_dynamic(self.updates)
        out = self.scatter_add(self.inputx, indices, updates)
        return out

def scatter_add_d_net(inputx, indices, updates):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = TestScatterAddDynamicNet(inputx, indices, updates)
    return net()

class TestScatterAddDynamicNet2(nn.Cell):
    def __init__(self, inputx):
        super(TestScatterAddDynamicNet2, self).__init__()
        self.scatter_add = P.ScatterAdd()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        indices = self.test_dynamic(indices)
        updates = self.test_dynamic(updates)
        out = self.scatter_add(self.inputx, indices, updates)
        return out

def scatter_add_d2_net(inputx, indices_1, updates_1,
                       indices_2, updates_2):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = TestScatterAddDynamicNet2(inputx)
    out1 = net(indices_1, updates_1)
    out2 = net(indices_2, updates_2)
    return (out1, out2)

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
def test_scatter_add_input_updated():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterAddNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[6., 8., 10.],
                         [12., 14., 16.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_large_shape_float32():
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.float32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.float32))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[[[1., 2., 3., 4.],
                           [5., 6., 7., 8.],
                           [9., 10., 11., 12.]],
                          [[13., 14., 15., 16.],
                           [17., 18., 19., 20.],
                           [21., 22., 23., 24.]]],
                         [[[73., 74., 75., 76.],
                           [77., 78., 79., 80.],
                           [81., 82., 83., 84.]],
                          [[85., 86., 87., 88.],
                           [89., 90., 91., 92.],
                           [93., 94., 95., 96.]]],
                         [[[25., 26., 27., 28.],
                           [29., 30., 31., 32.],
                           [33., 34., 35., 36.]],
                          [[37., 38., 39., 40.],
                           [41., 42., 43., 44.],
                           [45., 46., 47., 48.]]],
                         [[[49., 50., 51., 52.],
                           [53., 54., 55., 56.],
                           [57., 58., 59., 60.]],
                          [[61., 62., 63., 64.],
                           [65., 66., 67., 68.],
                           [69., 70., 71., 72.]]]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_small_float32_use_locking_false():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([1, 0]).astype(np.int32))
    updates = Tensor(np.arange(6).reshape((2, 3)).astype(np.float32))
    output = scatter_add_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[3., 4., 5.],
                         [0., 1., 2.]])
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

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_large_int32():
    inputx = Tensor(np.zeros((2, 3, 4)).astype(np.int32))
    indices = Tensor(np.array([[0, 0], [1, 1]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[[138., 140., 142., 144.],
                          [146., 148., 150., 152.],
                          [154., 156., 158., 160.]],
                         [[186., 188., 190., 192.],
                          [194., 196., 198., 200.],
                          [202., 204., 206., 208.]]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_disordered_int32():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32)))
    indices = Tensor(np.array([[[0, 1, 2],
                                [2, 1, 0]],
                               [[0, 0, 0],
                                [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[464., 468., 472., 476.],
                         [187., 188., 189., 190.],
                         [492., 496., 500., 504.]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_disordered_dynamic_int32():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32)))
    indices = Tensor(np.array([[[0, 1, 2],
                                [2, 1, 0]],
                               [[0, 0, 0],
                                [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))
    output = scatter_add_d_net(inputx, indices, updates)
    expected = np.array([[464., 468., 472., 476.],
                         [187., 188., 189., 190.],
                         [492., 496., 500., 504.]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_disordered_dynamic_int8():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int8)))
    indices = Tensor(np.array([[[0, 1, 2],
                                [2, 1, 0]],
                               [[0, 0, 0],
                                [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int8))
    output = scatter_add_d_net(inputx, indices, updates)
    expected = np.array([[464., 468., 472., 476.],
                         [187., 188., 189., 190.],
                         [492., 496., 500., 504.]]).astype(np.int8)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_disordered_dynamic_uint8():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.uint8)))
    indices = Tensor(np.array([[[0, 1, 2],
                                [2, 1, 0]],
                               [[0, 0, 0],
                                [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.uint8))
    output = scatter_add_d_net(inputx, indices, updates)
    expected = np.array([[464., 468., 472., 476.],
                         [187., 188., 189., 190.],
                         [492., 496., 500., 504.]]).astype(np.uint8)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_input_less_than_1_dynamic_float32():
    inputx = Tensor(np.array([[0.214141, 0.415151, 0.51516],
                              [0.876542, 0.451611, 0.55112],
                              [0.111244, 0.633333, 0.34444]]).astype(np.float32))
    indices = Tensor(np.array([[[1, 0, 2],
                                [2, 2, 0]],
                               [[1, 0, 1],
                                [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(np.float32))
    output = scatter_add_d_net(inputx, indices, updates)
    expected = np.array([[141.21414, 144.41515, 147.51517],
                         [208.87654, 212.45161, 216.55112],
                         [257.11124, 262.63333, 267.34442]], dtype=np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_add_dynamic_two_inputs():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices_1 = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates_1 = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    indices_2 = Tensor(np.array([[0, 0], [1, 1], [1, 0]]).astype(np.int32))
    updates_2 = Tensor(np.flip(np.arange(18).reshape((3, 2, 3)).astype(np.float32)))
    output_1, output_2 = scatter_add_d2_net(inputx, indices_1, updates_1,
                                            indices_2, updates_2)
    expected_1 = np.array([[6., 8., 10.],
                           [12., 14., 16.]])
    expected_2 = np.array([[39., 38., 37.],
                           [36., 35., 34.]])
    np.testing.assert_array_almost_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_almost_equal(output_2.asnumpy(), expected_2)
