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
"""CPU tests for stateful scatter arithmetic."""
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations import _scatter_ops
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class TestScatterAddNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super().__init__()
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_small_float32():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[6., 8., 10.],
                         [12., 14., 16.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_input_updated():
    """
    Feature: Test ScatterAdd
    Description: Net test op ScatterAdd
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterAddNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[6., 8., 10.],
                         [12., 14., 16.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_large_shape_float32():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_small_float32_use_locking_false():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([1, 0]).astype(np.int32))
    updates = Tensor(np.arange(6).reshape((2, 3)).astype(np.float32))
    output = scatter_add_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[3., 4., 5.],
                         [0., 1., 2.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_input_less_than_1_float32():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_float16():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float16))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float16))
    output = scatter_add_net(inputx, indices, updates)
    expected = np.array([[6., 8., 10.],
                         [12., 14., 16.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_large_float16():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_disordered_float16():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_large_int32():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_disordered_int32():
    """
    Feature: Test scatter with large shape
    Description: Net test op ScatterAdd
    Expectation: Success
    """
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_function():
    """
    Feature: test_scatter_add_function.
    Description: test cases for scatter add functional
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32)))
    indices = Tensor(np.array([[[0, 1, 2],
                                [2, 1, 0]],
                               [[0, 0, 0],
                                [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))
    output = F.scatter_add(input_x, indices, updates)
    expected = np.array([[464., 468., 472., 476.],
                         [187., 188., 189., 190.],
                         [492., 496., 500., 504.]]).astype(np.int32)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-6)


class TestScatterMulNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super().__init__()
        self.scatter_mul = _scatter_ops.ScatterMul(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_mul(self.inputx, self.indices, self.updates)
        return out


def scatter_mul_net(inputx, indices, updates):
    lock = True
    net = TestScatterMulNet(lock, inputx, indices, updates)
    return net()


def scatter_mul_use_locking_false_net(inputx, indices, updates):
    lock = False
    net = TestScatterMulNet(lock, inputx, indices, updates)
    return net()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_mul_input_updated():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterMulNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[0., 7., 16.],
                         [27., 40., 55.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_mul_output_updated_float32():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_mul_net(inputx, indices, updates)
    expected = np.array([[0., 7., 16.],
                         [27., 40., 55.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_mul_small_float32_use_locking_false():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_mul_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[0., 7., 16.],
                         [27., 40., 55.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


class TestScatterDivNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super().__init__()
        self.scatter_div = _scatter_ops.ScatterDiv(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_div(self.inputx, self.indices, self.updates)
        return out


def scatter_div_net(inputx, indices, updates):
    lock = True
    net = TestScatterDivNet(lock, inputx, indices, updates)
    return net()


def scatter_div_use_locking_false_net(inputx, indices, updates):
    lock = False
    net = TestScatterDivNet(lock, inputx, indices, updates)
    return net()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_div_input_updated():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(1, 13).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterDivNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[0., 0., 0.],
                         [0., 0., 0.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_div_output_updated_float32():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(1, 13).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_div_net(inputx, indices, updates)
    expected = np.array([[0., 0., 0.],
                         [0., 0., 0.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_div_small_float32_use_locking_false():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32) * 10)
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.ones(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_div_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[10., 10., 10.],
                         [10., 10., 10.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_div_output_int16():
    """
    Feature: test ScatterDiv output and input_x same value.
    Description: input is int16.
    Expectation: output and input_x have same value
    """
    input_x = Parameter(Tensor(np.array([[6, 6, 6], [2, 2, 2]]), mindspore.int16), name="x")
    indices = Tensor(np.array([0, 1]), mindspore.int32)
    updates = Tensor(np.array([[2, 2, 2], [2, 2, 2]]), mindspore.int16)
    output = _scatter_ops.ScatterDiv()(input_x, indices, updates)
    assert np.allclose(output.asnumpy(), input_x.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_div_output_float64():
    """
    Feature: test ScatterDiv output and input_x same value.
    Description: input is float64.
    Expectation: output and input_x have same value
    """
    input_x = Parameter(Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mindspore.float64), name="x")
    indices = Tensor(np.array([0, 1]), mindspore.int32)
    updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float64)
    output = _scatter_ops.ScatterDiv()(input_x, indices, updates)
    assert np.allclose(output.asnumpy(), input_x.asnumpy(), 0.0001, 0.0001)


class TestScatterMaxNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super().__init__()
        self.scatter_max = _scatter_ops.ScatterMax(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_max(self.inputx, self.indices, self.updates)
        return out


def scatter_max_net(inputx, indices, updates):
    lock = True
    net = TestScatterMaxNet(lock, inputx, indices, updates)
    return net()


def scatter_max_use_locking_false_net(inputx, indices, updates):
    lock = False
    net = TestScatterMaxNet(lock, inputx, indices, updates)
    return net()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_max_input_updated():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterMaxNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[6., 7., 8.],
                         [9., 10., 11.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_max_output_updated_float32():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_max_net(inputx, indices, updates)
    expected = np.array([[6., 7., 8.],
                         [9., 10., 11.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_max_small_float32_use_locking_false():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32) * 10)
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_max_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[10., 10., 10.],
                         [10., 10., 11.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


class TestScatterMinNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super().__init__()
        self.scatter_min = _scatter_ops.ScatterMin(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_min(self.inputx, self.indices, self.updates)
        return out


def scatter_min_net(inputx, indices, updates):
    lock = True
    net = TestScatterMinNet(lock, inputx, indices, updates)
    return net()


def scatter_min_use_locking_false_net(inputx, indices, updates):
    lock = False
    net = TestScatterMinNet(lock, inputx, indices, updates)
    return net()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_min_input_updated():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterMinNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[0., 0., 0.],
                         [0., 0., 0.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_min_output_updated_float32():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_min_net(inputx, indices, updates)
    expected = np.array([[0., 1., 1.],
                         [1., 1., 1.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_min_small_float32_use_locking_false():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_min_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[0., 1., 1.],
                         [1., 1., 1.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


class TestScatterUpdateNet(nn.Cell):
    def __init__(self, lock, inputx, indices, updates):
        super().__init__()
        self.scatter_update = P.ScatterUpdate(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_update(self.inputx, self.indices, self.updates)
        return out


def scatter_update_net(inputx, indices, updates):
    lock = True
    net = TestScatterUpdateNet(lock, inputx, indices, updates)
    return net()


def scatter_update_use_locking_false_net(inputx, indices, updates):
    lock = False
    net = TestScatterUpdateNet(lock, inputx, indices, updates)
    return net()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_update_input_updated():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True
    net = TestScatterUpdateNet(lock, inputx, indices, updates)
    net()
    expected = np.array([[6., 7., 8.],
                         [9., 10., 11.]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_update_output_updated_float32():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[6., 7., 8.],
                         [9., 10., 11.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_update_output_updated_huge_tensor_float32():
    """
    Feature: Test huge input tensor case of cpu kernel ScatterUpdate.
    Description: The first input tensor for cpu kernel ScatterUpdate is huge, and
                 the memory size of this tensor should be greater than 2147483647.
                 In this case, memory size of inputx tensor is 2147483652 (178956971 * 3 * sizeof(float32))
    Expectation: success.
    """
    inputx = Tensor(np.ones((178956971, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_update_net(inputx, indices, updates)
    expected = np.array([[6., 7., 8.],
                         [9., 10., 11.]])
    np.testing.assert_array_almost_equal(output.asnumpy()[0:2], expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_update_small_float32_use_locking_false():
    """
    Feature: Test scatter
    Description: Net test op scatter
    Expectation: Success
    """
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    output = scatter_update_use_locking_false_net(inputx, indices, updates)
    expected = np.array([[6., 7., 8.],
                         [9., 10., 11.]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


class TestScatterAddNetDynamic(nn.Cell):
    def __init__(self, lock):
        super().__init__()
        self.scatter_add = P.ScatterAdd(use_locking=lock)

    def construct(self, inputx, indices, updates):
        out = self.scatter_add(inputx, indices, updates)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_add_dynamic_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """
    inputx = Parameter(Tensor(np.zeros((2, 3)).astype(np.float32)))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    net = TestScatterAddNetDynamic(False)
    indices_dyn = Tensor(shape=[None, None], dtype=indices.dtype)
    updates_dyn = Tensor(shape=[None, None, None], dtype=updates.dtype)
    net.set_inputs(inputx, indices_dyn, updates_dyn)
    output = net(inputx, indices, updates)
    expected_shape = (2, 3)
    assert expected_shape == output.asnumpy().shape


class TestScatterUpdateNetDynamic(nn.Cell):
    def __init__(self, lock):
        super().__init__()
        self.scatter_update = P.ScatterUpdate(use_locking=lock)

    def construct(self, inputx, indices, updates):
        out = self.scatter_update(inputx, indices, updates)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_scatter_update_dynamic_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    inputx = Parameter(Tensor(np.zeros((2, 3)).astype(np.float32)))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    net = TestScatterUpdateNetDynamic(False)
    indices_dyn = Tensor(shape=[None, None], dtype=indices.dtype)
    updates_dyn = Tensor(shape=[None, None, None], dtype=updates.dtype)
    net.set_inputs(inputx, indices_dyn, updates_dyn)
    output = net(inputx, indices, updates)
    expected_shape = (2, 3)
    assert expected_shape == output.asnumpy().shape
