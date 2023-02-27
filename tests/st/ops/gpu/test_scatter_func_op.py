# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops.functional import vmap
from mindspore.ops.operations import _inner_ops as inner

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# all cases tested against dchip

func_map = {
    "update": P.ScatterUpdate,
    "add": P.ScatterAdd,
    "sub": P.ScatterSub,
    "max": P.ScatterMax,
    "min": P.ScatterMin,
}


class TestScatterFuncNet(nn.Cell):
    def __init__(self, func, lock, inputx, indices, updates):
        super(TestScatterFuncNet, self).__init__()

        self.scatter_func = func_map.get(func)(use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_func(self.inputx, self.indices, self.updates)
        return out


def scatter_func_net(func, inputx, indices, updates):
    lock = True
    net = TestScatterFuncNet(func, lock, inputx, indices, updates)
    return net()


def scatter_func_use_locking_false_net(func, inputx, indices, updates):
    lock = False
    net = TestScatterFuncNet(func, lock, inputx, indices, updates)
    return net()


class TestScatterFuncDynamicNet(nn.Cell):
    def __init__(self, func, inputx, indices, updates):
        super(TestScatterFuncDynamicNet, self).__init__()
        self.scatter_func = func_map.get(func)()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        indices = self.test_dynamic(self.indices)
        updates = self.test_dynamic(self.updates)
        out = self.scatter_func(self.inputx, indices, updates)
        return out


def scatter_func_d_net(func, inputx, indices, updates):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = TestScatterFuncDynamicNet(func, inputx, indices, updates)
    return net()


class TestScatterFuncDynamicNet2(nn.Cell):
    def __init__(self, func, inputx):
        super(TestScatterFuncDynamicNet2, self).__init__()
        self.scatter_func = func_map.get(func)()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        indices = self.test_dynamic(indices)
        updates = self.test_dynamic(updates)
        out = self.scatter_func(self.inputx, indices, updates)
        return out


def scatter_func_d2_net(func, inputx, indices_1, updates_1, indices_2, updates_2):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = TestScatterFuncDynamicNet2(func, inputx)
    out1 = net(indices_1, updates_1)
    out2 = net(indices_2, updates_2)
    return (out1, out2)


class ScatterFuncVmapNet(nn.Cell):
    def __init__(self, func):
        super(ScatterFuncVmapNet, self).__init__()
        self.scatter_func = func_map.get(func)()

    def construct(self, inputx, indices, updates):
        return self.scatter_func(inputx, indices, updates)


class VmapNet(nn.Cell):
    def __init__(self, net, inputx, in_axes, out_axes):
        super(VmapNet, self).__init__()
        self.net = net
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        return vmap(self.net, self.in_axes, self.out_axes)(self.inputx, indices, updates)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_small_float32():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))

    # update
    output = scatter_func_net("update", inputx, indices, updates)
    expected = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array([[-6.0, -8.0, -10.0], [-12.0, -14.0, -16.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_input_updated():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    lock = True

    # update
    net = TestScatterFuncNet("update", lock, inputx, indices, updates)
    net()
    expected = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

    # add
    net = TestScatterFuncNet("add", lock, inputx, indices, updates)
    net()
    expected = np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

    # sub
    net = TestScatterFuncNet("sub", lock, inputx, indices, updates)
    net()
    expected = np.array([[-6.0, -8.0, -10.0], [-12.0, -14.0, -16.0]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

    # max
    net = TestScatterFuncNet("max", lock, inputx, indices, updates)
    net()
    expected = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

    # min
    net = TestScatterFuncNet("min", lock, inputx, indices, updates)
    net()
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_large_shape_float32():
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.float32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.float32))

    # update
    output = scatter_func_net("update", inputx, indices, updates)
    expected = np.array(
        [
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
                [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            [
                [[72.0, 73.0, 74.0, 75.0], [76.0, 77.0, 78.0, 79.0], [80.0, 81.0, 82.0, 83.0]],
                [[84.0, 85.0, 86.0, 87.0], [88.0, 89.0, 90.0, 91.0], [92.0, 93.0, 94.0, 95.0]],
            ],
            [
                [[24.0, 25.0, 26.0, 27.0], [28.0, 29.0, 30.0, 31.0], [32.0, 33.0, 34.0, 35.0]],
                [[36.0, 37.0, 38.0, 39.0], [40.0, 41.0, 42.0, 43.0], [44.0, 45.0, 46.0, 47.0]],
            ],
            [
                [[48.0, 49.0, 50.0, 51.0], [52.0, 53.0, 54.0, 55.0], [56.0, 57.0, 58.0, 59.0]],
                [[60.0, 61.0, 62.0, 63.0], [64.0, 65.0, 66.0, 67.0], [68.0, 69.0, 70.0, 71.0]],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array(
        [
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]],
            ],
            [
                [[73.0, 74.0, 75.0, 76.0], [77.0, 78.0, 79.0, 80.0], [81.0, 82.0, 83.0, 84.0]],
                [[85.0, 86.0, 87.0, 88.0], [89.0, 90.0, 91.0, 92.0], [93.0, 94.0, 95.0, 96.0]],
            ],
            [
                [[25.0, 26.0, 27.0, 28.0], [29.0, 30.0, 31.0, 32.0], [33.0, 34.0, 35.0, 36.0]],
                [[37.0, 38.0, 39.0, 40.0], [41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0]],
            ],
            [
                [[49.0, 50.0, 51.0, 52.0], [53.0, 54.0, 55.0, 56.0], [57.0, 58.0, 59.0, 60.0]],
                [[61.0, 62.0, 63.0, 64.0], [65.0, 66.0, 67.0, 68.0], [69.0, 70.0, 71.0, 72.0]],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [
                [[1.0, 0.0, -1.0, -2.0], [-3.0, -4.0, -5.0, -6.0], [-7.0, -8.0, -9.0, -10.0]],
                [
                    [-11.0, -12.0, -13.0, -14.0],
                    [-15.0, -16.0, -17.0, -18.0],
                    [-19.0, -20.0, -21.0, -22.0],
                ],
            ],
            [
                [
                    [-71.0, -72.0, -73.0, -74.0],
                    [-75.0, -76.0, -77.0, -78.0],
                    [-79.0, -80.0, -81.0, -82.0],
                ],
                [
                    [-83.0, -84.0, -85.0, -86.0],
                    [-87.0, -88.0, -89.0, -90.0],
                    [-91.0, -92.0, -93.0, -94.0],
                ],
            ],
            [
                [
                    [-23.0, -24.0, -25.0, -26.0],
                    [-27.0, -28.0, -29.0, -30.0],
                    [-31.0, -32.0, -33.0, -34.0],
                ],
                [
                    [-35.0, -36.0, -37.0, -38.0],
                    [-39.0, -40.0, -41.0, -42.0],
                    [-43.0, -44.0, -45.0, -46.0],
                ],
            ],
            [
                [
                    [-47.0, -48.0, -49.0, -50.0],
                    [-51.0, -52.0, -53.0, -54.0],
                    [-55.0, -56.0, -57.0, -58.0],
                ],
                [
                    [-59.0, -60.0, -61.0, -62.0],
                    [-63.0, -64.0, -65.0, -66.0],
                    [-67.0, -68.0, -69.0, -70.0],
                ],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array(
        [
            [
                [
                    [1.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ],
                [
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                ],
            ],
            [
                [
                    [72.0, 73.0, 74.0, 75.0],
                    [76.0, 77.0, 78.0, 79.0],
                    [80.0, 81.0, 82.0, 83.0],
                ],
                [
                    [84.0, 85.0, 86.0, 87.0],
                    [88.0, 89.0, 90.0, 91.0],
                    [92.0, 93.0, 94.0, 95.0],
                ],
            ],
            [
                [
                    [24.0, 25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0, 31.0],
                    [32.0, 33.0, 34.0, 35.0],
                ],
                [
                    [36.0, 37.0, 38.0, 39.0],
                    [40.0, 41.0, 42.0, 43.0],
                    [44.0, 45.0, 46.0, 47.0],
                ],
            ],
            [
                [
                    [48.0, 49.0, 50.0, 51.0],
                    [52.0, 53.0, 54.0, 55.0],
                    [56.0, 57.0, 58.0, 59.0],
                ],
                [
                    [60.0, 61.0, 62.0, 63.0],
                    [64.0, 65.0, 66.0, 67.0],
                    [68.0, 69.0, 70.0, 71.0],
                ],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.ones((4, 2, 3, 4)).astype(np.float32)
    expected[0][0][0][0] = 0.0
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_small_float32_use_locking_false():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices = Tensor(np.array([1, 0]).astype(np.int32))
    updates = Tensor(np.arange(6).reshape((2, 3)).astype(np.float32))

    # update
    output = scatter_func_use_locking_false_net("update", inputx, indices, updates)
    expected = np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_use_locking_false_net("add", inputx, indices, updates)
    expected = np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_use_locking_false_net("sub", inputx, indices, updates)
    expected = np.array([[-3.0, -4.0, -5.0], [0.0, -1.0, -2.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_use_locking_false_net("max", inputx, indices, updates)
    expected = np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_use_locking_false_net("min", inputx, indices, updates)
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_input_less_than_1_float32():
    inputx = Tensor(
        np.array(
            [
                [0.214141, 0.415151, 0.51516],
                [0.876542, 0.451611, 0.55112],
                [0.111244, 0.633333, 0.34444],
            ]
        ).astype(np.float32)
    )
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(np.float32))

    # update
    indices_unique = Tensor(np.array([[[1, 0, 2]]]).astype(np.int32))
    updates_unique = Tensor(np.arange(34, 43).reshape((1, 1, 3, 3)).astype(np.float32))
    output = scatter_func_net("update", inputx, indices_unique, updates_unique)
    expected = np.array(
        [[37.0, 38.0, 39.0], [34.0, 35.0, 36.0], [40.0, 41.0, 42.0]], dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array(
        [
            [141.21414, 144.41515, 147.51517],
            [208.87654, 212.45161, 216.55112],
            [257.11124, 262.63333, 267.34442],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [-140.78586, -143.58485, -146.48483],
            [-207.12346, -211.54839, -215.44888],
            [-256.88876, -261.36667, -266.65558],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array(
        [
            [55.0, 56.0, 57.0],
            [64.0, 65.0, 66.0],
            [67.0, 68.0, 69.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_float16():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float16))
    indices = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float16))

    # update
    output = scatter_func_net("update", inputx, indices, updates)
    expected = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array([[-6.0, -8.0, -10.0], [-12.0, -14.0, -16.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_large_float16():
    inputx = Tensor(np.zeros((2, 3, 4)).astype(np.float16))
    indices = Tensor(np.array([[0, 0], [1, 1]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.float16))

    # update
    indices_unique = Tensor(np.array([[1, 0]]).astype(np.int32))
    updates_unique = Tensor(np.arange(87, 111).reshape((1, 2, 3, 4)).astype(np.float16))
    output = scatter_func_net("update", inputx, indices_unique, updates_unique)
    expected = np.array([
        [[99.0, 100.0, 101.0, 102.0], [103.0, 104.0, 105.0, 106.0], [107.0, 108.0, 109.0, 110.0]],
        [[87.0, 88.0, 89.0, 90.0], [91.0, 92.0, 93.0, 94.0], [95.0, 96.0, 97.0, 98.0]],
    ])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array([
        [[138.0, 140.0, 142.0, 144.0], [146.0, 148.0, 150.0, 152.0], [154.0, 156.0, 158.0, 160.0]],
        [[186.0, 188.0, 190.0, 192.0], [194.0, 196.0, 198.0, 200.0], [202.0, 204.0, 206.0, 208.0]],
    ])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array([
        [
            [-138.0, -140.0, -142.0, -144.0],
            [-146.0, -148.0, -150.0, -152.0],
            [-154.0, -156.0, -158.0, -160.0],
        ],
        [
            [-186.0, -188.0, -190.0, -192.0],
            [-194.0, -196.0, -198.0, -200.0],
            [-202.0, -204.0, -206.0, -208.0],
        ],
    ])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array([
        [[75.0, 76.0, 77.0, 78.0], [79.0, 80.0, 81.0, 82.0], [83.0, 84.0, 85.0, 86.0]],
        [[99.0, 100.0, 101.0, 102.0], [103.0, 104.0, 105.0, 106.0], [107.0, 108.0, 109.0, 110.0]],
    ])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.zeros((2, 3, 4)).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_disordered_float16():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.float16)))
    indices = Tensor(np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.float16))

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array(
        [[464.0, 468.0, 472.0, 476.0], [187.0, 188.0, 189.0, 190.0], [492.0, 496.0, 500.0, 504.0]]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [-374.0, -380.0, -386.0, -392.0],
            [-105.0, -108.0, -111.0, -114.0],
            [-418.0, -424.0, -430.0, -436.0],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array(
        [
            [95.0, 96.0, 97.0, 98.0],
            [79.0, 80.0, 81.0, 82.0],
            [107.0, 108.0, 109.0, 110.0],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.flip(np.arange(34, 46).reshape(3, 4).astype(np.float16))
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_large_int32():
    inputx = Tensor(np.zeros((2, 3, 4)).astype(np.int32))
    indices = Tensor(np.array([[0, 0], [1, 1]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))

    # update
    indices_unique = Tensor(np.array([[1, 0]]).astype(np.int32))
    updates_unique = Tensor(np.arange(87, 111).reshape((1, 2, 3, 4)).astype(np.int32))
    output = scatter_func_net("update", inputx, indices_unique, updates_unique)
    expected = np.array([
        [[99.0, 100.0, 101.0, 102.0], [103.0, 104.0, 105.0, 106.0], [107.0, 108.0, 109.0, 110.0]],
        [[87.0, 88.0, 89.0, 90.0], [91.0, 92.0, 93.0, 94.0], [95.0, 96.0, 97.0, 98.0]],
    ]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array([
        [[138.0, 140.0, 142.0, 144.0], [146.0, 148.0, 150.0, 152.0], [154.0, 156.0, 158.0, 160.0]],
        [[186.0, 188.0, 190.0, 192.0], [194.0, 196.0, 198.0, 200.0], [202.0, 204.0, 206.0, 208.0]],
    ]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array([
        [
            [-138.0, -140.0, -142.0, -144.0],
            [-146.0, -148.0, -150.0, -152.0],
            [-154.0, -156.0, -158.0, -160.0],
        ],
        [
            [-186.0, -188.0, -190.0, -192.0],
            [-194.0, -196.0, -198.0, -200.0],
            [-202.0, -204.0, -206.0, -208.0],
        ],
    ]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array([
        [[75.0, 76.0, 77.0, 78.0], [79.0, 80.0, 81.0, 82.0], [83.0, 84.0, 85.0, 86.0]],
        [[99.0, 100.0, 101.0, 102.0], [103.0, 104.0, 105.0, 106.0], [107.0, 108.0, 109.0, 110.0]],
    ])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.zeros((2, 3, 4)).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_disordered_int32():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32)))
    indices = Tensor(np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))

    # add
    output = scatter_func_net("add", inputx, indices, updates)
    expected = np.array(
        [[464.0, 468.0, 472.0, 476.0], [187.0, 188.0, 189.0, 190.0], [492.0, 496.0, 500.0, 504.0]]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [-374.0, -380.0, -386.0, -392.0],
            [-105.0, -108.0, -111.0, -114.0],
            [-418.0, -424.0, -430.0, -436.0],
        ]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_net("max", inputx, indices, updates)
    expected = np.array(
        [
            [95.0, 96.0, 97.0, 98.0],
            [79.0, 80.0, 81.0, 82.0],
            [107.0, 108.0, 109.0, 110.0],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_net("min", inputx, indices, updates)
    expected = np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32))
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_disordered_dynamic_int32():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32)))
    indices = Tensor(np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int32))

    # add
    output = scatter_func_d_net("add", inputx, indices, updates)
    expected = np.array(
        [[464.0, 468.0, 472.0, 476.0], [187.0, 188.0, 189.0, 190.0], [492.0, 496.0, 500.0, 504.0]]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_d_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [-374.0, -380.0, -386.0, -392.0],
            [-105.0, -108.0, -111.0, -114.0],
            [-418.0, -424.0, -430.0, -436.0],
        ]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_d_net("max", inputx, indices, updates)
    expected = np.array(
        [[95.0, 96.0, 97.0, 98.0], [79.0, 80.0, 81.0, 82.0], [107.0, 108.0, 109.0, 110.0]]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_d_net("min", inputx, indices, updates)
    expected = np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int32))
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_disordered_dynamic_int8():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.int8)))
    indices = Tensor(np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.int8))

    # add
    output = scatter_func_d_net("add", inputx, indices, updates)
    expected = np.array(
        [[464.0, 468.0, 472.0, 476.0], [187.0, 188.0, 189.0, 190.0], [492.0, 496.0, 500.0, 504.0]]
    ).astype(np.int8)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_d_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [-118.0, -124.0, 126.0, 120.0],
            [-105.0, -108.0, -111.0, -114.0],
            [94.0, 88.0, 82.0, 76.0],
        ]
    ).astype(np.int8)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_disordered_dynamic_uint8():
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(np.uint8)))
    indices = Tensor(np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(np.uint8))

    # add
    output = scatter_func_d_net("add", inputx, indices, updates)
    expected = np.array(
        [[464.0, 468.0, 472.0, 476.0], [187.0, 188.0, 189.0, 190.0], [492.0, 496.0, 500.0, 504.0]]
    ).astype(np.uint8)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_d_net("sub", inputx, indices, updates)
    expected = np.array(
        [[138.0, 132.0, 126.0, 120.0], [151.0, 148.0, 145.0, 142.0], [94.0, 88.0, 82.0, 76.0]]
    ).astype(np.uint8)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_input_less_than_1_dynamic_float32():
    inputx_np = np.array(
        [
            [0.214141, 0.415151, 0.51516],
            [0.876542, 0.451611, 0.55112],
            [0.111244, 0.633333, 0.34444],
        ]
    ).astype(np.float32)
    inputx = Tensor(inputx_np)
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(np.float32))

    # update
    indices_unique = Tensor(np.array([[[1, 0, 2]]]).astype(np.int32))
    updates_unique = Tensor(np.arange(34, 43).reshape((1, 1, 3, 3)).astype(np.float32))
    output = scatter_func_d_net("update", inputx, indices_unique, updates_unique)
    expected = np.array(
        [[37.0, 38.0, 39.0], [34.0, 35.0, 36.0], [40.0, 41.0, 42.0]], dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_func_d_net("add", inputx, indices, updates)
    expected = np.array(
        [
            [141.21414, 144.41515, 147.51517],
            [208.87654, 212.45161, 216.55112],
            [257.11124, 262.63333, 267.34442],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_func_d_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [-140.78586, -143.58485, -146.48483],
            [-207.12346, -211.54839, -215.44888],
            [-256.88876, -261.36667, -266.65558],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # max
    output = scatter_func_d_net("max", inputx, indices, updates)
    expected = np.array(
        [[55.0, 56.0, 57.0], [64.0, 65.0, 66.0], [67.0, 68.0, 69.0]], dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # min
    output = scatter_func_d_net("min", inputx, indices, updates)
    expected = inputx_np
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_dynamic_two_inputs():
    inputx = Tensor(np.zeros((2, 3)).astype(np.float32))
    indices_1 = Tensor(np.array([[0, 1], [0, 1]]).astype(np.int32))
    updates_1 = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    indices_2 = Tensor(np.array([[0, 0], [1, 1], [1, 0]]).astype(np.int32))
    updates_2 = Tensor(np.flip(np.arange(18).reshape((3, 2, 3)).astype(np.float32)))

    # update
    output_1, output_2 = scatter_func_d2_net(
        "update", inputx, indices_1, updates_1, indices_2, updates_2
    )
    expected_1 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    expected_2 = np.array([[17.0, 16.0, 15.0], [11.0, 10.0, 9.0]])
    np.testing.assert_array_almost_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_almost_equal(output_2.asnumpy(), expected_2)

    # add
    output_1, output_2 = scatter_func_d2_net(
        "add", inputx, indices_1, updates_1, indices_2, updates_2
    )
    expected_1 = np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])
    expected_2 = np.array([[39.0, 38.0, 37.0], [36.0, 35.0, 34.0]])
    np.testing.assert_array_almost_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_almost_equal(output_2.asnumpy(), expected_2)

    # sub
    output_1, output_2 = scatter_func_d2_net(
        "sub", inputx, indices_1, updates_1, indices_2, updates_2
    )
    expected_1 = np.array([[-6.0, -8.0, -10.0], [-12.0, -14.0, -16.0]])
    expected_2 = np.array([[-39.0, -38.0, -37.0], [-36.0, -35.0, -34.0]])
    np.testing.assert_array_almost_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_almost_equal(output_2.asnumpy(), expected_2)

    # max
    output_1, output_2 = scatter_func_d2_net(
        "max", inputx, indices_1, updates_1, indices_2, updates_2
    )
    expected_1 = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    expected_2 = np.array([[17.0, 16.0, 15.0], [11.0, 10.0, 11.0]])
    np.testing.assert_array_almost_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_almost_equal(output_2.asnumpy(), expected_2)

    # min
    output_1, output_2 = scatter_func_d2_net(
        "min", inputx, indices_1, updates_1, indices_2, updates_2
    )
    expected_1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected_2 = expected_1
    np.testing.assert_array_almost_equal(output_1.asnumpy(), expected_1)
    np.testing.assert_array_almost_equal(output_2.asnumpy(), expected_2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_indices_vmap():
    """
    Feature: test scatter_func vmap.
    Description: in_axes: (0, 0, None).
    Expectation: the result match with numpy result
    """
    inputx = Parameter(Tensor(np.array(
        [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]
    ).astype(np.int32)), name="inputx")
    indices = Tensor(np.array([[[0, 1], [1, 1]], [[0, 1], [0, 1]], [[1, 1], [1, 0]]]).astype(np.int32))
    updates = Tensor(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]).astype(np.int32))
    in_axes = (0, 0, None)
    out_axes = 0

    # scatter_max
    output = VmapNet(ScatterFuncVmapNet("max"), inputx, in_axes, out_axes)(indices, updates)
    expected = np.array(
        [[[1, 1, 2], [4, 4, 5]], [[3, 3, 3], [4, 4, 5]], [[4, 4, 4], [3, 4, 5]]]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_min
    output = VmapNet(ScatterFuncVmapNet("min"), inputx, in_axes, out_axes)(indices, updates)
    expected = np.array(
        [[[0, 1, 1], [2, 2, 2]], [[0, 1, 1], [2, 2, 2]], [[0, 1, 2], [1, 1, 1]]]
    ).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_update
    inputx = Parameter(Tensor(np.array(
        [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]
    ).astype(np.float32)), name="inputx")
    indices = Tensor(np.array([[0, 1], [1, 0], [0, 1]]).astype(np.int32))
    updates = Tensor(np.array([[1, 1, 1], [2, 2, 2]]).astype(np.float32))
    output = VmapNet(ScatterFuncVmapNet("update"), inputx, in_axes, out_axes)(indices, updates)
    expected = np.array(
        [[[1, 1, 1], [2, 2, 2]], [[2, 2, 2], [1, 1, 1]], [[1, 1, 1], [2, 2, 2]]]
    ).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_updates_vmap():
    """
    Feature: test scatter_func vmap.
    Description: in_axes: (0, None, 0).
    Expectation: the result match with numpy result
    """
    inputx = Parameter(Tensor(np.array([[0.1, 1.0, 2.2], [3.0, 4.3, 5.5]]).astype(np.float32)), name="inputx")
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.array([[1.0, 0.1], [1.2, 1.3]]).astype(np.float32))
    in_axes = (0, None, 0)
    out_axes = 0

    # scatter_max
    output = VmapNet(ScatterFuncVmapNet("max"), inputx, in_axes, out_axes)(indices, updates)
    expected = np.array([[1.0, 1.0, 2.2], [3.0, 4.3, 5.5]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_min
    output = VmapNet(ScatterFuncVmapNet("min"), inputx, in_axes, out_axes)(indices, updates)
    expected = np.array([[0.1, 0.1, 2.2], [1.2, 1.3, 5.5]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_update
    output = VmapNet(ScatterFuncVmapNet("update"), inputx, in_axes, out_axes)(indices, updates)
    expected = np.array([[1.0, 0.1, 2.2], [1.2, 1.3, 5.5]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
