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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

func_map = {
    "update": ops.ScatterNdUpdate,
    "add": ops.ScatterNdAdd,
    "sub": ops.ScatterNdSub,
}


class TestScatterNdFuncNet(nn.Cell):
    def __init__(self, func, lock, inputx, indices, updates):
        super(TestScatterNdFuncNet, self).__init__()

        self.scatter_func = func_map[func](use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_func(self.inputx, self.indices, self.updates)
        return out


def scatter_nd_func_net(func, inputx, indices, updates):
    lock = True
    net = TestScatterNdFuncNet(func, lock, inputx, indices, updates)
    return net()


def scatter_nd_func_use_locking_false_net(func, inputx, indices, updates):
    lock = False
    net = TestScatterNdFuncNet(func, lock, inputx, indices, updates)
    return net()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_traning
@pytest.mark.env_onecard
def test_scatter_nd_func_small_float32():
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)

    # update
    output = scatter_nd_func_net("update", inputx, indices, updates)
    expected = np.array([[1.0, 0.3, 3.6], [0.4, 2.2, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_nd_func_net("add", inputx, indices, updates)
    expected = np.array([[0.9, 0.3, 3.6], [0.4, 2.7, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_nd_func_net("sub", inputx, indices, updates)
    expected = np.array([[-1.1, 0.3, 3.6], [0.4, -1.7, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_nd_func_input_updated():
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)
    lock = True

    # update
    net = TestScatterNdFuncNet("update", lock, inputx, indices, updates)
    net()
    expected = np.array([[1.0, 0.3, 3.6], [0.4, 2.2, -3.2]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

    # add
    net = TestScatterNdFuncNet("add", lock, inputx, indices, updates)
    net()
    expected = np.array([[0.9, 0.3, 3.6], [0.4, 2.7, -3.2]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)

    # sub
    net = TestScatterNdFuncNet("sub", lock, inputx, indices, updates)
    net()
    expected = np.array([[-1.1, 0.3, 3.6], [0.4, -1.7, -3.2]])
    np.testing.assert_array_almost_equal(net.inputx.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_traning
@pytest.mark.env_onecard
def test_scatter_nd_func_small_float32_using_locking_false():
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)

    # update
    output = scatter_nd_func_use_locking_false_net("update", inputx, indices, updates)
    expected = np.array([[1.0, 0.3, 3.6], [0.4, 2.2, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_nd_func_use_locking_false_net("add", inputx, indices, updates)
    expected = np.array([[0.9, 0.3, 3.6], [0.4, 2.7, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_nd_func_use_locking_false_net("sub", inputx, indices, updates)
    expected = np.array([[-1.1, 0.3, 3.6], [0.4, -1.7, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_traning
@pytest.mark.env_onecard
def test_scatter_nd_func_small_int32():
    inputx = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[4], [3], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([9, 10, 11, 12]), mstype.float32)

    # update
    output = scatter_nd_func_net("update", inputx, indices, updates)
    expected = np.array([1, 11, 3, 10, 9, 6, 7, 12])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_nd_func_net("add", inputx, indices, updates)
    expected = np.array([1, 13, 3, 14, 14, 6, 7, 20])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_nd_func_net("sub", inputx, indices, updates)
    expected = np.array([1, -9, 3, -6, -4, 6, 7, -4])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_traning
@pytest.mark.env_onecard
def test_scatter_nd_func_multi_dims():
    inputx = Tensor(np.zeros((4, 4, 4)), mstype.float32)
    indices = Tensor(np.array([[0], [2]]), mstype.int32)
    updates = Tensor(
        np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ]
        ),
        mstype.float32,
    )

    # update
    output = scatter_nd_func_net("update", inputx, indices, updates)
    expected = np.array(
        [
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_nd_func_net("add", inputx, indices, updates)
    expected = np.array(
        [
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_nd_func_net("sub", inputx, indices, updates)
    expected = np.array(
        [
            [[-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7], [-8, -8, -8, -8]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7], [-8, -8, -8, -8]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_traning
@pytest.mark.env_onecard
def test_scatter_nd_func_one_value():
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0]), mstype.float32)

    # update
    output = scatter_nd_func_net("update", inputx, indices, updates)
    expected = np.array([[-0.1, 1.0, 3.6], [0.4, 0.5, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # add
    output = scatter_nd_func_net("add", inputx, indices, updates)
    expected = np.array([[-0.1, 1.3, 3.6], [0.4, 0.5, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # sub
    output = scatter_nd_func_net("sub", inputx, indices, updates)
    expected = np.array([[-0.1, -0.7, 3.6], [0.4, 0.5, -3.2]])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
