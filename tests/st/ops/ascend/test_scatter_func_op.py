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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops.functional import vmap
from mindspore import Tensor, Parameter, ParameterTuple

# all cases tested against dchip

func_map = {
    "max": ops.ScatterMax,
    "min": ops.ScatterMin,
    "update": ops.ScatterUpdate,
}


class TestScatterFuncNet(nn.Cell):
    def __init__(self, func, inputx):
        super(TestScatterFuncNet, self).__init__()

        self.scatter_func = func_map.get(func)()
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        out = self.scatter_func(self.inputx, indices, updates)
        return out


def scatter_func_forward(nptype):
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(nptype))
    indices = Tensor(
        np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(nptype))

    # scatter_max
    net = TestScatterFuncNet("max", inputx)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    expected = np.array(
        [[55.0, 56.0, 57.0], [64.0, 65.0, 66.0], [67.0, 68.0, 69.0]]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_min
    net = TestScatterFuncNet("min", inputx)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_update
    if nptype not in (np.float16, np.float32):
        return
    net = TestScatterFuncNet("update", inputx)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    expected = np.array(
        [[55.0, 56.0, 57.0], [64.0, 65.0, 66.0], [67.0, 68.0, 69.0]]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_func_dynamic_updates():
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.float32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.float32))
    updates_dy = Tensor(shape=(2, 2, 2, None, 4), dtype=mindspore.float32)

    # scatter_max
    net = TestScatterFuncNet("max", inputx)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = np.array([[[[1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]],
                         [[[72, 73, 74, 75], [76, 77, 78, 79], [80, 81, 82, 83]],
                          [[84, 85, 86, 87], [88, 89, 90, 91], [92, 93, 94, 95]]],
                         [[[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]],
                          [[36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]],
                         [[[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59]],
                          [[60, 61, 62, 63], [64, 65, 66, 67], [68, 69, 70, 71]]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_min
    net = TestScatterFuncNet("min", inputx)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = np.ones((4, 2, 3, 4)).astype(np.float32)
    expected[0][0][0][0] = 0.0
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_update
    net = TestScatterFuncNet("update", inputx)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]],
                         [[[72, 73, 74, 75], [76, 77, 78, 79], [80, 81, 82, 83]],
                          [[84, 85, 86, 87], [88, 89, 90, 91], [92, 93, 94, 95]]],
                         [[[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]],
                          [[36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]],
                         [[[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59]],
                          [[60, 61, 62, 63], [64, 65, 66, 67], [68, 69, 70, 71]]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_func_dynamic_indices():
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.int32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    indices_dy = Tensor(shape=(2, None), dtype=mindspore.int32)
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.int32))

    # scatter_max
    net = TestScatterFuncNet("max", inputx)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = np.array([[[[1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]],
                         [[[72, 73, 74, 75], [76, 77, 78, 79], [80, 81, 82, 83]],
                          [[84, 85, 86, 87], [88, 89, 90, 91], [92, 93, 94, 95]]],
                         [[[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]],
                          [[36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]],
                         [[[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59]],
                          [[60, 61, 62, 63], [64, 65, 66, 67], [68, 69, 70, 71]]]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_min
    net = TestScatterFuncNet("min", inputx)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = np.ones((4, 2, 3, 4)).astype(np.int32)
    expected[0][0][0][0] = 0
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    # scatter_update
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.float32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    indices_dy = Tensor(shape=(2, None), dtype=mindspore.int32)
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.float32))
    net = TestScatterFuncNet("update", inputx)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]],
                         [[[72, 73, 74, 75], [76, 77, 78, 79], [80, 81, 82, 83]],
                          [[84, 85, 86, 87], [88, 89, 90, 91], [92, 93, 94, 95]]],
                         [[[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]],
                          [[36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]],
                         [[[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59]],
                          [[60, 61, 62, 63], [64, 65, 66, 67], [68, 69, 70, 71]]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


class TestScatterFuncGradNet(nn.Cell):
    def __init__(self, network):
        super(TestScatterFuncGradNet, self).__init__()
        self.grad = ops.GradOperation(
            get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, indices, updates, dout):
        out = self.grad(self.network, self.params)(indices, updates, dout)
        return out


def scatter_func_grad(nptype):
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(nptype)))
    indices = Tensor(
        np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(nptype))
    dout = Tensor(np.flip(np.arange(0, 12).reshape((3, 4)).astype(nptype)))

    indices_expected = np.array(
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]).astype(nptype)
    updates_expected = np.array(
        [
            [
                [[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]],
                [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]
            ],
            [
                [[11, 10, 9, 8], [11, 10, 9, 8], [11, 10, 9, 8]],
                [[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]]
            ]
        ]).astype(nptype)

    # scatter_max
    net = TestScatterFuncGradNet(TestScatterFuncNet("max", inputx))
    output = net(indices, updates, dout)
    indices_grad = output[0][0]
    updates_grad = output[0][1]
    np.testing.assert_array_almost_equal(indices_grad, indices_expected)
    np.testing.assert_array_almost_equal(updates_grad, updates_expected)

    # scatter_min
    net = TestScatterFuncGradNet(TestScatterFuncNet("min", inputx))
    output = net(indices, updates, dout)
    indices_grad = output[0][0]
    updates_grad = output[0][1]
    np.testing.assert_array_almost_equal(indices_grad, indices_expected)
    np.testing.assert_array_almost_equal(updates_grad, updates_expected)

    # scatter_update
    if nptype not in (np.float16, np.float32):
        return
    net = TestScatterFuncGradNet(TestScatterFuncNet("update", inputx))
    output = net(indices, updates, dout)
    indices_grad = output[0][0]
    updates_grad = output[0][1]
    np.testing.assert_array_almost_equal(indices_grad, indices_expected)
    np.testing.assert_array_almost_equal(updates_grad, updates_expected)


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


class NestVmapNet(nn.Cell):
    def __init__(self, net, inputx, in_axes, out_axes):
        super(NestVmapNet, self).__init__()
        self.net = net
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        return vmap(vmap(self.net, self.in_axes, self.out_axes), self.in_axes, self.out_axes)(
            self.inputx, indices, updates)


def scatter_func_indices_vmap():
    inputx = Parameter(Tensor(np.array(
        [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]
    ).astype(np.float32)), name="inputx")
    indices = Tensor(np.array(
        [[[0, 1], [1, 1]], [[0, 1], [0, 1]], [[1, 1], [1, 0]]]).astype(np.int32))
    updates = Tensor(
        np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]).astype(np.float32))

    # scatter_update
    output = VmapNet(ScatterFuncVmapNet("update"), inputx,
                     (0, 0, None), 0)(indices, updates)
    expected = np.array(
        [[[1, 1, 1], [4, 4, 4]], [[3, 3, 3], [4, 4, 4]], [[4, 4, 4], [3, 3, 3]]]
    ).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_func_updates_vmap():
    inputx = Parameter(Tensor(np.array(
        [[0.1, 1.0, 2.2], [3.0, 4.3, 5.5]]).astype(np.float32)), name="inputx")
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.array([[1.0, 0.1], [1.2, 1.3]]).astype(np.float32))

    # scatter_update
    output = VmapNet(ScatterFuncVmapNet("update"), inputx,
                     (0, None, 0), 0)(indices, updates)
    expected = np.array([[1.0, 0.1, 2.2], [1.2, 1.3, 5.5]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_func_updates_nest_vmap():
    inputx = Parameter(Tensor(np.array(
        [
            [[0.1, 1.0, 2.2], [3.0, 4.3, 5.5]],
            [[0.1, 1.0, 2.2], [3.0, 4.3, 5.5]]
        ]
    ).astype(np.float32)), name="inputx")
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.array(
        [
            [[1.0, 0.1], [1.2, 1.3]],
            [[1.0, 0.1], [1.2, 1.3]]
        ]
    ).astype(np.float32))
    expected = np.array(
        [
            [[1.0, 0.1, 2.2], [1.2, 1.3, 5.5]],
            [[1.0, 0.1, 2.2], [1.2, 1.3, 5.5]]
        ]
    ).astype(np.float32)

    # scatter_update
    output = NestVmapNet(ScatterFuncVmapNet("update"), inputx,
                         (0, None, 0), 0)(indices, updates)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_forward_float16():
    """
    Feature: test scatter_func forward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_forward(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_forward(np.float16)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_forward_float32():
    """
    Feature: test scatter_func forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_forward(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_forward(np.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_forward_int32():
    """
    Feature: test scatter_func forward.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_forward(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_forward(np.int32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_dynamic_indices():
    """
    Feature: test scatter_func dynamic shape.
    Description: indices is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_dynamic_indices()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_dynamic_indices()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_dynamic_updates():
    """
    Feature: test scatter_func dynamic shape.
    Description: updates is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_dynamic_updates()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_dynamic_updates()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_grad_float16():
    """
    Feature: test scatter_func grad.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_grad(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_grad(np.float16)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_grad_float32():
    """
    Feature: test scatter_func grad.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_grad(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_grad(np.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_grad_int32():
    """
    Feature: test scatter_func grad.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_grad(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_grad(np.int32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_indices_vmap():
    """
    Feature: test scatter_func vmap.
    Description: in_axes: (0, 0, None).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_indices_vmap()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_indices_vmap()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_updates_vmap():
    """
    Feature: test scatter_func vmap.
    Description: in_axes: (0, None, 0).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_updates_vmap()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_updates_vmap()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_func_updates_nest_vmap():
    """
    Feature: test scatter_func nest vmap.
    Description: in_axes: (0, None, 0).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_func_updates_nest_vmap()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_func_updates_nest_vmap()
