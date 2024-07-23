# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops, context

import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

class GroupNormNet(nn.Cell):
    def __init__(self, num_groups, num_channels):
        super(GroupNormNet, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, dtype=ms.float32)

    def construct(self, x):
        out = self.group_norm(x)
        return out


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


class GroupNormVMapNet(nn.Cell):
    def __init__(self, net, in_axes, out_axes):
        super(GroupNormVMapNet, self).__init__()
        self.net = net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return ops.vmap(self.net, self.in_axes, self.out_axes)(input_x)


@test_utils.run_with_cell
def group_norm_forward_func(x):
    net = GroupNormNet(2, 2)
    return net(x)


@test_utils.run_with_cell
def group_norm_backward_func(x):
    return ops.grad(group_norm_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_forward(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm forward.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-1.6059085, -0.6882465, -0.6882465, 0.2294155],
                                [-1.1470774, -0.2294155, 0.6882465, 1.6059085],
                                [-0.6882465, 0.6882465, 1.1470774, 1.1470774],
                                [-0.2294155, -0.6882465, 1.6059085, -1.1470774]],
                               [[-0.08812092, 0.8518356, 0.38185734, -1.0280775],
                                [-1.0280775, -0.08812092, 0.38185734, 0.8518356],
                                [1.791792, -0.55809915, -1.4980557, -0.08812092],
                                [0.8518356, -0.08812092, 1.3218138, -1.9680339]]]]).astype(np.float32)
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6
    x = Tensor(input_x)
    output = group_norm_forward_func(x)
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_backward(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm backward.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)

    grad = np.array([[[[1, 2, 7, 1], [4, 2, 1, 3], [1, 6, 5, 2], [2, 4, 3, 2]],
                      [[9, 4, 3, 5], [1, 3, 7, 6], [5, 7, 9, 9], [1, 4, 6, 8]]]]).astype(np.float32)

    expect_output = np.array([[[[-0.69126546, -0.32903028, 1.9651246, -0.88445705],
                                [0.6369296, -0.37732816, -0.93275493, -0.11168876],
                                [-0.7878612, 1.3614, 0.8542711, -0.52222186],
                                [-0.37732816, 0.5886317, -0.11168876, -0.28073236]],

                               [[1.6447213, -0.38968924, -1.0174079, -0.55067265],
                                [-2.4305856, -1.1751484, 0.86250514, 0.5502673],
                                [0.39576983, 0.5470243, 1.1715001, 1.6447213],
                                [-1.7996241, -0.7051701, 0.7080077, 0.5437813]]]]).astype(np.float32)
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6
    gn_net = GroupNormNet(2, 2)
    gn_grad = Grad(gn_net)
    output = gn_grad(Tensor(input_x), Tensor(grad))
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_vmap(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm vmap feature.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    net = GroupNormNet(4, 4)
    output = GroupNormVMapNet(net, 0, 0)(Tensor(input_x))
    expect_output_shape = (1, 2, 4, 4)
    assert np.allclose(expect_output_shape, output.shape)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_forward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm forward with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(group_norm_forward_func)
    test_cell.set_inputs(x_dyn)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-1.6059085, -0.6882465, -0.6882465, 0.2294155],
                                [-1.1470774, -0.2294155, 0.6882465, 1.6059085],
                                [-0.6882465, 0.6882465, 1.1470774, 1.1470774],
                                [-0.2294155, -0.6882465, 1.6059085, -1.1470774]],
                               [[-0.08812092, 0.8518356, 0.38185734, -1.0280775],
                                [-1.0280775, -0.08812092, 0.38185734, 0.8518356],
                                [1.791792, -0.55809915, -1.4980557, -0.08812092],
                                [0.8518356, -0.08812092, 1.3218138, -1.9680339]]]]).astype(np.float32)
    output = test_cell(Tensor(input_x))
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_forward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm forward with dynamic rank.
    Expectation: success.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(group_norm_forward_func)
    test_cell.set_inputs(x_dyn)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-1.6059085, -0.6882465, -0.6882465, 0.2294155],
                                [-1.1470774, -0.2294155, 0.6882465, 1.6059085],
                                [-0.6882465, 0.6882465, 1.1470774, 1.1470774],
                                [-0.2294155, -0.6882465, 1.6059085, -1.1470774]],
                               [[-0.08812092, 0.8518356, 0.38185734, -1.0280775],
                                [-1.0280775, -0.08812092, 0.38185734, 0.8518356],
                                [1.791792, -0.55809915, -1.4980557, -0.08812092],
                                [0.8518356, -0.08812092, 1.3218138, -1.9680339]]]]).astype(np.float32)
    output = test_cell(Tensor(input_x))
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_backward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm backward with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(group_norm_backward_func)
    test_cell.set_inputs(x_dyn)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.zeros(shape=[1, 2, 4, 4]).astype(np.float32)
    output = test_cell(Tensor(input_x))
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_group_norm_backward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test GroupNorm backwarad with dynamic rank.
    Expectation: success.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(group_norm_backward_func)
    test_cell.set_inputs(x_dyn)
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.zeros(shape=[1, 2, 4, 4]).astype(np.float32)
    output = test_cell(Tensor(input_x))
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_ops_group_norm_dyn():
    """
    Feature: pyboost function.
    Description: test GroupNorm backwarad with dynamic rank/shape.
    Expectation: success.
    """
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    in1 = Tensor(input_x)
    in2 = Tensor(input_x)
    TEST_OP(group_norm_forward_func, [[in1], [in2]], '', disable_input_check=True, disable_yaml_check=True)
