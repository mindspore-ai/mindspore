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

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reflection_pad1d_input3d(mode):
    """
    Feature: ReflectionPad1d
    Description: Test ReflectionPad1d with 3D input.
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
    padding = (3, 1)
    net = nn.ReflectionPad1d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[3, 2, 1, 0, 1, 2, 3, 2],
                                        [7, 6, 5, 4, 5, 6, 7, 6]]]).astype(np.float32))

    assert np.array_equal(output.asnumpy(), expected_output)

    padding = 2
    expected_output = Tensor(np.array([[[2, 1, 0, 1, 2, 3, 2, 1],
                                        [6, 5, 4, 5, 6, 7, 6, 5]]]).astype(np.float32))
    net = nn.ReflectionPad1d(padding)
    output = net(x)
    assert np.array_equal(output.asnumpy(), expected_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reflection_pad1d_input2d(mode):
    """
    Feature: ReflectionPad1d
    Description: Test ReflectionPad1d with 2D input.
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).astype(np.float32))
    padding = (3, 1)
    net = nn.ReflectionPad1d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[3, 2, 1, 0, 1, 2, 3, 2],
                                       [7, 6, 5, 4, 5, 6, 7, 6]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)

    padding = 2
    expected_output = Tensor(np.array([[2, 1, 0, 1, 2, 3, 2, 1],
                                       [6, 5, 4, 5, 6, 7, 6, 5]]).astype(np.float32))
    net = nn.ReflectionPad1d(padding)
    output = net(x)
    assert np.array_equal(output.asnumpy(), expected_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reflection_pad2d_input4d(mode):
    r"""
    Feature: ReflectionPad2d
    Description: Test ReflectionPad2d with 4D input.
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]).astype(np.float32))
    padding = (1, 1, 2, 0)
    net = nn.ReflectionPad2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[[7, 6, 7, 8, 7], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                                         [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)

    padding = 2
    output = nn.ReflectionPad2d(padding)(x)
    expected_output = Tensor(np.array([[[[8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                         [2, 1, 0, 1, 2, 1, 0], [5, 4, 3, 4, 5, 4, 3],
                                         [8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                         [2, 1, 0, 1, 2, 1, 0]]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reflection_pad2d_input3d(mode):
    r"""
    Feature: ReflectionPad2d
    Description: Test ReflectionPad2d with 3D input.
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).astype(np.float32))
    padding = (1, 1, 2, 0)
    net = nn.ReflectionPad2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[7, 6, 7, 8, 7], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                                        [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]]).astype(np.float32))

    padding = 2
    output = nn.ReflectionPad2d(padding)(x)

    expected_output = Tensor(np.array([[[8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                        [2, 1, 0, 1, 2, 1, 0], [5, 4, 3, 4, 5, 4, 3],
                                        [8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                        [2, 1, 0, 1, 2, 1, 0]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reflection_pad_3d(mode):
    """
    Feature: ReflectionPad3d
    Description: Infer process of ReflectionPad3d with three type parameters.
    Expectation: success
    """
    context.set_context(mode=mode)
    arr = np.arange(8).astype(np.float32).reshape((1, 2, 2, 2))
    x = Tensor(arr)
    padding = (1, 1, 1, 0, 0, 1)
    net3d = nn.ReflectionPad3d(padding)
    output = net3d(x)
    expected_output = Tensor(np.array([[[[3, 2, 3, 2], [1, 0, 1, 0], [3, 2, 3, 2]],
                                        [[7, 6, 7, 6], [5, 4, 5, 4], [7, 6, 7, 6]],
                                        [[3, 2, 3, 2], [1, 0, 1, 0], [3, 2, 3, 2]]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)

    padding = 1
    output = nn.ReflectionPad3d(padding)(x)
    expected_output = Tensor(np.array([[[[7., 6., 7., 6.], [5., 4., 5., 4.],
                                         [7., 6., 7., 6.], [5., 4., 5., 4.]],
                                        [[3., 2., 3., 2.], [1., 0., 1., 0.],
                                         [3., 2., 3., 2.], [1., 0., 1., 0.]],
                                        [[7., 6., 7., 6.], [5., 4., 5., 4.],
                                         [7., 6., 7., 6.], [5., 4., 5., 4.]],
                                        [[3., 2., 3., 2.], [1., 0., 1., 0.],
                                         [3., 2., 3., 2.], [1., 0., 1., 0.]]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)
