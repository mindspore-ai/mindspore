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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn


class TensorSplitNet(nn.Cell):
    def construct(self, x, indices_or_sections, axis=0):
        out = x.tensor_split(indices_or_sections, axis)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_tensor_split_int(mode):
    """
    Feature: tensor_split
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is int.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = TensorSplitNet()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = 3
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_tensor_split_list(mode):
    """
    Feature: tensor_split
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is tuple(int) or tuple(int).
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = TensorSplitNet()
    a = np.array(np.arange(10).reshape((5, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = [2, 4]
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


class VSplitNet(nn.Cell):
    def construct(self, x, indices_or_sections):
        out = x.vsplit(indices_or_sections)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_vsplit_int(mode):
    """
    Feature: vsplit
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is int.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = VSplitNet()
    a = np.arange(20).reshape((10, 2))
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = 3
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections, axis=0)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_vsplit_list(mode):
    """
    Feature: vsplit
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is tuple(int) or tuple(int).
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = VSplitNet()
    a = np.array(np.arange(10).reshape((5, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = [2, 4]
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections, axis=0)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


class HSplitNet(nn.Cell):
    def construct(self, x, indices_or_sections):
        out = x.hsplit(indices_or_sections)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_hsplit_int(mode):
    """
    Feature: hsplit
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is int.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = HSplitNet()
    a = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = 3
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections, axis=1)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_hsplit_list(mode):
    """
    Feature: hsplit
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is tuple(int) or tuple(int).
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = HSplitNet()
    a = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = [2, 4]
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections, axis=1)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


class DSplitNet(nn.Cell):
    def construct(self, x, indices_or_sections):
        out = x.dsplit(indices_or_sections)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_dsplit_int(mode):
    """
    Feature: dsplit
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is int.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = DSplitNet()
    a = np.array(np.arange(20).reshape((1, 2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = 3
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections, axis=2)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_dsplit_list(mode):
    """
    Feature: dsplit
    Description: Verify the result of tensor_split when the type of `indices_or_sections` is tuple(int) or tuple(int).
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = DSplitNet()
    a = np.array(np.arange(20).reshape((1, 2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    indices_or_sections = [2, 4]
    out = net(x, indices_or_sections)
    expect = np.array_split(a, indices_or_sections, axis=2)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


class SplitNet(nn.Cell):
    def construct(self, x, split_size_or_sections, axis=0):
        out = x.split(split_size_or_sections, axis)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_split_int(mode):
    """
    Feature: split
    Description: Verify the result of split.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = SplitNet()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = 5
    out = net(x, split_size_or_sections)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_split_list(mode):
    """
    Feature: split
    Description: Verify the result of split.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = SplitNet()
    a = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = [2, 3, 5]
    out = net(x, split_size_or_sections, axis=1)
    expect = [np.array([[0, 1], [10, 11]], dtype=np.float32),
              np.array([[2, 3, 4], [12, 13, 14]], dtype=np.float32),
              np.array([[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)
