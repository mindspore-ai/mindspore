# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import Tensor


class kwargs_Net(nn.Cell):
    def construct(self, x, dim, index):
        return x.index_select(dim=dim, index=index)


class args_Net(nn.Cell):
    def construct(self, x, dim, index):
        return x.index_select(dim, index)


class deprecated_Net(nn.Cell):
    def construct(self, x, axis, index):
        return x.index_select(axis=axis, index=index)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_tensor_index_select_pynative_kwargs():
    """
    Feature: tensor.index_select
    Description: Verify the result of tensor.index_select.
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = kwargs_Net()
    x = Tensor(np.arange(16).astype(np.float32).reshape(2, 2, 4))
    index = Tensor([0,], ms.int32)
    output_x = net(x, 1, index)
    expect_output = Tensor([[[0, 1, 2, 3]], [[8, 9, 10, 11]]], ms.int32)
    assert np.allclose(output_x.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_tensor_index_select_graph_kwargs():
    """
    Feature: tensor.index_select
    Description: Verify the result of tensor.index_select.
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    net = kwargs_Net()
    x = Tensor(np.arange(16).astype(np.float32).reshape(2, 2, 4))
    index = Tensor([0,], ms.int32)
    output_x = net(x, 1, index)
    expect_output = Tensor([[[0, 1, 2, 3]], [[8, 9, 10, 11]]], ms.int32)
    assert np.allclose(output_x.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_index_select_args(mode):
    """
    Feature: tensor.index_select
    Description: Verify the result of tensor.index_select.
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})
    net = args_Net()
    x = Tensor(np.arange(16).astype(np.float32).reshape(2, 2, 4))
    index = Tensor([0,], ms.int32)
    output_x = net(x, 1, index)
    expect_output = Tensor([[[0, 1, 2, 3]], [[8, 9, 10, 11]]], ms.int32)
    assert np.allclose(output_x.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_deprecated_tensor_index_select(mode):
    """
    Feature: tensor.index_select
    Description: Verify the result of tensor.index_select.
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})
    net = deprecated_Net()
    x = Tensor(np.arange(16).astype(np.float32).reshape(2, 2, 4))
    index = Tensor([0,], ms.int32)
    output_x = net(x, 1, index)
    expect_output = Tensor([[[0, 1, 2, 3]], [[8, 9, 10, 11]]], ms.int32)
    assert np.allclose(output_x.asnumpy(), expect_output.asnumpy())
