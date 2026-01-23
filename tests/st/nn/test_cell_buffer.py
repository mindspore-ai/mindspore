# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
""" test cell buffers"""
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import torch
from torch import nn as nn_torch
from torch import Tensor as Tensor_torch
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.common.parameter import _Buffer


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.buffer_0 = _Buffer(Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer('buffer_1', Tensor(np.array([3, 4, 5]).astype(np.float32)))
        self.param_0 = ms.Parameter(Tensor(np.array([10, 20, 30]).astype(np.float32)))

    def construct(self, x):
        return (x + self.buffer_0) * self.buffer_1 + self.param_0


class ComplexNet(nn.Cell):
    def __init__(self, sub_net):
        super().__init__()
        self.sub_net = sub_net
        self.buffer_0 = _Buffer(Tensor(np.array([6, 6, 6]).astype(np.float32)))
        self.register_buffer('buffer_1', Tensor(np.array([7, 7, 7]).astype(np.float32)))
        self.param_0 = ms.Parameter(Tensor(np.array([8, 8, 8]).astype(np.float32)))

    def construct(self, x):
        return self.sub_net(x) + self.buffer_0 - self.buffer_1 - self.param_0


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_buffer_basic(mode):
    """
    Feature: test buffer of simple net
    Description: Verify the result of buffer of simple net
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = SimpleNet()
    net.register_buffer("buffer_2", Tensor(np.array([6, 7, 8]).astype(np.float32)))
    tensor1 = Tensor(np.array([1, 2, 3]).astype(np.float32))
    tensor2 = Tensor(np.array([3, 4, 5]).astype(np.float32))
    tensor3 = Tensor(np.array([6, 7, 8]).astype(np.float32))
    my_input = Tensor(np.array([2, 2, 2]).astype(np.float32))
    assert len(net._buffers) == 3  # pylint: disable=W0212
    my_names = [k for k, _ in net.named_buffers()]
    expect_names = ["buffer_0", "buffer_1", "buffer_2"]
    my_buffers = [v for _, v in net.named_buffers()]
    my_buffers2 = list(net.buffers())
    expect_buffers = [tensor1, tensor2, tensor3]
    for i, name in enumerate(my_names):
        assert expect_names[i] == name
    for i, buffer in enumerate(my_buffers):
        assert np.allclose(expect_buffers[i].asnumpy(), buffer.asnumpy())
    for i, buffer in enumerate(my_buffers2):
        assert np.allclose(expect_buffers[i].asnumpy(), buffer.asnumpy())
    my_output = net(my_input)
    expect_output = Tensor(np.array([19, 36, 55]).astype(np.float32))
    assert np.allclose(my_output.asnumpy(), expect_output.asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_complex_net_buffer(mode):
    """
    Feature: test buffer of complex net
    Description: Verify the result of buffer of complex net
    Expectation: success
    """
    ms.set_context(mode=mode)
    sub_net = SimpleNet()
    net0 = ComplexNet(sub_net)
    net0.register_buffer("buffer_2", Tensor(np.array([10, 10, 10]).astype(np.float32)))
    tensor1 = Tensor(np.array([6, 6, 6]).astype(np.float32))
    tensor2 = Tensor(np.array([7, 7, 7]).astype(np.float32))
    tensor3 = Tensor(np.array([10, 10, 10]).astype(np.float32))
    tensor4 = Tensor(np.array([1, 2, 3]).astype(np.float32))
    tensor5 = Tensor(np.array([3, 4, 5]).astype(np.float32))
    my_input = Tensor(np.array([2, 2, 2]).astype(np.float32))
    assert len(net0._buffers) == 3  # pylint: disable=W0212
    my_names = [k for k, _ in net0.named_buffers()]
    expect_names = ["buffer_0", "buffer_1", "buffer_2", "sub_net.buffer_0", "sub_net.buffer_1"]
    my_buffers = [v for _, v in net0.named_buffers()]
    my_buffers2 = list(net0.buffers())
    expect_buffers = [tensor1, tensor2, tensor3, tensor4, tensor5]
    for i, name in enumerate(my_names):
        assert expect_names[i] == name
    for i, buffer in enumerate(my_buffers):
        assert np.allclose(expect_buffers[i].asnumpy(), buffer.asnumpy())
    for i, buffer in enumerate(my_buffers2):
        assert np.allclose(expect_buffers[i].asnumpy(), buffer.asnumpy())
    my_output = net0(my_input)
    expect_output = Tensor(np.array([10, 27, 46]).astype(np.float32))
    assert np.allclose(my_output.asnumpy(), expect_output.asnumpy())
    sub_net_buffer = net0.get_buffer("sub_net.buffer_1")
    assert np.allclose(sub_net_buffer.asnumpy(), tensor5.asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_buffer_exception(mode):
    """
    Feature: test exception of buffer
    Description: Verify the result of buffer exception
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = SimpleNet()
    with pytest.raises(TypeError, match='buffer name should be a string'):
        net.register_buffer(2, Tensor(np.array([6, 7, 8]).astype(np.float32)))
    with pytest.raises(KeyError, match='contain "."'):
        net.register_buffer("buffer.name", Tensor(np.array([6, 7, 8]).astype(np.float32)))
    with pytest.raises(AttributeError, match='has no attribute'):
        net.get_buffer("buffer_5")


class Net(nn.Cell):
    """MindSpore network for testing buffer registration and named_buffers functionality"""
    def __init__(self, with_hook_buffers=False):
        super().__init__()
        self.register_buffer("buffer1", Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer1", Tensor(np.array([4, 5, 6]).astype(np.float32)))
        self.register_buffer("buffer2", Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer3", Tensor(np.array([1, 2, 3, 4]).astype(np.float32)), persistent=True)
        self.register_buffer("buffer4", Tensor(np.array([1, 2, 3, 4, 5]).astype(np.float32)), persistent=False)
        self.register_buffer("buffer5", None, persistent=False)
        self.register_buffer("buffer6", None, persistent=True)
        self.tmp = Tensor(np.array([2, 2, 3]).astype(np.float32))
        self.register_buffer("buffer7", self.tmp)
        self.register_buffer("buffer8", self.tmp)
        if with_hook_buffers:
            self.register_buffer("normal_buffer", Tensor(np.array([4, 5, 6]).astype(np.float32)))
            self.register_buffer("special_buffer", Tensor(np.array([1, 2, 3]).astype(np.float32)))

    def construct(self, x):
        return x + self.buffer2


class NetTorch(nn_torch.Module):
    """PyTorch network for comparison with MindSpore buffer behavior"""
    def __init__(self, with_hook_buffers=False):
        super().__init__()
        self.register_buffer("buffer1", Tensor_torch(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer1", Tensor_torch(np.array([4, 5, 6]).astype(np.float32)))
        self.register_buffer("buffer2", Tensor_torch(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer3", Tensor_torch(np.array([1, 2, 3, 4]).astype(np.float32)), persistent=True)
        self.register_buffer("buffer4", Tensor_torch(np.array([1, 2, 3, 4, 5]).astype(np.float32)),
                             persistent=False)
        self.register_buffer("buffer5", None, persistent=False)
        self.register_buffer("buffer6", None, persistent=True)
        self.tmp = Tensor_torch(np.array([2, 2, 3]).astype(np.float32))
        self.register_buffer("buffer7", self.tmp)
        self.register_buffer("buffer8", self.tmp)
        if with_hook_buffers:
            self.register_buffer("normal_buffer", Tensor_torch(np.array([4, 5, 6]).astype(np.float32)))
            self.register_buffer("special_buffer", Tensor_torch(np.array([1, 2, 3]).astype(np.float32)))

    def forward(self, x):
        return x + self.buffer2


@arg_mark(
    plat_marks=['cpu_linux', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=0.01)
def test_named_buffers_with_one_cell():
    """
    Feature: test named_buffers with parameters
    Description: Verify the result of named_buffers with prefix, recurse and remove_duplicate parameters
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    net_torch = NetTorch()
    torch_name_buffers = net_torch.named_buffers()
    for k, v in net.named_buffers():
        torch_tmp = next(torch_name_buffers)
        assert k == torch_tmp[0], \
            f"Buffer name mismatch: MindSpore='{k}', PyTorch='{torch_tmp[0]}'"
        assert np.array_equal(v, torch_tmp[1]), \
            f"Buffer '{k}' values not equal: MindSpore buffer: {v}, PyTorch buffer: {torch_tmp[1]}"
    with pytest.raises(StopIteration):
        next(torch_name_buffers)
    torch_name_buffers = net_torch.named_buffers(prefix="test")
    for k, v in net.named_buffers(prefix="test"):
        torch_tmp = next(torch_name_buffers)
        assert k == torch_tmp[0], \
            f"[prefix='test']buffer name mismatch: MindSpore='{k}', PyTorch='{torch_tmp[0]}'"
        assert np.array_equal(v, torch_tmp[1]), \
            f"[prefix='test']buffer '{k}' values not equal: MindSpore buffer: {v}, PyTorch buffer: {torch_tmp[1]}"
    with pytest.raises(StopIteration):
        next(torch_name_buffers)
    torch_name_buffers = net_torch.named_buffers(prefix="test", recurse=True)
    for k, v in net.named_buffers(prefix="test", recurse=True):
        torch_tmp = next(torch_name_buffers)
        assert k == torch_tmp[0], \
            f"[Prefix='test', Recurse=True] Buffer name mismatch: MindSpore='{k}', PyTorch='{torch_tmp[0]}'"
        assert np.array_equal(v, torch_tmp[1]), \
            f"[Prefix='test', Recurse=True] Buffer '{k}' values not equal: \
               MindSpore buffer: {v}, PyTorch buffer: {torch_tmp[1]}"
    with pytest.raises(StopIteration):
        next(torch_name_buffers)
    torch_name_buffers = net_torch.named_buffers(prefix="test", remove_duplicate=True)
    for k, v in net.named_buffers(prefix="test", remove_duplicate=True):
        torch_tmp = next(torch_name_buffers)
        assert k == torch_tmp[0], \
            f"[Prefix='test', RemoveDuplicate=True] Buffer name mismatch: MindSpore='{k}', PyTorch='{torch_tmp[0]}'"
        assert np.array_equal(v, torch_tmp[1]), \
            f"[prefix='test', RemoveDuplicate=True] Buffer '{k}' values not equal: \
              MindSpore buffer: {v}, PyTorch buffer: {torch_tmp[1]}"
    with pytest.raises(StopIteration):
        next(torch_name_buffers)


def modify_buffer_based_on_name2(cell, name, buffer):  # pylint: disable=unused-argument
    """Hook function to modify buffer based on name"""
    if name == "special_buffer":
        return buffer * 2
    return buffer


@arg_mark(
    plat_marks=['cpu_linux', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=2.00)
def test_register_buffer_hook():
    """
    Feature: test register_buffer hook with one cell
    Description: Verify register_buffer hook can modify buffer during registration
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    handler_ms = ms.nn.cell.register_cell_buffer_registration_hook(modify_buffer_based_on_name2)
    handler_torch = torch.nn.modules.module.register_module_buffer_registration_hook(modify_buffer_based_on_name2)
    net = Net(with_hook_buffers=True)
    net_torch = NetTorch(with_hook_buffers=True)
    torch_name_buffers = net_torch.named_buffers()
    for k, v in net.named_buffers():
        torch_tmp = next(torch_name_buffers)
        assert k == torch_tmp[0], \
            f"[With Hook Active] Buffer name mismatch: MindSpore='{k}', PyTorch='{torch_tmp[0]}'"
        assert np.array_equal(v, torch_tmp[1]), \
            f"[With Hook Active] Buffer '{k}' values not equal: MindSpore buffer: {v}, PyTorch buffer: {torch_tmp[1]}"

    x = np.array([1, 2, 5]).astype(np.float32)
    net(Tensor(x))
    net_torch(Tensor_torch(x))
    torch_name_buffers = net_torch.named_buffers()
    for k, v in net.named_buffers():
        torch_tmp = next(torch_name_buffers)
        assert k == torch_tmp[0], \
            f"[After Forward] Buffer name mismatch: MindSpore='{k}', PyTorch='{torch_tmp[0]}'"
        assert np.array_equal(v, torch_tmp[1]), \
            f"[After Forward] Buffer '{k}' values not equal: MindSpore buffer: {v}, PyTorch buffer: {torch_tmp[1]}"
    handler_ms.remove()
    handler_torch.remove()
