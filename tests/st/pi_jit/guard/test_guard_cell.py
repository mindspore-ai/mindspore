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
"""Test guard for Cell"""
import sys
import pytest
import mindspore as ms
from mindspore import Tensor, context, jit, nn, ops

from tests.mark_utils import arg_mark
from tests.st.pi_jit.conftest import run_in_subprocess
from ..share.utils import match_array, assert_no_graph_break

context.set_context(mode=context.PYNATIVE_MODE)


@run_in_subprocess({'GLOG_v': '1', 'MS_SUBMODULE_LOG_v': '{PI:0}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_Cell_1():
    """
    Feature: Test guard for nn.Cell.
    Description: Call different Cell instance, should recompile.
    Expectation: Guard failed, should recompile.
    """

    class Net(nn.Cell):
        def construct(self, x: Tensor, y: Tensor):
            return ops.matmul(x, y)

    pynative_net = Net()
    jit_net1 = Net()
    jit_net2 = Net()
    jit_net1.construct = jit(jit_net1.construct, capture_mode='bytecode')
    jit_net2.construct = jit(jit_net2.construct, capture_mode='bytecode')

    x = ops.randn(2, 4)
    y = ops.randn(4, 2)
    o1 = pynative_net(x, y)

    o2 = jit_net1(x, y)
    match_array(o1, o2, error=7)
    assert_no_graph_break(jit_net1.construct, call_count=1)

    o3 = jit_net2(x, y)
    match_array(o1, o3, error=7)

    x = ops.randn(2, 4)
    y = ops.randn(4, 2)
    o4 = pynative_net(x, y)

    o5 = jit_net1(x, y)
    match_array(o4, o5, error=7)

    o6 = jit_net2(x, y)
    match_array(o4, o6, error=7)
    # not need guard unused self, grad is guard by `pynative_executor.grad`
    assert_no_graph_break(jit_net2.construct, call_count=4)


class MLP(nn.Cell):
    def __init__(self, layers: int, dim: int, has_bias=False):
        super(MLP, self).__init__()
        self.layers = nn.CellList([nn.Dense(dim, dim, has_bias=has_bias) for _ in range(layers)])

    if sys.version_info >= (3, 8):
        def construct(self, x: Tensor):
            for i in range(len(self.layers)):
                x = self.layers[i](x)
            return x
    else:
        # In python 3.7, SETUP_LOOP bytecode is not supported by pijit, so for-loop will cause graph break.
        def construct(self, x: Tensor):
            n = len(self.layers)
            x = self.layers[0](x)
            x = self.layers[1](x)
            if n > 2:
                x = self.layers[2](x)
            return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_CellList_1():
    """
    Feature: Test guard for CellList.
    Description: Do not modify CellList, only change its input.
    Expectation: Guard success, should not recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    o3 = net(2 * x)
    match_array(o2 * 2, o3)
    assert_no_graph_break(net.construct, call_count=2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_guard_for_CellList_2():
    """
    Feature: Test guard for CellList.
    Description: Reset to a new CellList, but the Cells in it are all the same with the old CellList.
    Expectation: Guard failed, should recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    layers = [net.layers[i] for i in range(len(net.layers))]
    new_celllist = nn.CellList(layers)
    net.layers = new_celllist
    o3 = net(x)
    match_array(o2, o3)
    assert_no_graph_break(net.construct, call_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_guard_for_CellList_3():
    """
    Feature: Test guard for CellList.
    Description: Modify the self.training property of the CellList.
    Expectation: Guard failed, should recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    net.layers.training = not net.layers.training
    o3 = net(x)
    match_array(o2, o3)
    assert_no_graph_break(net.construct, call_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_guard_for_CellList_4():
    """
    Feature: Test guard for CellList.
    Description: Modify the self.require_grad property of the CellList.
    Expectation: Guard failed, should recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    net.layers.requires_grad = not net.layers.requires_grad
    o3 = net(x)
    match_array(o2, o3)
    assert_no_graph_break(net.construct, call_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_CellList_5():
    """
    Feature: Test guard for CellList.
    Description: Append a new Cell to the CellList.
    Expectation: Guard failed, should recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    relu = nn.ReLU()
    net.layers.append(relu)
    o3 = net(x)
    match_array(relu(o2), o3)
    assert_no_graph_break(net.construct, call_count=1)


class AddBias(nn.Cell):
    def __init__(self, bias: int):
        super().__init__()
        self._bias = bias

    def construct(self, x: Tensor):
        return x + self._bias


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_CellList_6():
    """
    Feature: Test guard for CellList.
    Description: Delete a Cell from the CellList.
    Expectation: Guard failed, should recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    net.layers.append(AddBias(bias=1))
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    del net.layers[-1]
    o3 = net(x)
    match_array(o2, o3 + 1)
    assert_no_graph_break(net.construct, call_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_CellList_7():
    """
    Feature: Test guard for CellList.
    Description: Replace a Cell in the CellList to a new Cell.
    Expectation: Guard failed, should recompile.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MLP(layers=2, dim=4)
    net.layers.append(AddBias(bias=1))
    x = Tensor([1, 2, 3, 4], dtype=ms.float32)
    o1 = net(x)

    net.construct = jit(net.construct, capture_mode='bytecode')
    o2 = net(x)
    match_array(o1, o2)
    assert_no_graph_break(net.construct, call_count=1)

    net.layers[-1] = AddBias(bias=0)
    o3 = net(x)
    match_array(o2, o3 + 1)
    assert_no_graph_break(net.construct, call_count=1)
