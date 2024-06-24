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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.common.api import _pynative_executor
from tests.st.utils import test_utils


@test_utils.run_with_cell
def broadcast_to_forward_func(x, shape):
    return ms.ops.auto_generate.broadcast_to(x, shape)

@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_broadcast(context_mode):
    """
    Feature: pyboost function.
    Description: test function broadcast_to forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)

    shape = (4, 5, 2, 3, 4, 5, 6)
    x_np = np.random.rand(2, 3, 1, 5, 1).astype(np.float32)
    output = P.BroadcastTo(shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 5, 7, 4, 5, 6)
    x_np = np.arange(20).reshape((4, 5, 1)).astype(np.int32)
    output = P.BroadcastTo(shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (8, 5, 7, 4, 5, 6)
    x_np = np.arange(24).reshape((1, 4, 1, 6)).astype(np.bool)
    output = P.BroadcastTo(shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 4, 5, 2, 3, 4, 5, 7)
    x_np = np.random.rand(2, 3, 1, 5, 1).astype(np.float16)
    output = P.BroadcastTo(shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 4, 5, 6)
    x_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = P.BroadcastTo(shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float16)
    output = P.BroadcastTo(shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (2, 3, 4, 5)
    x1_np = np.random.rand(4, 5).astype(np.float32)
    output = P.BroadcastTo(shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, shape)
    assert np.allclose(output.asnumpy(), expect)


def broadcast_to_dtype(dtype):
    """
    Basic function to test data type of BroadcastTo.
    """
    shape = (2, 3, 4, 5)
    x1_np = np.random.rand(4, 5).astype(dtype)
    output = P.BroadcastTo(shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, shape)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_to_dtype(context_mode):
    """
    Feature: Test supported data types of BroadCastTo.
    Description: all data types
    Expectation: success.
    """
    context.set_context(mode=context_mode)
    types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.complex64, np.complex128]
    for dtype in types:
        broadcast_to_dtype(dtype=dtype)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_dyn_init(context_mode):
    """
    Feature: pyboost function.
    Description: Test running the op with -1's in the init shape to support varied inputs.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)

    ms_shape = (-1, -1, 5, 6)
    np_shape = (3, 4, 5, 6)
    x_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    output = P.BroadcastTo(ms_shape)(Tensor(x_np))
    expect = np.broadcast_to(x_np, np_shape)
    assert np.allclose(output.asnumpy(), expect)

    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float16)
    output = P.BroadcastTo(ms_shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, np_shape)
    assert np.allclose(output.asnumpy(), expect)

    ms_shape = (2, 3, -1, -1)
    np_shape = (2, 3, 4, 5)
    x1_np = np.random.rand(4, 5).astype(np.float32)
    output = P.BroadcastTo(ms_shape)(Tensor(x1_np))
    expect = np.broadcast_to(x1_np, np_shape)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_dyn_invalid_init(context_mode):
    """
    Feature: pyboost function.
    Description: Test running the op with -1's in the init shape in incorrect positions.
    Expectation: Expected to fail.
    """
    context.set_context(mode=context_mode)
    ms_shape = (2, -1, 4, 5)
    x_np = np.random.rand(4, 5).astype(np.float32)
    with pytest.raises(ValueError):
        P.BroadcastTo(ms_shape)(Tensor(x_np))
        _pynative_executor.sync()

    ms_shape = (-1, 1, -1, -1)
    x_np = np.random.rand(4, 5).astype(np.float32)
    with pytest.raises(ValueError):
        P.BroadcastTo(ms_shape)(Tensor(x_np))
        _pynative_executor.sync()


class BroadcastToNet(Cell):
    """
    Construct of dynamic input for BroadcastTo.
    """

    def __init__(self, shape):
        super().__init__()
        self.broadcastto = P.BroadcastTo(shape)

    def construct(self, input_x):
        return self.broadcastto(input_x)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_to_dynamic_shape(context_mode):
    """
    Feature: Test dynamic shape of BroadcastTo operator
    Description: dynamic input
    Expectation: success.
    """
    context.set_context(mode=context_mode)
    shape = (2, 2, 3)
    input_x_np = np.random.randn(2, 3).astype(np.float32)
    input_x = Tensor(input_x_np)
    input_dyn = Tensor(shape=[None, 3], dtype=input_x.dtype)
    broadcast_to_net = BroadcastToNet(shape)
    broadcast_to_net.set_inputs(input_dyn)
    output = broadcast_to_net(input_x)
    expect = np.broadcast_to(input_x_np, shape)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_exception(context_mode):
    """
    Feature: Test invalid input and target shape in of BroadcastTo.
    Description: target shape is empty, but input shape is not empty.
    Expectation: the result match with expected result.
    """
    with pytest.raises(Exception) as info:
        context.set_context(mode=context_mode)
        shape = (0,)
        x_np = np.random.randint(1, 4)
        P.BroadcastTo(shape)(Tensor(x_np))
        assert "ValueError: For 'BroadcastTo', each dimension pair, input_x shape and target shape must be equal or \
        input dimension is 1 or target dimension is -1. But got input_x shape: [const vector][], target shape: \
        [const vector][0]." in str(info.value)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_broadcast_to_forward(mode):
    """
    Feature: Ops.
    Description: test broadcast_to.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    shape = (128, 1, 77, 77)
    x_np = np.arange(128).reshape((128, 1, 1, 1)).astype(np.float32)
    x = Tensor(x_np)
    out = broadcast_to_forward_func(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(out.asnumpy(), expect)
