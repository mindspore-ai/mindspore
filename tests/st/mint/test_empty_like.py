# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=unused-variable
"""Tests for mint.empty_like: shape/dtype/device, pin_memory, and dynamic shape."""
import pytest
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import mint, Tensor
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


class Net(ms.nn.Cell):
    def construct(self, x, dtype=None, device=None, pin_memory=False):
        return mint.empty_like(x, dtype=dtype, device=device, pin_memory=pin_memory)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_like_normal1(mode):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    net = Net()
    y = net(input_tensor)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float32)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_like_normal2(mode):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    net = Net()
    y = net(input_tensor, dtype=mstype.float64)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float64)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_like_normal3(mode):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    net = Net()
    y = net(input_tensor, dtype=mstype.float64, device="Ascend")
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float64)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_empty_like_normal4():
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    net = Net()
    y = net(input_tensor, dtype=mstype.float64, device="CPU")
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float64)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_like_normal5(mode):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = mint.ones((1, 2, 3))

    net = Net()
    y = net(input_tensor)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, input_tensor.dtype)

def empty_like_forward_func_dyn_test(input_tensor, dtype=None):
    y = Net()(input_tensor, dtype=dtype)
    return y.shape

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_empty_like_dynamic_shape():
    """
    Feature: Test empty_like with dynamic shape.
    Description: call mint.empty_like with valid input and index.
    Expectation: return the correct value.
    """
    tensor_1 = Tensor(np.arange(6).reshape(2, 3), dtype=mstype.float32)

    tensor_2 = Tensor(np.arange(24).reshape(2, 3, 4), dtype=mstype.float32)

    TEST_OP(empty_like_forward_func_dyn_test,
            [[tensor_1], [tensor_2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'])

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.00)
@pytest.mark.parametrize("device_name", ['npu', 'cpu', 'Ascend', 'CPU'])
def test_empty_like_device1(device_name):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    net = Net()
    y = net(input_tensor, device=device_name)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float32)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize("device_name", ['npu', 'Ascend'])
def test_empty_like_device2(device_name):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    net = Net()
    y = net(input_tensor, device=device_name)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float32)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential',
          mem_peak=2.00)
@pytest.mark.parametrize('pin_memory', [None, True, False])
def test_empty_like_pin_memory(pin_memory):
    """
    Feature: Test empty_like with pin_memory parameter.
    Description: call mint.empty_like with valid input.
    Expectation: return the correct value.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    dtype = mstype.float32

    net = Net()
    if pin_memory is None:
        y = net(input_tensor, dtype=dtype, device="CPU")
    else:
        y = net(input_tensor, dtype=dtype, device="CPU", pin_memory=pin_memory)
    if pin_memory is None:
        assert not y.is_pinned()
    elif pin_memory:
        assert y.is_pinned()
    else:
        assert not y.is_pinned()

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_empty_like_cpu_kbk_pin_memory_true_raises():
    """
    Feature: empty_like with pin_memory=True in GRAPH O0 mode on CPU device.
    Description: In GRAPH O0 mode, CPU backend should raise when pin_memory=True.
    Expectation: Raise RuntimeError with proper message.
    """
    np.random.seed(0)
    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    net = Net()

    with pytest.raises(RuntimeError, match="pin_memory"):
        _ = net(input_tensor, dtype=mstype.float32, pin_memory=True)

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_empty_like_cpu_kbk_pin_memory_false_ok():
    """
    Feature: empty_like with pin_memory=False in GRAPH O0 mode on CPU device.
    Description: In GRAPH O0 mode, CPU backend should work when pin_memory=False.
    Expectation: Run success and output has expected shape/dtype and not pinned.
    """
    np.random.seed(1)
    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    net = Net()
    y = net(input_tensor, dtype=mstype.float32, pin_memory=False)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float32)
    assert not y.is_pinned()
