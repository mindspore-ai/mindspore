import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class NetNone(nn.Cell):
    def construct(self, x, other):
        x.div_(other)
        return x


class NetFloor(nn.Cell):
    def construct(self, x, other):
        x.div_(other, rounding_mode="floor")
        return x


class NetTrunc(nn.Cell):
    def construct(self, x, other):
        x.div_(other, rounding_mode="trunc")
        return x


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_divide_none_tensor(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetNone()
    x = Tensor(np.array([1.0, 5.0, 7.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.25, 2.5, 2.5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential',
          mem_peak=2.00)
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_divide_none_scalar(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetNone()
    x = Tensor(np.array([1.0, 5.0, 7.5]), mstype.float32)
    y = 4.0
    output = net(x, y)
    expected = np.array([0.25, 1.25, 1.875], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_divide_floor(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetFloor()
    x = Tensor(np.array([1.0, 5.0, 9.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_divide_trunc(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetTrunc()
    x = Tensor(np.array([1.0, 5.0, 9.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
