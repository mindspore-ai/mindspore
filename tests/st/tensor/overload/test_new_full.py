"""Tests for Tensor.new_full and backend compatibility."""

import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.graph.api import _pynative_executor


class Net(nn.Cell):
    def construct(self, x, size, fill_value, dtype):
        return x.new_full(size, fill_value, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard', mem_peak=4.0)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, mstype.int32])
def test_new_full(mode, dtype):
    """
    Feature: tensor.new_full()
    Description: Verify the result of tensor.new_full
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.arange(4).reshape((2, 2)), dtype=mstype.float32)
    fill_value = 1
    output = net(x, (3, 3), fill_value, dtype)
    expected = np.ones((3, 3))
    if dtype is None:
        assert output.dtype == mstype.float32
    else:
        assert output.dtype == dtype
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard')
def test_new_full_cpu_graph_unsupported():
    """
    Feature: Tensor.new_full on CPU in graph mode.
    Description: Verify that CPU graph compilation rejects NewFull.
    Expectation: Raise RuntimeError for the unsupported operator.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net()
    x = Tensor([1.0, 2.0], dtype=mstype.float32)
    with pytest.raises(RuntimeError, match=r"Unsupported op \[NewFull\] on CPU"):
        net(x, (2, 3), 1, None)
        _pynative_executor.sync()
