import numpy as np
import pytest
from mindspore import ops, jit, context
import mindspore as ms
from tests.mark_utils import arg_mark


@jit(mode="PIJit")
def greater_forward_func(x, y):
    return ops.greater(x, y)

@jit(mode="PIJit")
def greater_backward_func(x, y):
    return ops.grad(greater_forward_func, (0,))(x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_forward():
    """
    Feature: Ops.
    Description: test op greater.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([False, True, False])
    out = greater_forward_func(x, y)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op greater.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([0, 0, 0])
    out = greater_backward_func(x, y)
    assert np.allclose(out.asnumpy(), expect_out)
