import numpy as np
import mindspore as ms
from mindspore import ops
from tests.mark_utils import arg_mark


def greater_equal_forward_func(x, y):
    return ops.greater_equal(x, y)


def greater_equal_backward_func(x, y):
    return ops.grad(greater_equal_forward_func, (0,))(x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_equal_forward():
    """
    Feature: Ops.
    Description: test op greater_equal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([True, True, False])
    out = greater_equal_forward_func(x, y)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_equal_backward():
    """
    Feature: Auto grad.
    Description: test op greater_equal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([0, 0, 0])
    grads = greater_equal_backward_func(x, y)
    assert np.allclose(grads.asnumpy(), expect_out)
