import numpy as np
import pytest
from mindspore import ops, jit, context
import mindspore as ms


@jit(mode="PIJit")
def greater_forward_func(x, y):
    return ops.greater(x, y)

@jit(mode="PIJit")
def greater_backward_func(x, y):
    return ops.grad(greater_forward_func, (0,))(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
