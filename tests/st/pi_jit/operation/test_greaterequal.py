import numpy as np
import pytest
import mindspore as ms
from mindspore import ops


def greater_equal_forward_func(x, y):
    return ops.greater_equal(x, y)


def greater_equal_backward_func(x, y):
    return ops.grad(greater_equal_forward_func, (0,))(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
