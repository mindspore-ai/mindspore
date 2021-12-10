import pytest
import numpy as np
from mindspore import Tensor, ms_function, context
import mindspore.common.dtype as mstype
from mindspore.common.initializer import One


context.set_context(mode=context.GRAPH_MODE)

def test_tensor():
    """
    Feature: JIT Fallback
    Description: Test Tensor() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor(1)
        return me_x
    print(foo())


def test_tensor_bool():
    """
    Feature: JIT Fallback
    Description: Test Tensor(bool) in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor([True, True, False])
        return me_x
    print(foo())


def test_tensor_array():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor([1])
        return me_x
    print(foo())


def test_tensor_with_mstype():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with mstype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor(1, mstype.int32)
        return me_x
    print(foo())


def test_tensor_array_with_mstype():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) with mstype in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor([1], mstype.int32)
        return me_x
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_array_astype():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) with astype() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor([1.1, -2.1]).astype("float32")
        return me_x
    print(foo())


def test_tensor_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with numpy in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor(np.zeros([1, 2, 3]), mstype.float32)
        return me_x
    print(foo())


def test_tensor_with_init():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with init in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor(shape=(1, 3), dtype=mstype.float32, init=One())
        return me_x
    print(foo())


def test_tensor_reshape():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with reshape() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        me_x = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        return me_x
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_abs():
    """
    Feature: JIT Fallback
    Description: Test Tensor.abs() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        a = Tensor([1.1, -2.1]).astype("float32")
        out = a.abs()
        return out
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_all():
    """
    Feature: JIT Fallback
    Description: Test Tensor.all() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        a = Tensor([True, True, False])
        out = a.all()
        return out
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_any():
    """
    Feature: JIT Fallback
    Description: Test Tensor.any() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        a = Tensor([True, True, False])
        out = a.any()
        return out
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_argmax():
    """
    Feature: JIT Fallback
    Description: Test Tensor.argmax() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        out = a.argmax()
        return out
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_argmin():
    """
    Feature: JIT Fallback
    Description: Test Tensor.argmin() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        out = a.argmin()
        return out
    print(foo())


@pytest.mark.skip(reason='Not support in graph jit fallback feature yet')
def test_tensor_astype():
    """
    Feature: JIT Fallback
    Description: Test Tensor.astype() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        a = Tensor(np.ones((1, 2, 2, 1), dtype=np.float32))
        out = a.astype("int32")
        return out
    print(foo())
