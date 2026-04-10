# coding=utf-8

import pytest
import mindspore as ms
from mindspore import numpy as np
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark


@jit
def jit_in(a, b):
    return a in b


def pynative_in(a, b):
    return a in b


@jit
def jit_not_in(a, b):
    return a not in b


def pynative_not_in(a, b):
    return a not in b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize("jit_fn, pynative_fn", [(jit_in, pynative_in), (jit_not_in, pynative_not_in)])
@pytest.mark.parametrize('a', [1, 0])
@pytest.mark.parametrize('b', [[1, 2, 3], {1: 1, 2: 2}, (1, 2, 3)])
@pytest.mark.parametrize('use_mutable', [True, False])
def test_in_not_in(jit_fn, pynative_fn, a, b, use_mutable):
    """
    Feature: Test 'in' and 'not in' operators
    Description: Validate the behavior of 'in' and 'not in' operators for different types of data structures.
    Expectation: Both should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    if use_mutable:
        b = ms.mutable(b)
    o1 = pynative_fn(a, b)
    o2 = jit_fn(a, b)
    assert o1 == o2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize("jit_fn, pynative_fn", [(jit_in, pynative_in), (jit_not_in, pynative_not_in)])
@pytest.mark.parametrize('a', [Tensor(np.ones((2, 3)).astype(np.float32))])
@pytest.mark.parametrize('b', [Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))])
def test_tensor_in_list(jit_fn, pynative_fn, a, b):
    """
    Feature: Test 'in' and 'not in' operators with Tensors and lists
    Description: Validate the behavior of 'in' and 'not in' operators when a Tensor is in a list.
    Expectation: Both PIJit and PSJit functions should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    o1 = jit_fn(a, [a, b])
    o2 = pynative_fn(a, [a, b])
    assert o1 == o2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize("jit_fn, pynative_fn", [(jit_in, pynative_in), (jit_not_in, pynative_not_in)])
@pytest.mark.parametrize('a', ['123', '012'])
@pytest.mark.parametrize('b', ['123-456', '456-789'])
def test_string_in_not_in(jit_fn, pynative_fn, a, b):
    """
    Feature: Test 'in' and 'not in' operators with strings
    Description: Validate the behavior of 'in' and 'not in' operators when the operands are strings.
    Expectation: Both PIJit and PSJit functions should return the same results.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    o1 = jit_fn(a, b)
    o2 = pynative_fn(a, b)
    assert o1 == o2
