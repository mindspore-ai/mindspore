import pytest
import os
import numpy as np
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.nn import Cell


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_half_unroll_basic():
    """
    Feature: Half unroll compile optimization for for statement.
    Description: Only test for statement.
    Expectation: Correct result and no exception.
    """
    class ForLoopBasic(Cell):
        def __init__(self):
            super().__init__()
            self.array = (Tensor(np.array(10).astype(np.int32)), Tensor(np.array(5).astype(np.int32)))

        @jit(mode="PIJit")
        def construct(self, x):
            output = x
            for i in self.array:
                output += i

            return output

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ForLoopBasic()
    x = Tensor(np.array(10).astype(np.int32))
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = '1'
    res = net(x)
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = ''
    assert res == 25


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_half_unroll_if():
    """
    Feature: Half unroll compile optimization for for statement.
    Description: Test for-in statements.
    Expectation: Correct result and no exception.
    """
    class ForLoopIf(Cell):
        def __init__(self):
            super().__init__()
            self.array = (Tensor(np.array(10).astype(np.int32)), Tensor(np.array(5).astype(np.int32)))

        @jit(mode="PIJit")
        def construct(self, x):
            output = x
            for i in self.array:
                if i < 10:
                    output += i

            return output

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ForLoopIf()
    x = Tensor(np.array(10).astype(np.int32))
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = '1'
    res = net(x)
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = ''
    assert res == 15
