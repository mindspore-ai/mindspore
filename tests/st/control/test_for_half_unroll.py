import os
import numpy as np
from tests.st.control.cases_register import case_register
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
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

        def construct(self, x):
            output = x
            for i in self.array:
                output += i

            return output

    net = ForLoopBasic()
    x = Tensor(np.array(10).astype(np.int32))
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = '1'
    res = net(x)
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = ''
    assert res == 25


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
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

        def construct(self, x):
            output = x
            for i in self.array:
                if i < 10:
                    output += i

            return output

    net = ForLoopIf()
    x = Tensor(np.array(10).astype(np.int32))
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = '1'
    res = net(x)
    os.environ['MS_DEV_FOR_HALF_UNROLL'] = ''
    assert res == 15
