import pytest
import numpy as np
from mindspore import Tensor, jit, context
import mindspore.ops as ops
from .share.utils import match_array
from tests.mark_utils import arg_mark


class ExpandDimsTest():
    def __init__(self, axis):
        self.expandDims = ops.ExpandDims()
        self.axis = axis

    @jit(mode="PIJit")
    def test1(self, input_x):
        return self.expandDims(input_x, self.axis)

    @jit
    def test2(self, input_x):
        return self.expandDims(input_x, self.axis)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('input_x', [Tensor(np.ones((2, 2, 2, 2)).astype(np.float32))])
def test_method_annotation(input_x):
    """
    Feature: Method Annotation Testing
    Description: Test the methods of a class with annotation using different inputs.
    Expectation: The results of the test1 and test2 methods should match for the given input.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    axis = 0
    expand_dim = ExpandDimsTest(axis)
    res = expand_dim.test1(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = expand_dim.test2(input_x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
