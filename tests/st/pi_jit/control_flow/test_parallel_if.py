from mindspore import Tensor, jit, context
from mindspore.common import dtype as mstype
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_if_raise_raise():
    """
    Feature: Parallel if transformation
    Description: raise in if requires that the after_if func graph should not
                be called, so it cannot be transformed. The outer if can be transformed.
    Expectation: success
    """

    @jit(mode="PIJit")
    def foo(x, y, z):
        out = z
        if x >= y:
            if x > y:
                raise ValueError("x is bigger y")
        else:
            out = out * 2
        out = out + out
        return out
    context.set_context(mode=context.PYNATIVE_MODE)
    x = 3
    y = 2
    z = Tensor([5], mstype.int32)

    with pytest.raises(ValueError):
        foo(x, y, z)
