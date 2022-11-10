import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_gaussian_nll_loss_abnormal_full(mode):
    """
    Feature: gaussian_nll_loss
    Description: Verify abnormal inputs of gaussian_nll_loss
    Expectation: raise TypeError
    """
    context.set_context(mode=mode)
    arr1 = np.arange(8).reshape((4, 2))
    arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    var = Tensor(np.ones((4, 1)), mstype.float32)
    with pytest.raises(TypeError):
        ops.gaussian_nll_loss(a, b, var, full=1, eps=1e-6, reduction='mean')
