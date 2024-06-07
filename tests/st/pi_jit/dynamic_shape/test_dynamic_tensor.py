from mindspore.common import Tensor
from mindspore.common import dtype as mstype
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_shape_not_none():
    '''
    Description:
        1. create a tensor, all args are int
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    Tensor(input_data=None, dtype=mstype.float32, shape=[2, 4], init=1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_shape():
    '''
    Description:
        1. create a tensor, all args are None
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    x = Tensor(dtype=mstype.float32, shape=[None, 4])
    s = x.shape
    assert s == (-1, 4)
