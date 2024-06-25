from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_shape_not_none():
    '''
    Description:
        1. create a tensor, all args are int
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    Tensor(input_data=None, dtype=mstype.float32, shape=[2, 4], init=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
