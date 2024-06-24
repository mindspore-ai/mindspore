from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import array_ops as P


class UnravelIndex(nn.Cell):
    def __init__(self):
        super(UnravelIndex, self).__init__()
        self.unravel_index = P.UnravelIndex()

    def construct(self, indices, dims):
        return self.unravel_index(indices, dims)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unravel_index_0d_graph():
    """
    Feature: UnravelIndex gpu TEST.
    Description: 0d test case for UnravelIndex
    Expectation: the result match to numpy
    """
    indices = Tensor(np.array([1621]), mstype.int32)
    dims = Tensor(np.array([6, 7, 8, 9]), mstype.int32)

    output_np = np.array([[3], [1], [4], [1]]).astype(np.int32)
    output_ms = UnravelIndex()(indices, dims)

    assert (output_ms.asnumpy() == output_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unravel_index_1d_pynative():
    """
    Feature: UnravelIndex gpu TEST.
    Description: 1d test case for UnravelIndex
    Expectation: the result match to numpy
    """

    indices = Tensor(np.array([2, 5, 7]), mstype.int64)
    dims = Tensor(np.array([3, 3]), mstype.int64)

    output_np = np.array([[0, 1, 2], [2, 2, 1]]).astype(np.int64)
    output_ms = UnravelIndex()(indices, dims)

    assert (output_ms.asnumpy() == output_np).all()
