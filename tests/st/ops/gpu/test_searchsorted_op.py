from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P



class SearchSortedNet(nn.Cell):
    def __init__(self, out_int32=mindspore.int32, right=False):
        super(SearchSortedNet, self).__init__()
        self.searchsorted = P.SearchSorted(dtype=out_int32, right=right)

    def construct(self, sequence, values):
        return self.searchsorted(sequence, values)


def search_sorted(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input1 = Tensor(np.array([[0, 1, 3, 5, 7], [2, 4, 6, 8, 10]]), mindspore.float32)
    input2 = Tensor(np.array([[3, 6, 9], [3, 6, 9]]), mindspore.float32)
    net = SearchSortedNet(out_int32=mindspore.int32, right=False)
    expect = np.array([[2, 4, 5], [1, 2, 4]], dtype=np.int32)
    output = net(input1, input2)
    assert np.allclose(output.asnumpy(), expect, loss, loss)


def search_sorted_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input1 = Tensor(np.array([[0, 1, 3, 5, 7], [2, 4, 6, 8, 10]]), mindspore.float32)
    input2 = Tensor(np.array([[3, 6, 9], [3, 6, 9]]), mindspore.float32)
    net = SearchSortedNet(out_int32=mindspore.int32, right=False)
    expect = np.array([[2, 4, 5], [1, 2, 4]], dtype=np.int32)
    output = net(input1, input2)
    assert np.allclose(output.asnumpy(), expect, loss, loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_search_sorted_graph_int32():
    """
    Feature: ALL To ALL
    Description: test cases for SearchSorted
    Expectation: the result match to pytorch
    """
    search_sorted(loss=1.0e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_search_sorted_pynative_int32():
    """
    Feature: ALL To ALL
    Description: test cases for SearchSorted
    Expectation: the result match to pytorch
    """
    search_sorted_pynative(loss=1.0e-5)
