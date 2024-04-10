from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
import numpy as np
from .test_dynamic_rank_tensor_getitem import IndexFactory
import pytest


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.n = 2

    def construct(self, x):
        x[...] = 1
        out = x
        return out * self.n


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_rank_setitem_ellipsis():
    '''
    Description:
        1. dynamic rank setitem ellipsis
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net1()
    pi_net = Net1()
    x = Tensor(np.random.rand(2, 3, 4), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    ps_net.set_inputs(d)
    pi_net.set_inputs(d)
    fact = IndexFactory(ps_net, pi_net)
    fact.compare_forward_grad(x)
