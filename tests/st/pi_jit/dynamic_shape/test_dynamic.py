from mindspore._c_expression import update_pijit_default_config
from mindspore.nn import Cell
from mindspore import ops
from mindspore import context, jit
from mindspore.common import dtype
from mindspore.common import Tensor
import numpy as np
import pytest
from tests.mark_utils import arg_mark


class DynamicFactory:
    def __init__(self, ps_net):
        self.ps_net = ps_net

    def forward_cmp(self, inputs):
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(fn=self.ps_net.construct, mode="PIJit")(inputs)
        self.ps_net(inputs)

class Net7(Cell):
    def __init__(self):
        super().__init__()
        self.pow_op = ops.Pow()

    def construct(self, x):
        a = self.pow_op(x, 0.0)
        #print(type(a),"hejianheng")
        b = ops.rrelu(a)
        return b


@pytest.mark.skip
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_frontend_optimize():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net with pow rrelu
        2. run twice for Resize
        3. set inputs for pow frontend pass
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net7()

    #x = np.random.randn(3, 4, 5).astype(np.float32)
    #s = np.random.randn(3, 4, 5).astype(np.float32)
    d = Tensor(np.random.randn(3, 4, 5), dtype=dtype.float32)
    fact = DynamicFactory(ps_net)
    fact.forward_cmp(d)
