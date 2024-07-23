from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.common import dtype as mstype


class NetExp(nn.Cell):
    def __init__(self):
        super(NetExp, self).__init__()
        self.exp = ops.exp

    def construct(self, x):
        return self.exp(x)


def test_exp_tensor_api(ms_type):
    """
    Feature: Test exp forward tensor api given data type.
    Description: Test exp tensor api.
    Expectation: The result match to the expect value.
    """
    x_ms = Tensor([0.2, 0.74, 0.04], ms_type)
    exp = NetExp()
    output = exp(x_ms)
    output = output.float().asnumpy() if ms_type == mstype.bfloat16 else output.asnumpy()

    expected = np.array([1.2216413, 2.0923362, 1.0408515]).astype(np.float32)
    assert np.allclose(output, expected, rtol=0.004, atol=0.004)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_exp_tensor_bf16_api():
    """
    Feature: Test exp forward tensor api.
    Description: Test exp for bfloat16.
    Expectation: the result match with the expected result.
    :return:
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_exp_tensor_api(mstype.bfloat16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_exp_tensor_api(mstype.bfloat16)
