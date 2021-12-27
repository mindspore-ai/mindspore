import pytest
import numpy as np
from mindspore import ops, Tensor, context
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell


class AssignNet(Cell):
    def __init__(self, input_variable):
        super(AssignNet, self).__init__()
        self.op = ops.Assign()
        self.input_data = input_variable

    def construct(self, input_x):
        return self.op(self.input_data, input_x)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_assign_as_output():
    """
    Feature: PyNative MindRT
    Description: Test PyNative MindRT RefNode.
    Expectation: No exception.
    """

    np.random.seed(0)
    input_np = np.random.randn(5, 5).astype(dtype=np.int32)
    context.set_context(mode=context.PYNATIVE_MODE)
    input_variable = Parameter(Tensor(np.random.randn(5, 5).astype(dtype=np.float32)))
    input_x = Tensor(input_np)
    net = AssignNet(input_variable)
    out = net(input_x)
    assert input_np.all() == out.asnumpy().astype(dtype=np.int32).all()
