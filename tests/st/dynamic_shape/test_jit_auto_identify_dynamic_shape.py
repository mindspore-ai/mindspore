import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class MulNet(nn.Cell):
    def __init__(self):
        super(MulNet, self).__init__()
        self.mul = P.BatchMatMul()

    @jit
    def construct(self, x, y):
        output = self.mul(x, y)
        return output


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul_net = MulNet()

    def construct(self, x, y):
        out = self.matmul_net(x, y)
        return out


def test_case(device_target):
    """
    Feature: auto dynamic identify test.
    Description:  auto dynamic identify test in gpu and Ascend.
    Expectation: Assert the result with expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)
    mul_net = Net()

    # step1:(2, 2), (2, 3);
    x1 = Tensor(((1, 1), (1, 1)), dtype=mstype.int32)
    y1 = Tensor(((2, 2, 2), (2, 2, 2)), dtype=mstype.int32)
    output = mul_net(x1, y1)
    expect = Tensor(np.array([[4, 4, 4], [4, 4, 4]]).astype(np.int32))
    print("step1 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step2:(2, 2), (2, 3); use real shape cache
    x2 = Tensor(((1, 1), (1, 1)), dtype=mstype.int32)
    y2 = Tensor(((3, 3, 3), (3, 3, 3)), dtype=mstype.int32)
    output = mul_net(x2, y2)
    expect = Tensor(np.array([[6, 6, 6], [6, 6, 6]]).astype(np.int32))
    print("step2 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step3:(3, 2), (2, 3); generate to (-1, -1), (2, 3)
    x3 = Tensor(((1, 1), (1, 1), (1, 1)), dtype=mstype.int32)
    y3 = Tensor(((3, 3, 3), (3, 3, 3)), dtype=mstype.int32)
    output = mul_net(x3, y3)
    expect = Tensor(np.array([[6, 6, 6], [6, 6, 6], [6, 6, 6]]).astype(np.int32))
    print("step3 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step4:(4, 2), (2, 3);  use cache (-1, -1), (2, 3)
    x4 = Tensor(((1, 1), (1, 1), (1, 1), (1, 1)), dtype=mstype.int32)
    y4 = Tensor(((3, 3, 3), (3, 3, 3)), dtype=mstype.int32)
    output = mul_net(x4, y4)
    expect = Tensor(np.array([[6, 6, 6], [6, 6, 6], [6, 6, 6], [6, 6, 6]]).astype(np.int32))
    print("step4 finish", flush=True)

    # step5:(2, 2), (2, 4);  generate to (2, 2), (-1, -1)
    x5 = Tensor(((2, 2), (2, 2)), dtype=mstype.int32)
    y5 = Tensor(((2, 2, 2, 2), (2, 2, 2, 2)), dtype=mstype.int32)
    output = mul_net(x5, y5)
    expect = Tensor(np.array([[8, 8, 8, 8], [8, 8, 8, 8]]).astype(np.int32))
    print("step5 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step6:(3, 2), (2, 4); generate to (-1, -1), (-1, -1)
    x6 = Tensor(((1, 1), (1, 1), (1, 1)), dtype=mstype.int32)
    y6 = Tensor(((3, 3, 3, 3), (3, 3, 3, 3)), dtype=mstype.int32)
    output = mul_net(x6, y6)
    expect = Tensor(np.array([[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]).astype(np.int32))
    print("step6 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step7:(2, 5), (5, 2); generate to (-1, -1), (-1, -1)
    x7 = Tensor(((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)), dtype=mstype.int32)
    y7 = Tensor(((3, 3), (3, 3), (3, 3), (3, 3), (3, 3)), dtype=mstype.int32)
    output = mul_net(x7, y7)
    expect = Tensor(np.array([[15, 15], [15, 15]]).astype(np.int32))
    print("step7 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step8:(2, 2), (2, 3); generate new cache because this is float32
    x8 = Tensor(((1, 1), (1, 1)), dtype=mstype.float32)
    y8 = Tensor(((3, 3, 3), (3, 3, 3)), dtype=mstype.float32)
    output = mul_net(x8, y8)
    expect = Tensor(np.array([[6, 6, 6], [6, 6, 6]]).astype(np.float32))
    print("step8 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step9:(2, 2), (2, 4); generate (2, 2), (-1, -1)
    x9 = Tensor(((1, 1), (1, 1)), dtype=mstype.float32)
    y9 = Tensor(((3, 3, 3, 3), (3, 3, 3, 3)), dtype=mstype.float32)
    output = mul_net(x9, y9)
    expect = Tensor(np.array([[6, 6, 6, 6], [6, 6, 6, 6]]).astype(np.float32))
    print("step9 finish", flush=True)
    assert(output.asnumpy() == expect).all()

    # step10:(2, 4, 1, 3), (2, 4, 3, 4); generate new cache because rank changed
    x10 = Tensor(np.ones(shape=[2, 4, 1, 3]), mstype.float32)
    y10 = Tensor(np.ones(shape=[2, 4, 3, 4]), mstype.float32)
    output = mul_net(x10, y10)
    print("step10 finish", flush=True)
    expect = Tensor(np.array([[[[3, 3, 3, 3]], [[3, 3, 3, 3]], [[3, 3, 3, 3]], [[3, 3, 3, 3]]],
                              [[[3, 3, 3, 3]], [[3, 3, 3, 3]], [[3, 3, 3, 3]], [[3, 3, 3, 3]]]]).astype(np.float32))
    assert(output.asnumpy() == expect).all()



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_auto_identify_case_gpu():
    """
    Feature: Test auto identify dynamic shape with jit.
    Description:  auto identify dynamic shape in gpu.
    Expectation: Assert the result with expected.
    """
    test_case("GPU")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_identify_case_ascend():
    """
    Feature: Test auto identify dynamic shape with jit.
    Description:  auto identify dynamic shape in Ascend.
    Expectation: Assert the result with expected.
    """
    test_case("Ascend")
