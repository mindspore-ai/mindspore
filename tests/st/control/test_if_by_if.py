import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype
from mindspore.nn import Cell
import mindspore.ops.operations as P
import mindspore.ops.functional as F

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_by_if_basic():
    class SubNet(Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name='a')
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name='b')

        def construct(self, x):
            if self.a > self.b:
                x = self.mul(x, 1)
                while self.b < 6:
                    x = self.add(x, x)
                    self.b += 1
            return x

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.subnet = SubNet()
            self.relu = P.ReLU()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name='a')
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name='b')
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name='c')

        def func(self, x):
            for _ in range(0, 2):
                x = self.add(x, 0)
            return x

        def construct(self, x):
            if self.a > self.b:
                x = self.subnet(x)
            else:
                x = self.relu(x)
            if self.a < self.c:
                x = self.func(x)
            else:
                x = self.add(x, 2)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    net = Net()
    out_ms = net(Tensor(input_np))
    out_np = input_np * 4
    assert np.allclose(out_ms.asnumpy(), out_np)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_branch_same_shape():
    """
    Feature: control flow function.
    Description: Two branch must return the same shape.
    Expectation: Null.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x, y):
            for k in range(1):
                if x != 1:
                    for _ in range(1):
                        y = k * x
                        y = self.a + y
                        if x > 5:
                            break
                if x == 5:
                    for _ in range(1):
                        y = self.a - y
                        if x == y:
                            continue
            return x + y

    x = np.array([-1], np.float32)
    y = np.array([2], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    context.set_context(mode=context.GRAPH_MODE)
    fgrad = grad_net(Tensor(x), Tensor(y))
    print(fgrad)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parallel_if_add_by_zero():
    """
    Feature: AddByZero optimization in parallel if.
    Description: AddByZero optimization should not be performed when one node is a Load CNode.
    Expectation: out is a value before second assignment.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(3, dtype.float32), name="a")
            self.zero = Tensor(0, dtype.float32)

        def construct(self, x):
            out = self.zero
            if x > 0:
                F.assign(self.param_a, 1)
                out = out + self.param_a
                F.assign(self.param_a, 6)
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(3, dtype.float32)
    net = Net()
    out = net(x)
    assert np.allclose(out.asnumpy(), np.array(1, np.float32))
