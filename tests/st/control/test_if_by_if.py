import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell
import mindspore.ops.operations as P



@pytest.mark.level0
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
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
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
