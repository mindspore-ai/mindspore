# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter, mutable
from mindspore.ops import operations as P
from mindspore.ops.operations import _sequence_ops as S
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_single_add():
    """
    Feature: Support tensor inplace.
    Description: Fix the input host tensor.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x, y):
            self.assignadd(x, y)
            return x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    net(input_x, input_y)
    assert input_x == 5


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd_cpu = P.AssignAdd()
            self.assignadd_cpu.set_device("CPU")
            self.z = Parameter(Tensor(1))

        def construct(self, x, y):
            self.assignadd_cpu(self.z, x)
            self.assignadd_ascend(self.z, y)
            self.assignadd_cpu(self.z, x)
            self.assignadd_ascend(self.z, y)
            return self.z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 11


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_add_cnode1():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x, y):
            z = self.add(x, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 15


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_add_cnode2():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = x + y
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 15



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_add_cnode3():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = x + y
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            return z + x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 17


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_add_cnode4():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = x + y
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            return z + x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 19



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_heter_control_flow_add_cnode4():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = x + y
            if z < 10:
                self.assignadd(z, x)
                self.assignadd_ascend(z, y)
                self.assignadd(z, x)
                self.assignadd_ascend(z, y)
                self.assignadd(z, x)
                return z + x
            return x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 19


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_add_cnode5():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x, y):
            z = self.add(x, y)
            z1 = self.assignadd(z, x)
            z2 = self.assignadd_ascend(z1, y)
            z3 = self.assignadd(z2, x)
            z4 = self.assignadd_ascend(z3, y)
            return z4

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 15


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_control_flow_add_cnode():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x, y):
            z = self.add(x, y)
            if z < 10:
                z1 = self.assignadd(z, x)
                z2 = self.assignadd_ascend(z1, y)
                z3 = self.assignadd(z2, x)
                z4 = self.assignadd_ascend(z3, y)
                return z4
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 15


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_control_flow_inplace_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.z1 = Parameter(Tensor(1))
            self.z2 = Parameter(Tensor(2))

        def construct(self, x, y):
            if x < 3:
                z = self.z1
            else:
                z = self.z2
            self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(4)
    net = Net()
    out1 = net(input_x, input_y)
    out2 = net(input_y, input_x)
    assert out1 == 5
    assert out2 == 4


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_control_flow_assign_for_stack_input():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.z1 = Parameter(Tensor(1))
            self.z2 = Parameter(Tensor(2))

        def construct(self, x, y):
            if x < 3:
                z = self.z1
            else:
                z = self.z2
            P.Assign()(z, y)
            return self.z1, self.z2

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(4)
    net = Net()
    out1 = net(input_x, input_y)
    out2 = net(input_y, input_x)
    assert out1[0] == 4
    assert out1[1] == 2
    assert out2[0] == 4
    assert out2[1] == 2


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_control_flow_inplace_for_stack_input():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()

        def construct(self, x, y):
            if x < 3:
                z = x + y
            else:
                z = x * y
            self.assignadd_ascend(z, x)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(4)
    net = Net()
    out1 = net(input_x, input_y)
    out2 = net(input_y, input_x)
    assert out1 == 8
    assert out2 == 12


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_control_flow_assign_for_entrance_input():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.z1 = Parameter(Tensor(1))
            self.z2 = Parameter(Tensor(2))

        def construct(self, x, y):
            if x < 3:
                P.Assign()(self.z1, y)
            else:
                P.Assign()(self.z2, y)
            return self.z1, self.z2

    input_x1 = ms.Tensor(2)
    input_y1 = ms.Tensor(4)
    input_x2 = ms.Tensor(4)
    input_y2 = ms.Tensor(5)
    net = Net()
    out1 = net(input_x1, input_y1)
    out2 = net(input_x2, input_y2)
    assert out1[0] == 4
    assert out1[1] == 2
    assert out2[0] == 4
    assert out2[1] == 5


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_control_flow_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.z = Parameter(Tensor(1))

        def construct(self, x, y):
            if x < 10:
                self.assignadd(self.z, x)
                self.assignadd_ascend(self.z, y)
                self.assignadd(self.z, x)
                self.assignadd_ascend(self.z, y)
            return self.z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 11


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_control_flow_for_root_entrance_actor():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.add = P.Add()
            self.add.set_device("CPU")
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = self.add(x, y)
            self.assignadd_ascend(z, y)
            if x < 10:
                self.assignadd(z, x)
                self.assignadd_ascend(z, y)
                self.assignadd(z, x)
                self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 18


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_inplace_add_parameter():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.z = Parameter(Tensor(1))

        def construct(self, x, y):
            out = self.assignadd(y, self.z)
            self.assignadd_ascend(out, y)
            self.assignadd(out, x)
            self.assignadd_ascend(out, self.z)
            return out

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 11


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_heter_control_flow_input_heter_for_assign():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.assign = P.Assign()
            self.assign.set_device("CPU")

        def construct(self, x, y):
            z = x + y
            if z < 10:
                z1 = self.assignadd(z, x)
                z2 = z + y
                z1 = z1 + x
                return z1, z2
            return x, y

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 9
    assert out[1] == 10


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=2048.0)
def test_heter_inplace_add_input():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = self.assignadd(y, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 24


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_inplace_add_cnode():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z = x + y
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 15


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_heter_inplace_add_self():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            self.assignadd(x, x)
            self.assignadd_ascend(x, y)
            self.assignadd(x, x)
            self.assignadd_ascend(x, y)
            return x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 17


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_heter_inplace_add_self_multi_heter():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            self.assignadd(x, x)
            self.assignadd_ascend(x, y)
            self.assignadd(x, x)
            self.assignadd_ascend(x, y)
            self.assignadd(x, x)
            self.assignadd_ascend(x, y)
            self.assignadd(x, x)
            self.assignadd_ascend(x, y)
            return x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 77


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_control_flow_inplace_from_entrance_actor():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            if x < 3:
                self.assignadd(x, x)
                self.assignadd_ascend(x, y)
                self.assignadd(x, x)
                self.assignadd_ascend(x, y)
            return x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 17


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=2048.0)
def test_heter_inplace_by_output():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z1 = self.assignadd(x, x)
            z2 = self.assignadd_ascend(z1, y)
            z3 = self.assignadd(z2, x)
            z4 = self.assignadd_ascend(z3, y)
            return z4

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 17


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_control_flow_inplace_from_entrance_actor_by_output():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            z4 = x
            if x < 3:
                z1 = self.assignadd(x, x)
                z2 = self.assignadd_ascend(z1, y)
                z3 = self.assignadd(z2, x)
                z4 = self.assignadd_ascend(z3, y)
            return z4

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 17


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_heter_control_flow_inplace_from_stack_actor():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y):
            if x < 3:
                z = x
            else:
                z = y
            self.assignadd(z, x)
            self.assignadd_ascend(z, y)
            self.assignadd(y, x)
            self.assignadd_ascend(z, y)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 17


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=2048.0)
def test_heter_backoff_inplace_in_super_kernel_actor():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.seq_slice = S.SequenceSlice()

        def construct(self, x, y, z):
            aa = P.AddN()(z)
            z = z.append(x)
            z = z[0]
            self.assignadd_ascend(z, y)
            self.assignadd_ascend(z, aa)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    input_z = mutable([input_x, input_y], dynamic_len=True)
    net = Net()
    out1 = net(input_x, input_y, input_z)
    out2 = net(input_x, input_y, input_z)
    print("out:", out1)
    assert out1 == 10
    assert out2 == 10


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=2048.01)
def test_heter_backoff_inplace_in_super_kernel_actor_2():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.seq_slice = S.SequenceSlice()

        def construct(self, x, y, z):
            aa = P.AddN()(z)
            z = z.append(x)
            z = z[0]
            z = self.assignadd_ascend(z, y)
            z = self.assignadd_ascend(z, aa)
            return z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    input_z = mutable([input_x, input_y], dynamic_len=True)
    net = Net()
    out1 = net(input_x, input_y, input_z)
    out2 = net(input_x, input_y, input_z)
    print("out:", out1)
    assert out1 == 10
    assert out2 == 10


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_heter_backoff_control_flow_inplace_in_super_kernel_actor():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.seq_slice = S.SequenceSlice()

        def construct(self, x, y, z):
            if x < 3:
                z = z.append(x)
                z = z[0]
                self.assignadd_ascend(z, y)
                self.assignadd_ascend(z, x)
                return z
            return x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    input_z = mutable([input_x, input_y], dynamic_len=True)
    net = Net()
    out = net(input_x, input_y, input_z)
    print("out:", out)
    assert out == 7


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_control_flow_inplace_in_super_kernel_actor():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")
            self.seq_slice = S.SequenceSlice()
            self.z = Parameter(Tensor(1))

        def construct(self, x, y):
            z = x + y
            if x < 10:
                aa = self.assignadd_ascend(z, x)
            else:
                aa = self.z
            return aa

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 7


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_inplace_nopnode():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            z = self.squeeze(x)
            self.assignadd_ascend(z, Tensor([2, 2, 2, 2]))
            return z

    input_x = Tensor(np.zeros([4, 1]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out.asnumpy() == [2, 2, 2, 2])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_nopnode_inplace():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            self.assignadd_ascend(x, Tensor([[2, 2, 2, 2]]))
            z = self.squeeze(x)
            return z

    input_x = Tensor(np.zeros([1, 4]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out.asnumpy() == [2, 2, 2, 2])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_inplace_parameter_multi_output():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x):
            y = self.assignadd(x, 1)
            return x, y

    x = ms.Tensor(2)
    net = Net()
    out = net(x)
    assert out[0] == 3
    assert out[1] == 3


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=2048.0)
def test_inplace_cnode_multi_output():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x):
            x = x + 1
            y = self.assignadd(x, 1)
            return x, y

    x = ms.Tensor(2)
    net = Net()
    out = net(x)
    assert out[0] == 4
    assert out[1] == 4


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_inplace_control_flow_parameter_multi_output():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x):
            if x < 3:
                y = self.assignadd(x, 1)
                return x, y
            return x, x

    x = ms.Tensor(2)
    net = Net()
    out = net(x)
    assert out[0] == 3
    assert out[1] == 3


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard",
        essential_mark="essential", mem_peak=1.00)
def test_inplace_control_flow_cnode_multi_output():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x):
            x = x + 1
            if x < 4:
                x = x + 1
                y = self.assignadd(x, 1)
                return x, y
            return x, x

    x = ms.Tensor(2)
    net = Net()
    out = net(x)
    assert out[0] == 5
    assert out[1] == 5


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_cnode_nopnode_inplace():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            x = x + Tensor([[2, 2, 2, 2]])
            y = self.squeeze(x)
            self.assignadd(y, Tensor([3, 3, 3, 3]))
            return x, y

    context.set_context(device_target="CPU")
    input_x = Tensor(np.zeros([1, 4]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out[0].asnumpy() == [2, 2, 2, 2])
    assert np.all(out[1].asnumpy() == [5, 5, 5, 5])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_cnode_double_nopnode_inplace1():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            x = x + Tensor([[2, 2, 2, 2]])
            y = self.squeeze(x)
            z = self.squeeze(y)
            self.assignadd(z, Tensor([3, 3, 3, 3]))
            return x, y, z

    context.set_context(device_target="CPU")
    input_x = Tensor(np.zeros([1, 4]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out[0].asnumpy() == [[2, 2, 2, 2]])
    assert np.all(out[1].asnumpy() == [2, 2, 2, 2])
    assert np.all(out[2].asnumpy() == [5, 5, 5, 5])



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_cnode_double_nopnode_inplace2():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            x = x + Tensor([[2, 2, 2, 2]])
            y = self.squeeze(x)
            z = self.squeeze(y)
            self.assignadd(z, Tensor([3, 3, 3, 3]))
            return x, z

    context.set_context(device_target="CPU")
    input_x = Tensor(np.zeros([1, 4]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out[0].asnumpy() == [[2, 2, 2, 2]])
    assert np.all(out[1].asnumpy() == [5, 5, 5, 5])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_cnode_inplace_nopnode1():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            x = x + Tensor([[2, 2, 2, 2]])
            self.assignadd(x, Tensor([[3, 3, 3, 3]]))
            y = self.squeeze(x)
            z = self.squeeze(y)
            self.assignadd(x, Tensor([[4, 4, 4, 4]]))
            return x, y, z

    context.set_context(device_target="CPU")
    input_x = Tensor(np.zeros([1, 4]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out[0].asnumpy() == [[9, 9, 9, 9]])
    assert np.all(out[1].asnumpy() == [5, 5, 5, 5])
    assert np.all(out[2].asnumpy() == [5, 5, 5, 5])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_cnode_inplace_nopnode2():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            x = x + Tensor([[2, 2, 2, 2]])
            self.assignadd(x, Tensor([[3, 3, 3, 3]]))
            y = self.squeeze(x)
            z = self.squeeze(y)
            zz = self.squeeze(z)
            self.assignadd(x, Tensor([[4, 4, 4, 4]]))
            return x, zz

    context.set_context(device_target="CPU")
    input_x = Tensor(np.zeros([1, 4]).astype(np.int64))
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert np.all(out[0].asnumpy() == [[9, 9, 9, 9]])
    assert np.all(out[1].asnumpy() == [5, 5, 5, 5])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=2048.01)
def test_heter_inplace_dynamic_shape():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd_ascend = P.AssignAdd()
            self.assignadd = P.AssignAdd()
            self.assignadd.set_device("CPU")

        def construct(self, x, y, z):
            while x < 4:
                x = x + 1
                self.assignadd(z, x)
                self.assignadd_ascend(z, y)
                self.assignadd(z, x)
                self.assignadd_ascend(z, y)
            return z

    x_dyn = Tensor(shape=[None], dtype=ms.int64)
    y_dyn = Tensor(shape=[None], dtype=ms.int64)
    z_dyn = Tensor(shape=[None], dtype=ms.int64)
    net = Net()
    net.set_inputs(x_dyn, y_dyn, z_dyn)

    input_x = Tensor([1])
    input_y = Tensor([3])
    input_z = Tensor([1])
    out1 = net(input_x, input_y, input_z)

    input_x = Tensor([1, 1, 1])
    input_y = Tensor([3, 3, 3])
    input_z = Tensor([1, 1, 1])
    out2 = net(input_x, input_y, input_z)

    assert out1 == 37
    assert np.all(out2.asnumpy() == [37, 37, 37])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_cnode_execute_twice():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x, y):
            return self.assignadd(x, y)

    x1 = ms.Tensor(2)
    y1 = ms.Tensor(3)
    x2 = ms.Tensor(4)
    y2 = ms.Tensor(6)
    net = Net()
    out1 = net(x1, y1)
    assert out1 == 5
    out2 = net(x2, y2)
    assert out2 == 10


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_output_input_device_address():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x, y):
            return self.assignadd(x, y)

    x1 = ms.Tensor(2)
    y1 = ms.Tensor(3)
    y2 = ms.Tensor(6)
    net = Net()
    context.set_context(device_target="Ascend")
    context.set_context(mode=context.PYNATIVE_MODE)
    x2 = net(x1, y1)
    context.set_context(mode=context.GRAPH_MODE)
    out = net(x2, y2)
    assert out == 11


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_output_input_device_address_for_control_flow():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()

        def construct(self, x, y):
            z = self.assignadd(x, y)
            if z < 10:
                return z + 1
            return z + 2

    x1 = ms.Tensor(2)
    y1 = ms.Tensor(3)
    y2 = ms.Tensor(6)
    net = Net()
    context.set_context(device_target="Ascend")
    context.set_context(mode=context.PYNATIVE_MODE)
    x2 = net(x1, y1)
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    out = net(x2, y2)
    assert out == 14
