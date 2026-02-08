# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
# ============================================================================
""" test kwargs with side effect. """
import pytest
import numpy as np
from mindspore.ops import operations as P

import mindspore as ms
from mindspore import Tensor, Parameter, context, nn, jit, ops
from mindspore.ops import GradOperation
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_kwargs_has_side_effect():
    """
    Feature: Support kwargs has side effect.
    Description: Support kwargs has side effect.
    Expectation: No exception.
    """

    def multi_forward(input_x, call_func=None):
        return call_func(input_x)

    class KwargsTestNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor([1.0], ms.float32), name="para1")
            self.assign = P.Assign()

        def my_assign_value(self, value):
            self.assign(self.param, value * 2)
            return self.param + 2

        def construct(self, x):
            return multi_forward(x, call_func=self.my_assign_value)

    net = KwargsTestNet()
    out = net(Tensor([10], ms.float32))
    print(out)
    assert out == 22


@pytest.mark.skip("the key in kwargs is any")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_kwargs_key_value_both_is_custom_class_attr():
    """
    Feature: Support the kwargs is any.
    Description: Graph syntax resolve support custom class input is kwargs.
    Expectation: No error.
    """

    class Config:
        def __init__(self, **kwargs):
            self.aaa = kwargs.pop("aaa", 2.0)
            self.input1 = "input1"

    class Model(ms.nn.Cell):
        def construct(self, input1, input2):
            return input1 * input2

    class Net(ms.nn.Cell):
        def __init__(self, net_config):
            super().__init__()
            self.config = net_config
            self.model = Model()

        def construct(self):
            arg_dict = {self.config.input1: self.config.aaa + 1, "input2": self.config.aaa}
            return self.model(**arg_dict)

    config = Config()
    net = Net(config)
    output = net()
    assert output == 6


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_useless_kwargs():
    """
    Feature: Support the kwargs is not used in function.
    Description: Graph syntax support kwargs.
    Expectation: No error.
    """
    x = Tensor([1, 2])

    @jit
    def func(*args, **conf):
        def ff(x, *args, **conf):
            return ops.mul(*x, *args)

        return ff(*args)

    res = func((x,), x, a=x)

    assert np.allclose(res.asnumpy(), np.array([1, 4]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_partial_kwargs():
    """
    Feature: Support Not all key-value parameters are fully utilized in the iinput function.
    Description: Graph syntax support kwargs.
    Expectation: No error.
    """
    x = Tensor([1, 2])

    @jit
    def func(*args, **conf):
        def ff(x, *args, **conf):
            res = ops.mul(*x, *args)
            res = res + conf.get('a', 0)
            return res

        return ff(*args)

    res = func((x,), x, a=x, b=x)

    assert np.allclose(res.asnumpy(), np.array([1, 4]))


class GradOperationNet(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args, **kwargs):
        gradient_function = self.grad_op(self.net)
        return gradient_function(*args, **kwargs)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_args_var_kwargs_001():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    class Net(nn.Cell):
        def construct(self, **kwargs):
            return kwargs["a"] + kwargs.get("b")

    assert all(Net()(a=Tensor([1, 2, 3]), b=Tensor(2), c=3) == Tensor([3, 4, 5]))
    ms_grad = GradOperationNet(Net(), get_all=True)(a=Tensor(1), b=Tensor(2), c=3)
    assert len(ms_grad) == 2
    assert ms_grad[0] == 1 and ms_grad[1] == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_args_var_kwargs_002():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    class Net(nn.Cell):
        def construct(self, **kwargs):
            return kwargs

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    assert Net()() == {}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_args_var_mixed_001():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    @jit
    def return_x(*, x, **y):
        return x + y["c"]

    @jit
    def func(a=3, **kwargs):
        x = return_x(x=Tensor([1]), c=a)
        return kwargs["b"] + x

    out = func(a=Tensor(3), b=Tensor(5))
    assert out == 9


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_args_var_mixed_002():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    class SubNet(nn.Cell):
        def construct(self, *, x, y):
            return x - y

    class Net(nn.Cell):
        def construct(self, a, *args, b, c=2, **kwargs):
            if args[0] >= 0:
                out1 = a + len(args) + b - c + kwargs["d"] + 1
                out2 = SubNet()(y=a, x=b)
            else:
                out1 = out2 = Tensor(0)
            return out1 + out2

    net = Net()
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    out = net(a, Tensor(0), b=b, d=Tensor(3))
    assert all(out == Tensor([5, 5, 5]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_args_var_kwargs_index_exception():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    class Net(nn.Cell):
        def construct(self, *args, a=5, **kwargs):
            return args[-2] + args[1]

    net = Net()
    with pytest.raises(IndexError):
        net(*[5])

    class Net1(nn.Cell):
        def construct(self, *args, a=5, **kwargs):
            return kwargs["c"]

    net1 = Net1()
    with pytest.raises((KeyError, ValueError)):
        net1(b=3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_args_var_kwargs_grad():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    class Net(nn.Cell):
        def construct(self, **kwargs):
            return kwargs["a"] + kwargs["b"]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.grad = ops.grad(net, grad_position=(0, 1))

        def construct(self, **kwargs):
            return self.grad(**kwargs)

    class Net1(nn.Cell):
        def construct(self, *, a, b):
            return a + b

    @jit
    def grad_kwargs(a, b):
        out = ops.grad(Net1(), grad_position=0)(a=a, b=b)
        return out

    if context.get_context("mode") == context.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            GradNet(Net())(a=Tensor(3), b=Tensor(5))
    else:
        out = GradNet(Net())(a=Tensor(3), b=Tensor(5))
        assert out == (1, 1)
        out1 = grad_kwargs(Tensor(1), Tensor(2))
        assert all(out1 == Tensor([1, 1]))


@jit
def func1(x=0, y=0, z=0):
    return x * 100 + y * 10 + z


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_1():
    """
    Feature: Support the kwargs.
    Description: Use positional arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func1(1, 2, 3)

    assert output == 123


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_2():
    """
    Feature: Support the kwargs.
    Description: Use positional arguments for default parameters.
                 Number of passed parameters is less than interface parameters.
    Expectation: Expect the correct output.
    """
    output = func1(1, 2)

    assert output == 120


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_6():
    """
    Feature: Support the kwargs.
    Description: Mix positional and keyword arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func1(1, 2, z=3)

    assert output == 123


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_7():
    """
    Feature: Support the kwargs.
    Description: Use out-of-order keyword arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func1(x=1, z=3, y=2)

    assert output == 123


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_8():
    """
    Feature: Support the kwargs.
    Description: Use keyword arguments for default parameters. Mismatched parameter count.
    Expectation: Expect the correct output.
    """
    output = func1(x=1, z=3)

    assert output == 103


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_9():
    """
    Feature: Support the kwargs.
    Description: Use keyword arguments for default parameters. Mismatched parameter count.
    Expectation: Expect the correct output.
    """
    output = func1()

    assert output == 0


@jit
def func2(x, y=0, z=0):
    return x * 100 + y * 10 + z

@jit
def func3(*, x=0, y=0, z=0):
    return x * 100 + y * 10 + z

@jit
def func4(x, *, y=0, z=0):
    return x * 100 + y * 10 + z


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_14():
    """
    Feature: Support the kwargs.
    Description: Use out-of-order keyword arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func2(1, z=3, y=2)

    assert output == 123


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_15():
    """
    Feature: Support the kwargs.
    Description: Use positional arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func2(1, 2, 3)

    assert output == 123


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_16():
    """
    Feature: Support the kwargs.
    Description: Use out-of-order keyword arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func2(x=1, z=3, y=2)

    assert output == 123


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_19():
    """
    Feature: Support the kwargs.
    Description: Use out-of-order keyword arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func3(x=1, z=3)

    assert output == 103


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_with_kwargs_input_20():
    """
    Feature: Support the kwargs.
    Description: Use out-of-order keyword arguments for default parameters.
    Expectation: Expect the correct output.
    """
    output = func4(x=1, z=3)

    assert output == 103
