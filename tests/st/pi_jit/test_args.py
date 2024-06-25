import pytest
import numpy as onp
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
import mindspore.nn as nn
from .share.utils import match_array
from tests.mark_utils import arg_mark


@jit(mode="PIJit")
def sum_args(a, b=1):
    return a + b


@jit(mode="PIJit")
def sum_args_vargs(a, *args, b=1):
    return a + b + args[0]


@jit(mode="PIJit")
def sum_args_vargs_kwargs(a, *args, b=1, **kwargs):
    return a + b + args[0] + kwargs["s"]


@jit(mode="PIJit")
def default_scalar_arg(a, b=1):
    return a, b


@jit(mode="PIJit")
def default_tuple_arg(a, b=(1, 2)):
    return a, b


@jit(mode="PIJit")
def default_scalar_arg_relu(a, b=1):
    relu = nn.ReLU()
    a = relu(a)
    b = relu(b)
    return a, b


@jit(mode="PIJit")
def default_none_arg(a, b=None):
    return a, b


@jit(mode="PIJit")
def parser_key_value_unsed(a, **kwargs):
    return a


def parser_one_default_arg_scalar_in_subnet():
    class SubNetDefaultScalarArg(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit(mode="PIJit")
        def construct(self, x, x1=4):
            if x1 == 4:
                x = self.relu(x)
            else:
                x = -x
            return x

    class NetOut(Cell):
        def __init__(self):
            super().__init__()
            self.net_inside = SubNetDefaultScalarArg()

        def construct(self, x):
            x = self.net_inside(x)
            return x

    net = NetOut()
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))
    out = net(tensor1)
    return out


def parser_one_default_arg_scalar_use():
    class NetDefaultScalarArg2(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit(mode="PIJit")
        def construct(self, x, x1=4):
            if x1 == 4:
                x = self.relu(x)
            else:
                x = -x
            return x

    net = NetDefaultScalarArg2()
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))
    out = net(tensor1)
    return out


def parser_one_default_arg_tensor_tuple():

    class NetDefaultTensorTupleArg(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit(mode="PIJit")
        def construct(self, x=(Tensor(onp.full((2, 3), 2).astype(onp.float32)),
                               Tensor(onp.full((3, 2), 4).astype(onp.float32)))):
            x = self.relu(x[0])
            x1 = self.relu(x[1])
            return x, x1

    class NetOut(Cell):
        def __init__(self):
            super().__init__()
            self.net_inside = NetDefaultTensorTupleArg()

        def construct(self):
            x, x1 = self.net_inside()
            return x, x1

    net = NetOut()
    out = net()
    return out


def parser_three_default_mixed_args_subnet():
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))
    tensor2 = Tensor(onp.full((3, 2), 4).astype(onp.float32))

    class SubNetDefaultMixedArgs(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit(mode="PIJit")
        def construct(self, y, x=3, x1=None, x2=(1, 2)):
            if x == 3:
                if x1 is None:
                    if x2[0] == 1:
                        y = self.relu(y)
                        return y

            return -y

    class NetOut(Cell):
        def __init__(self):
            super(NetOut, self).__init__()
            self.net_inside = SubNetDefaultMixedArgs()

        def construct(self, x, y=(tensor2,)):
            x = self.net_inside(x)

            return x, y

    net = NetOut()
    out = net(tensor1, tensor2)
    match_array(out[0].asnumpy(), tensor1.asnumpy(), error=0, err_msg=str(
        "parser_three_default_mixed_args_subnet match failed"))
    match_array(out[1].asnumpy(), tensor2.asnumpy(), error=0, err_msg=str(
        "parser_three_default_mixed_args_subnet match failed"))


def parser_key_value_not_tensor():
    class NetKeyValueArg(Cell):

        @jit(mode="PIJit")
        def construct(self, y, **x):
            if x["a"] == 5:
                y = y+y
            return y + x["b"][0]

    class Netout(Cell):
        def __init__(self):
            super().__init__()
            self.in_net = NetKeyValueArg()

        def construct(self, x):
            x = self.in_net(x, a=5, b=(x,))
            return x

    net = Netout()
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))

    out = net(tensor1)
    match_array(out.asnumpy(), tensor1.asnumpy()*3, error=0,
                err_msg=str("parser_key_value_not_tensor match failed"))


def parser_args_var_kwargs_empty():
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, **kwargs):
            return kwargs

    assert Net()() == {}


def parser_args_var_kwargs_add():
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, **kwargs):
            return kwargs["a"] + kwargs.get("b")

    dict_a = {"a": Tensor([1, 2, 3]), "b": Tensor([1])}
    assert all(Net()(**dict_a) == Tensor([2, 3, 4]))


def parser_args_var_mixed_001():

    def return_x(*, x, **y):
        return x + y["c"]

    @jit(mode="PIJit")
    def func(a=3, **kwargs):
        x = return_x(x=Tensor([1]), c=a)
        return kwargs["b"] + x

    out = func(a=Tensor(3), b=Tensor(5))
    assert out == 9


def parser_args_var_mixed_002():
    class SubNet(nn.Cell):
        def construct(self, *, x, y):
            return x - y

    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, a, *args, **kwargs):
            if args[0] >= 0:
                out1 = a + len(args) + kwargs["d"] + 1
                out2 = SubNet()(y=a, x=args[0])
            else:
                out1 = out2 = Tensor([0, 0, 0])
            return out1 + out2

    net = Net()
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    out = net(a, Tensor(0), b=b, d=Tensor(3))
    assert all(out == Tensor([5, 5, 5]))


@jit
def ms_sum_args(a, b=1):
    return a + b


@jit
def ms_sum_args_vargs(a, *args, b=1):
    return a + b + args[0]


@jit
def ms_sum_args_vargs_kwargs(a, *args, b=1, **kwargs):
    return a + b + args[0] + kwargs["s"]


@jit
def ms_default_scalar_arg(a, b=1):
    return a, b


@jit
def ms_default_tuple_arg(a, b=(1, 2)):
    return a, b


@jit
def ms_default_scalar_arg_relu(a, b=1):
    relu = nn.ReLU()
    a = relu(a)
    b = relu(b)
    return a, b


@jit
def ms_default_none_arg(a, b=None):
    return a, b


@jit
def ms_parser_key_value_unsed(a):
    return a


def ms_parser_one_default_arg_scalar_in_subnet():
    class SubNetDefaultScalarArg(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit
        def construct(self, x, x1=4):
            if x1 == 4:
                x = self.relu(x)
            else:
                x = -x
            return x

    class NetOut(Cell):
        def __init__(self):
            super().__init__()
            self.net_inside = SubNetDefaultScalarArg()

        def construct(self, x):
            x = self.net_inside(x)
            return x

    net = NetOut()
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))
    out = net(tensor1)
    return out


def ms_parser_one_default_arg_scalar_use():
    class NetDefaultScalarArg2(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit
        def construct(self, x, x1=4):
            if x1 == 4:
                x = self.relu(x)
            else:
                x = -x
            return x

    net = NetDefaultScalarArg2()
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))
    out = net(tensor1)
    return out


def ms_parser_one_default_arg_tensor_tuple():

    class NetDefaultTensorTupleArg(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit
        def construct(self, x=(Tensor(onp.full((2, 3), 2).astype(onp.float32)),
                               Tensor(onp.full((3, 2), 4).astype(onp.float32)))):
            x = self.relu(x[0])
            x1 = self.relu(x[1])
            return x, x1

    class NetOut(Cell):
        def __init__(self):
            super().__init__()
            self.net_inside = NetDefaultTensorTupleArg()

        def construct(self):
            x, x1 = self.net_inside()
            return x, x1

    net = NetOut()
    out = net()
    return out


def ms_parser_three_default_mixed_args_subnet():
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))
    tensor2 = Tensor(onp.full((3, 2), 4).astype(onp.float32))

    class SubNetDefaultMixedArgs(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit
        def construct(self, y, x=3, x1=None, x2=(1, 2)):
            if x == 3:
                if x1 is None:
                    if x2[0] == 1:
                        y = self.relu(y)
                        return y

            return -y

    class NetOut(Cell):
        def __init__(self):
            super(NetOut, self).__init__()
            self.net_inside = SubNetDefaultMixedArgs()

        def construct(self, x, y=(tensor2,)):
            x = self.net_inside(x)

            return x, y

    net = NetOut()
    out = net(tensor1, tensor2)
    match_array(out[0].asnumpy(), tensor1.asnumpy(), error=0, err_msg=str(
        "ms_parser_three_default_mixed_args_subnet match failed"))
    match_array(out[1].asnumpy(), tensor2.asnumpy(), error=0, err_msg=str(
        "ms_parser_three_default_mixed_args_subnet match failed"))


def ms_parser_key_value_not_tensor():
    class NetKeyValueArg(Cell):

        @jit
        def construct(self, y, **x):
            if x["a"] == 5:
                y = y+y
            return y + x["b"][0]

    class Netout(Cell):
        def __init__(self):
            super().__init__()
            self.in_net = NetKeyValueArg()

        def construct(self, x):
            x = self.in_net(x, a=5, b=(x,))
            return x

    net = Netout()
    tensor1 = Tensor(onp.full((2, 3), 2).astype(onp.float32))

    out = net(tensor1)
    match_array(out.asnumpy(), tensor1.asnumpy()*3, error=0,
                err_msg=str("ms_parser_key_value_not_tensor match failed"))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [sum_args])
@pytest.mark.parametrize('ms_func', [ms_sum_args])
@pytest.mark.parametrize('a', [1])
def test_arg1(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [sum_args])
@pytest.mark.parametrize('ms_func', [ms_sum_args])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [10])
def test_arg2(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [sum_args_vargs])
@pytest.mark.parametrize('ms_func', [ms_sum_args_vargs])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [10])
@pytest.mark.parametrize('c', [100])
def test_arg_vargs(func, ms_func, a, b, c):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b, c)
    ms_res = ms_func(a, b, c)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [sum_args_vargs_kwargs])
@pytest.mark.parametrize('ms_func', [ms_sum_args_vargs_kwargs])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [10])
@pytest.mark.parametrize('c', [100])
@pytest.mark.parametrize('d', [1000])
def test_arg_vargs_kwargs(func, ms_func, a, b, c, d):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b, c, s=d)
    ms_res = ms_func(a, b, c, s=d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [default_scalar_arg])
@pytest.mark.parametrize('ms_func', [ms_default_scalar_arg])
@pytest.mark.parametrize('a', [Tensor(onp.full((2, 3), 2).astype(onp.float32))])
@pytest.mark.parametrize('b', [Tensor(onp.full((3, 2), 4).astype(onp.float32))])
def test_default_scalar_arg(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a scalar default arg, but pass a tensor to it
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [default_tuple_arg])
@pytest.mark.parametrize('ms_func', [ms_default_tuple_arg])
@pytest.mark.parametrize('a', [Tensor(onp.full((2, 3), 2).astype(onp.float32))])
@pytest.mark.parametrize('b', [Tensor(onp.full((3, 2), 4).astype(onp.float32))])
def test_default_tuple_arg(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a tuple default arg, but pass a tensor to it
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [default_scalar_arg_relu])
@pytest.mark.parametrize('ms_func', [ms_default_scalar_arg_relu])
@pytest.mark.parametrize('a', [Tensor(onp.full((2, 3), 2).astype(onp.float32))])
@pytest.mark.parametrize('b', [Tensor(onp.full((3, 2), 4).astype(onp.float32))])
def test_default_scalar_arg_relu(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a scalar default arg, but pass a tensor to it and do relu
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [default_none_arg])
@pytest.mark.parametrize('ms_func', [ms_default_none_arg])
@pytest.mark.parametrize('a', [Tensor(onp.full((2, 3), 2).astype(onp.float32))])
@pytest.mark.parametrize('b', [Tensor(onp.full((3, 2), 4).astype(onp.float32))])
def test_default_none_arg(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a none default arg, but pass a tensor to it
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_key_value_unsed])
@pytest.mark.parametrize('ms_func', [ms_parser_key_value_unsed])
@pytest.mark.parametrize('a', [Tensor(onp.full((2, 3), 2).astype(onp.float32))])
def test_parser_key_value_unsed(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a key-value arg, and do not use it and do not pass a key-value to it
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_one_default_arg_scalar_in_subnet])
@pytest.mark.parametrize('ms_func', [ms_parser_one_default_arg_scalar_in_subnet])
def test_parser_arg_scalar_subnet(func, ms_func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a scalar default arg and use it in sub-network
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_one_default_arg_scalar_use])
@pytest.mark.parametrize('ms_func', [ms_parser_one_default_arg_scalar_use])
def test_parser_arg_scalar_outside(func, ms_func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
TEST_SUMMARY:define a scalar default arg, use it in outside network
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_one_default_arg_tensor_tuple])
@pytest.mark.parametrize('ms_func', [ms_parser_one_default_arg_tensor_tuple])
def test_parser_arg_tensor_subnet(func, ms_func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY:define a default tensor tuple arg, use it in sub-network
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    ms_res = ms_func()
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_three_default_mixed_args_subnet])
@pytest.mark.parametrize('ms_func', [ms_parser_three_default_mixed_args_subnet])
def test_parser_three_default_arg_tensor_subnet(func, ms_func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: no error
    TEST_SUMMARY:define a scalar default arg, a none arg and a tuple arg, and use them in sub-network
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func()
    ms_func()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_key_value_not_tensor])
@pytest.mark.parametrize('ms_func', [ms_parser_key_value_not_tensor])
def test_parser_key_value1(func, ms_func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: no error
    TEST_SUMMARY:define a key-value arg, and use it in sub-network, pass two non-tensor data to it
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func()
    ms_func()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_args_var_kwargs_empty])
def test_parser_key_value2(func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: no error
    TEST_SUMMARY:  变长关键字参数  接收dict作为输入
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_args_var_kwargs_add])
def test_parser_key_value3(func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: no error
    TEST_SUMMARY:  命名关键字参数  *, a, b
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_args_var_mixed_001])
def test_parser_key_value4(func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: no error
    TEST_SUMMARY:  位置参数 可变参数 命名关键字  不定长关键字  非顶层Cell
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [parser_args_var_mixed_002])
def test_parser_key_value5(func):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: no error
    TEST_SUMMARY:   命名关键字不命名
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func()
