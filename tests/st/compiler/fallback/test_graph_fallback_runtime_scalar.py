# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test graph JIT Fallback runtime feature """
import math
import mindspore as ms
from mindspore.ops import composite as C
from tests.mark_utils import arg_mark


class GradNet(ms.nn.Cell):
    def __init__(self, network, get_all=False, get_by_list=False):
        super().__init__()
        self.network = network
        self.grad = C.GradOperation(get_all, get_by_list)

    def construct(self, *inputs):
        grads = self.grad(self.network)(*inputs)
        return grads


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_return_scalar():
    """
    Feature: Return scalar.
    Description: Support return scalar type.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, y):
        return x + y

    out1 = func(ms.mutable(1), ms.mutable(2))
    out2 = func(ms.mutable(3), ms.mutable(4.0))
    out3 = func(ms.mutable(5.0), ms.mutable(6.0))
    assert isinstance(out1, int) and out1 == 3
    assert isinstance(out2, float) and abs(out2 - 7) < 1e-6
    assert isinstance(out3, float) and abs(out3 - 11) < 1e-6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_return_scalar_tuple():
    """
    Feature: Return scalar.
    Description: Support return scalar type.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, y):
        return x + y, x - y, x * y

    out = func(ms.mutable(6), ms.mutable(4))
    assert isinstance(out[0], int) and out[0] == 10
    assert isinstance(out[1], int) and out[1] == 2
    assert isinstance(out[2], int) and out[2] == 24


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_builtin_int():
    """
    Feature: Return scalar.
    Description: Support builtin function int().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x)

    x = ms.Tensor(1)
    out = func(x)
    print(f'out: {out}')
    assert isinstance(out, int) and out == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_builtin_float():
    """
    Feature: Return scalar.
    Description: Support builtin function float().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return float(x)

    x = ms.Tensor(1.0)
    out = func(x)
    print(f'out: {out}')
    assert isinstance(out, float) and math.isclose(out, 1, abs_tol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_builtin_bool():
    """
    Feature: Return scalar.
    Description: Support builtin function bool().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return bool(x)

    x = ms.Tensor(1)
    out = func(x)
    print(f'out: {out}')
    assert out is True


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_builtin_scalar_grad():
    """
    Feature: Return scalar.
    Description: Test scalar grad.
    Expectation: No exception.
    """
    class Net(ms.nn.Cell):
        def construct(self, x):
            out = int(x), float(x), bool(x)
            return out

    x = ms.Tensor(1)
    net = Net()
    grad = GradNet(net)
    out_grad = grad(x)
    assert out_grad == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scalar_in_tuple_output():
    """
    Feature: Return scalar.
    Description: Support builtin function.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x), float(x)

    x = ms.Tensor(10)
    out = func(x)
    print(f'out: {out}')
    assert isinstance(out[0], int) and out[0] == 10
    assert isinstance(out[1], float) and math.isclose(out[1], 10, abs_tol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_int_asnumpy():
    """
    Feature: Return scalar.
    Description: Support tensor.asnumpy().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x.asnumpy())

    x = ms.Tensor([5])
    out = func(x)
    print(f'out: {out}')
    assert out == 5


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_int_asnumpy_calculation():
    """
    Feature: Return scalar.
    Description: Support tensor.asnumpy().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x.asnumpy()) + 1

    x = ms.Tensor([5])
    out = func(x)
    print(f'out: {out}')
    assert out == 6


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_int_tensor_asnumpy_calculation():
    """
    Feature: Return scalar.
    Description: Support tensor.asnumpy().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(ms.Tensor(x.asnumpy())) + 1

    x = ms.Tensor([5])
    out = func(x)
    print(f'out: {out}')
    assert out == 6


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_int_mutable():
    """
    Feature: Return scalar.
    Description: Support mutable.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x)

    out = func(ms.mutable(1))
    print(f'out: {out}')
    assert isinstance(out, int) and out == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_float_mutable():
    """
    Feature: Return scalar.
    Description: Support mutable.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return float(x)

    out = func(ms.mutable(1.0))
    print(f'out: {out}')
    assert isinstance(out, float) and math.isclose(out, 1, abs_tol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_bool_condition():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        if bool(x):
            return x + 1
        return x - 1

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert out == 6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_int_condition():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        if int(x) == 5:
            return x + 2
        return x + 3

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert out == 7


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_float_condition():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        if float(x):
            return x - 2
        return x - 3

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert out == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_condition():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        if ms.Tensor(int(x)):
            return x * 2
        return x * 3

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert out == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_bool_asnumpy():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return bool(x.asnumpy())

    x = ms.Tensor(5)
    out = func(x)
    assert out is True


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_bool_asnumpy_condition():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        if bool(x.asnumpy()):
            return x * 2
        return x * 3

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert out == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scalar_int_calculation():
    """
    Feature: Return scalar.
    Description: Support scalar calculation. ScalarAdd does not support Ascend platform now.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x) + 1

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert isinstance(out, int) and out == 6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_combine_calculation():
    """
    Feature: Return scalar.
    Description: Support scalar calculation.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return int(x) + float(x), int(float(x) + 1.5), float(int(x) + 1.5)

    x = ms.Tensor(5)
    out = func(x)
    print(f'out: {out}')
    assert isinstance(out[0], float) and abs(out[0] - 10) < 1e-6
    assert isinstance(out[1], int) and out[1] == 6
    assert isinstance(out[2], float) and abs(out[2] - 6.5) < 1e-6


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scalar_in_inner_function():
    """
    Feature: Return scalar.
    Description: Support scalar in output list.
    Expectation: No exception.
    """
    def inner_func(x):
        if bool(x):
            return int(x) + 1
        return int(x) + 2

    @ms.jit
    def func(x):
        return inner_func(x)

    x = ms.Tensor(10)
    out = func(x)
    print(f'out: {out}')
    assert out == 11


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_scalar_tuple_in_inner_function():
    """
    Feature: Return scalar.
    Description: Support scalar in output list.
    Expectation: No exception.
    """
    def inner_func(x):
        if bool(x):
            return int(x) + 1, x
        return int(x) + 2, x * 2

    @ms.jit
    def func(x):
        return inner_func(x)

    x = ms.Tensor(10)
    out = func(x)
    print(f'out: {out}')
    assert out[0] == 11


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scalar_in_list():
    """
    Feature: Return scalar.
    Description: Support scalar in output list.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return [1, 2, 3, 4]

    out = func()
    print(f'out: {out}')
    assert out == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scalar_in_dict():
    """
    Feature: Return scalar.
    Description: Support scalar in output dict.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return dict(x='x', y=2)

    out = func()
    print(f'out: {out}')
    assert out == {'x': 'x', 'y': 2}
    assert isinstance(out.get('y'), int)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scalar_in_dict_with_int_value():
    """
    Feature: Return scalar.
    Description: Support scalar in output dict.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return dict(x=1, y=2)

    out = func()
    print(f'out: {out}')
    assert out == {'x': 1, 'y': 2}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scalar_in_dict_with_tuple_value():
    """
    Feature: Return scalar.
    Description: Support scalar in output dict.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return dict(x=(1, 2), y=(3, 4))

    out = func()
    print(f'out: {out}')
    assert out == {'x': (1, 2), 'y': (3, 4)}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scalar_in_dict_with_empty_tuple():
    """
    Feature: Return scalar.
    Description: Support scalar in output dict.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return dict(x=1, y=(), z=2)

    out = func()
    print(f'out: {out}')
    assert out == {'x': 1, 'y': (), 'z': 2}
