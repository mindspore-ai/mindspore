# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
""" syntax that JIT Fallback not support yet """

import math
from collections import deque
import pytest
import numpy as np

from mindspore import context, nn
from mindspore import Tensor, jit
from mindspore.common import mutable
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_interpret_object():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Do not support to return interpret object yet.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return [1, x, np.array([1, 2, 3, 4])]

    output = foo(Tensor([2]))
    assert len(output) == 3
    assert output[0] == 1
    assert output[1] == Tensor([2])
    assert np.all(output[2] == np.array([1, 2, 3, 4]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_raise_error_in_variable_scene():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Do not support to raise error in variable scene.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        if x == y:
            raise ValueError("x and y is equal")
        return x - y

    output = foo(Tensor([2]), Tensor([1]))
    assert output == Tensor([1])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_str_format_in_variable_scene():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Do not support to add variable in str.format.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        return "{}, {}".format(x, y)

    output = foo(Tensor([2]), Tensor([1]))
    assert output == "[2], [1]"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_numpy_asarray_with_variable_scene():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: np.asarray() function can not use variable as input.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = [x, y]
        return np.asarray(a)

    output = foo(mutable(1), mutable(2))
    assert np.all(output == np.array([1, 2]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_in_with_none():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: syntax 'in' do not support None input.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, None]
        return None in a

    assert foo()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_sequence_in_sequence():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: syntax 'in' do not support sequence in sequence check.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 2, [3, 4]]
        return [3, 4] in a

    assert foo()


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_all_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: the label in all statement can not be parsed
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = x.asnumpy()
        b = all(ele < y for ele in a)
        return b

    assert foo(Tensor([1, 2, 3, 4]), Tensor([10]))


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_slice_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Abstract slice can not in valid graph.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = slice(0, 3, 1)
        return x[a]

    out = foo(Tensor([1, 2, 3, 4, 5])).asnumpy()
    assert np.all(out.asnumpy() == Tensor([1, 2, 3]).asnumpy())


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_slice_with_mutable_input():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: name 'mutable' is not defined.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = slice(mutable(0), 3, 1)
        return a.step

    assert foo() == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_compress_with_mutable_input():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: the value including list is not correct.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        cond = [1, 0, 1, 1]
        a = [x, x + 1, x + 2, x + 3]
        z = np.compress(cond, a)
        return z

    assert (foo(Tensor([1])) == [1, 3, 4]).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_math_ceil_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: math.ceil(x) with variable x.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return math.ceil(x)

    ret = foo(mutable(10.75))
    assert ret == 11


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_unpack_interpret_node():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: * operator can not unpack a interpret object.
    Expectation: No exception.
    """

    def sum_func(a, b, c, d):
        return a + b + c + d

    def test(shape):
        reverse = reversed(shape)
        return sum_func(*reverse)

    @jit
    def foo(x):
        return test([1, 2, 3, 4])

    ret = foo([1, 2, 3, 4])
    assert ret == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_call_third_party_class():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: call third party class is not support in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        ret = deque()
        for i in x:
            ret.appendleft(i)
        return ret

    out = foo([1, 2, 3, 4])
    assert isinstance(out, deque)
    assert out == deque([4, 3, 2, 1])


@pytest.mark.skip(reason="kwargs with AbstractAny, fix later")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_np_ix_with_variable():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: call third party class is not support in graph mode.
    Expectation: No exception.
    """

    def convert(start, stop, step):
        shape = [5,]
        grids = ([np.array(list(range(start, stop, step)), dtype=np.int64)] +
                 [np.array(list(range(dim_size)), dtype=np.int64) for dim_size in shape[1:]])
        mesh = np.ix_(*grids)
        return np.stack(np.broadcast_arrays(*mesh), axis=-1)

    @jit
    def foo():
        return convert(mutable(0), mutable(5), mutable(1))

    ret = foo()
    assert (ret == [[0], [1], [2], [3], [4]]).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_generate_tensor_using_variable_numpy_array():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: When using variable numpy array to generate Tensor, numpy array type is wrong.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable(2)
        ret = np.arange(a)
        return Tensor(ret)

    out = foo()
    assert (out == Tensor([0, 1])).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_numpy_prod_with_variable_axis():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: numpy function do not support variable int input.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = x.asnumpy()
        return np.prod(a, axis=y)

    ret = foo(Tensor([[1, 2], [3, 4]]), mutable(1))
    assert np.all(ret == np.array([2, 12]))


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_with_interpret_object():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: interpret object can not get __len__ attribute.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = 0
        for i, j in zip(reversed(x), reversed(y)):
            a = a + i + j
        return a

    ret = foo([1, 2, 3], [4, 5, 6])
    assert ret == 21


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_with_interpret_object_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Value is wrong.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        a = []
        for i, j in list(zip(reversed(x), reversed(y))):
            a.append(i + j)
        return a

    ret = foo(mutable([1, 2, 3]), mutable([4, 5, 6]))
    assert ret == [9, 7, 5]


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_import_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support import in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_import():
        import numpy as inner_np  # pylint: disable=W0404
        x = inner_np.array(10, inner_np.float64)
        return x

    test_import_out = test_import()
    print("out:", test_import_out)


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_from_import_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support from and import in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_from_import(x):
        from mindspore.scipy.ops import Eig
        s, u = Eig()(x)
        return s, u

    context.set_context(device_target='CPU')
    x = Tensor(np.array([[1, 0], [0, 1]]).astype(np.float32))
    test_from_import_out = test_from_import(x)
    print("out:", test_from_import_out)


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_delete_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support delete in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_delete(x):
        y = x + 1
        z = y * 2
        del y
        return x, z

    test_delete_out = test_delete(2)
    print("out:", test_delete_out)


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_annassign_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support annassign in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_annassign(x):
        (y): int = x
        return y

    test_annassign_out = test_annassign(2)
    print("out:", test_annassign_out)


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_try_except_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support try except in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_try_except(x, y):
        global_out = 1
        try:
            global_out = x / y
        except ZeroDivisionError:
            print("division by zero, y is zero.")
        return global_out

    test_try_except_out = test_try_except(1, 0)
    print("out:", test_try_except_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_import_and_match_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support import match in graph mode.
    Expectation: No exception.
    """
    import re
    @jit
    def test_import_match():
        line = "Cats are smarter than dogs"
        search_obj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)

        if search_obj:
            print("search_obj.group() : ", search_obj.group())
            print("search_obj.group(1) : ", search_obj.group(1))
            print("search_obj.group(2) : ", search_obj.group(2))
        else:
            print("Nothing found!!")

    test_import_match_out = test_import_match()
    print("out:", test_import_match_out)


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_set_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support set in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_set():
        x = {1, 2, 3}
        return x

    test_set_out = test_set()
    assert test_set_out == {1, 2, 3}


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_set_comprehension_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support set comprehension in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_set_comprehension():
        x = {1, 2, 3}
        y = {i * i for i in x}
        return y

    test_set_comprehension_out = test_set_comprehension()
    assert test_set_comprehension_out == {1, 4, 9}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_comprehension_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support dict comprehension in graph mode.
    Expectation: No exception.
    """

    @jit
    def test_dict_comprehension():
        x = (1, 2, 3)
        y = {i: i * i for i in x}
        return y

    test_dict_comprehension_out = test_dict_comprehension()
    assert test_dict_comprehension_out == {1: 1, 2: 4, 3: 9}


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_yield_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support yield in graph mode.
    Expectation: No exception.
    """
    @jit
    def test_yield():
        def fab(max_num):
            n, a, b = 0, 0, 1
            while n < max_num:
                yield b
                a, b = b, a + b
                n = n + 1

        for n in fab(5):
            print(n)

    test_yield()


@pytest.mark.skip(reason="not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_yield_from_in_graph():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support yield from in graph mode.
    Expectation: No exception.
    """
    @jit
    def chain(*iterables):
        for i in iterables:
            yield from i

    s = "ABC"
    t = tuple(range(3))
    chain_out = list(chain(s, t))
    print("out:", chain_out)
    assert chain_out == ['A', 'B', 'C', 0, 1, 2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_assign_class_member():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support yield from in graph mode.
    Expectation: No exception.
    """
    class InnerNet(nn.Cell):
        def __init__(self, x):
            super(InnerNet, self).__init__()
            self.x = x

        def construct(self, x):
            return x + self.x

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.cell_list = nn.CellList()
            self.net = InnerNet(0)
            self.cell_list.append(self.net)

        def construct(self, x):
            self.cell_list[0].x = x
            return self.cell_list[0].x.shape

    with pytest.raises(RuntimeError, match="In graph mode, only attribute and name of class members can be assigned."):
        net = Net()
        x = Tensor([1, 2, 3])
        out = net(x)
        print("out:", out)
