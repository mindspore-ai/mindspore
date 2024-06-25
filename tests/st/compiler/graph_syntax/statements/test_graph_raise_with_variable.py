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
""" test graph raise """
# pylint: disable=R1705
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor, context, jit
from mindspore import dtype as mstype
from mindspore.ops.operations._inner_ops import TopTypeof
from mindspore._extends.parse import compile_config
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_1():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x > 10:
                raise ValueError(f"The input can not be {x}.")

    compile_config.FALLBACK_SUPPORT_LIST_DICT_INPLACE = 1
    with pytest.raises(ValueError) as raise_info_9:
        net = RaiseNet()
        x = Tensor(11)
        res = net(x)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_9.value)
    compile_config.FALLBACK_SUPPORT_LIST_DICT_INPLACE = 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise(string % var).
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x > 10:
                raise ValueError(f"The input can not be %s." % x)

    with pytest.raises(ValueError) as raise_info_10:
        net = RaiseNet()
        res = net(Tensor(11))
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_10.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_3():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x > 10:
                raise ValueError(f"The input can not be ", x, ".")

    with pytest.raises(ValueError) as raise_info_11:
        net = RaiseNet()
        res = net(Tensor(11))
        print("res:", res)
    assert "('The input can not be ', Tensor(shape=[], dtype=Int64, value= 11), '.')" or \
        "('The input can not be ', Tensor(shape=[1], dtype=Int64, value= [11]), '.')" in str(
            raise_info_11.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_list():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, y):
            x = [Tensor(1), Tensor(2), Tensor(3), Tensor(4)]
            if y > 10:
                raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_list:
        net = RaiseNet()
        y = Tensor(11)
        res = net(y)
        print("res:", res)
    assert "[Tensor(shape=[], dtype=Int64, value= 1)," or \
        "(Tensor(shape=[1], dtype=Int64, value= [1])," in str(
            raise_info_list.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_tuple_1():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, y):
            x = (Tensor(1), Tensor(2), Tensor(3), Tensor(4))
            if y > 10:
                raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_tuple:
        net = RaiseNet()
        y = Tensor(11)
        res = net(y)
        print("res:", res)
    assert "(Tensor(shape=[], dtype=Int64, value= 1)," or \
        "(Tensor(shape=[1], dtype=Int64, value= [1])," in str(
            raise_info_tuple.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_tuple_2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, y):
            x = (Tensor(1), Tensor(2), Tensor(3), Tensor(4))
            if y > 10:
                raise ValueError("test_string_tuple", x)

    with pytest.raises(ValueError) as raise_info_string_tuple:
        net = RaiseNet()
        y = Tensor(11)
        res = net(y)
        print("res:", res)
    assert "('test_string_tuple', (Tensor(shape=[], dtype=Int64, value= 1)" or \
        "('test_string_tuple', (Tensor(shape=[1], dtype=Int64, value= [1])" in str(
            raise_info_string_tuple.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_joinedstr_tensor():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x > 0:
                raise RuntimeError(f"The input should not be {x}.")

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        res = net(x)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@pytest.mark.skip(reason='Not support dict yet')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_with_variable_dic():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, z):
            x = Tensor(1)
            y = Tensor(2)
            z = {"x": x, "y": y}
            if z["x"] > 10:
                raise ValueError(z)

    with pytest.raises(ValueError) as raise_info_list:
        net = RaiseNet()
        z = Tensor(11)
        res = net(z)
        print("res:", res)
    assert "{'x': Tensor(shape=[], dtype=Int64, value= 1)" or \
        "{'x': Tensor(shape=[1], dtype=Int64, value= [1])" in str(
            raise_info_list.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_control_flow1():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y):
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_variable_control_flow2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y):  # pylint: disable=R1711
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")
            return None

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


def _raise_func(x):
    raise ValueError(x)


def _check_test(shp, x):
    def _check(shp, x):
        if shp[0] > 3:
            _raise_func(f"Check failed. Wrong shape, {x}.")
        return True
    ret = _check(shp, x)
    ms.ops.stop_gradient(ret)


class CheckNet(ms.nn.Cell):
    def __init__(self):
        super(CheckNet, self).__init__()
        self.one = ms.Tensor(1, dtype=ms.float32)

    def construct(self, x):
        shp = x.shape
        _check_test(shp, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_isolated_raise():
    """
    Feature: Isolated raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    np_data = np.random.randint(6, size=(4,))
    data = ms.Tensor(np_data, dtype=ms.float32)
    net = CheckNet()
    with pytest.raises(ValueError) as err:
        net(data)
    assert "Check failed. Wrong shape," in str(err.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_in_control_flow():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, y, z):
            if z >= 1:
                raise ValueError(f"The input maybe {y}")

    with pytest.raises(ValueError) as raise_info_list:
        y = [Tensor(1), Tensor(2), Tensor(3)]
        net = RaiseNet()
        z = Tensor(1)
        net(y, z)
    assert "The input maybe [" in str(raise_info_list.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_none_join():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y):  # pylint: disable=R1711
            if x != y:
                return None
            raise RuntimeError(f"The input should not be {x}.")

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_raise_join():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y):  # pylint: disable=R1711
            if x > y:
                raise RuntimeError(f"The input {x} should not greater {y}.")
            if x == y:
                raise RuntimeError(f"The input {x} should not equal {y}.")
            return None

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input 1 should not equal 1" in str(
        raise_info_joinedstr_tensor.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_parse_with_interpret():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y, z):  # pylint: disable=R1711
            if z >= 1:
                raise TypeError(f"x: {type(x)}, y: {y}, z: {z}")
            return None

    input_x = [Tensor([1, 2, 3]), Tensor([4, 5, 6])]
    input_y = [Tensor([1]), Tensor([2]), Tensor([3])]
    input_z = Tensor(3)
    net = RaiseNet()
    with pytest.raises(TypeError) as raise_info_joinedstr_tensor:
        net(input_x, input_y, input_z)
    assert "x:" in str(raise_info_joinedstr_tensor.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_parse_with_interpret_2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y, z):  # pylint: disable=R1711
            if z >= 1:
                raise TypeError(f"x: {type(x)}, y: {y}, z: {z}")
            return None

    input_x = [Tensor([1, 2, 3]), Tensor([4, 5, 6])]
    input_y = [Tensor([1]), Tensor([2]), Tensor([3])]
    input_z = Tensor(0)
    net = RaiseNet()
    assert net(input_x, input_y, input_z) is None


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_input_error_type_1():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y=ValueError):
            if x > 10:
                raise y(f"The input can not be {x}.")

    with pytest.raises(ValueError) as raise_info:
        net = RaiseNet()
        x = Tensor(11)
        res = net(x)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_raise_with_input_error_type_2():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            y = ValueError
            if x > 10:
                raise y(f"The input can not be {x}.")

    with pytest.raises(ValueError) as raise_info:
        net = RaiseNet()
        x = Tensor(11)
        res = net(x)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_raise_join_in_control_flow():
    """
    Feature: graph raise.
    Description: Test raise join in control flow.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        if y < x:
            raise ValueError("The input should not be ", x)
        return x + y

    x = Tensor([1], dtype=mstype.int32)
    y = Tensor([2], dtype=mstype.int32)
    res = foo(x, y)
    assert res == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_raise_join_in_control_flow_2():
    """
    Feature: graph raise.
    Description: Test raise join in control flow.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        out = x
        if y > x:
            out = x + y
        elif x == y:
            raise ValueError("The input should not be ", y)
        return out

    x = Tensor([1], dtype=mstype.int32)
    y = Tensor([2], dtype=mstype.int32)
    res = foo(x, y)
    assert res == 3


class SimpleCellReLu(nn.Cell):
    def construct(self, x):
        return nn.ReLU()(x)


class SimpleCellRaise(nn.Cell):
    def construct(self, x):
        raise ValueError("The input should not be ", x)


class CellInList(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(SimpleCellReLu())
        self.cell_list.append(SimpleCellRaise())
        self.cell_list.append(SimpleCellRaise())

    def construct(self, index, x):
        return self.cell_list[index](x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_cell_in_list():
    """
    Feature: graph raise.
    Description: Test raise join in control flow(switch_layer).
    Expectation: No exception.
    """
    net = CellInList()
    x = Tensor(np.ones((1, 1, 224, 224)), mstype.float64)
    idx = Tensor(0, mstype.int32)
    out = net(idx, x)
    relu_func = nn.ReLU()
    true_value = relu_func(x)
    ret = np.allclose(out.asnumpy(), true_value.asnumpy())
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_raise_constant_folding():
    """
    Feature: graph raise.
    Description: Test raise join in control flow.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        if x > 10:
            raise ValueError(f"The input can not be {x}.")
        return 1.0

    with pytest.raises(ValueError) as raise_info_constant:
        x = Tensor(11)
        res = foo(x)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_constant.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_raise_constant_folding_int64():
    """
    Feature: graph raise.
    Description: Test raise join in control flow.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        if x > 10:
            raise ValueError(f"The input can not be {x}.")
        return 1

    with pytest.raises(ValueError) as raise_info_constant_int64:
        x = Tensor(11)
        res = foo(x)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_constant_int64.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_assert_tensor_join_assert():
    """
    Feature: graph raise.
    Description: Test raise join in control flow.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, x, y):
            output = self.add(x, y)
            assert output == Tensor(8, ms.int32), f"The output is {output}, y is {y}"
            return output

    x = Tensor(2, ms.int32)
    y = Tensor(3, ms.int32)
    with pytest.raises(AssertionError) as err:
        net = Net()
        net(x, y)
    assert "The output is 5, y is 3" in str(err)


def judge_tuple_index_dim_check_error(index_dim, data_dim, x):
    if index_dim > data_dim:
        raise IndexError(f"The dim of index cannot be greater than indexed data, but got "
                         f"dim of index:{index_dim}, dim of data:{data_dim}, {x}")


def judge_tuple_index_dim(data, tuple_index, x):
    data_dim = data.ndim
    index_dim = 0
    for index in tuple_index:
        if isinstance(TopTypeof()(index), mstype.TensorType) and index.dtype == mstype.bool_:
            index_dim += index.ndim
        else:
            index_dim += 1
    judge_tuple_index_dim_check_error(index_dim, data_dim, x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_raise_in_sub_func_graph_with_isolate_node():
    """
    Feature: graph raise.
    Description: Test raise isolate node in sub graph.
    Expectation: No exception.
    """
    @ms.jit
    def bool_index(data_input, index_input, x):
        tuple_index = (0, index_input)
        judge_tuple_index_dim(data_input, tuple_index, x)
        return data_input

    with pytest.raises(IndexError) as err:
        index = Tensor([[0, 1], [0, 1]], dtype=ms.bool_)
        data = Tensor([[0, 1], [2, 3]])
        output = bool_index(data, index, Tensor([1]))
        print(output)
    assert "The dim of index cannot be greater than indexed data" in str(err)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_in_method():
    """
    Feature: graph raise.
    Description: Test raise in graph mode.
    Expectation: No exception.
    """
    class NetRaiseInMethod(nn.Cell):
        def construct(self, x, y, z):
            if x == 1:
                return Tensor(10, mstype.int32)
            elif x == 20:
                raise ValueError('Illegal case')
            else:
                return y + z

    net = NetRaiseInMethod()
    x = Tensor(0, mstype.int32)
    y = Tensor(5, mstype.int32)
    z = Tensor(2, mstype.int32)
    out = net(x, y, z)
    assert out == 7
