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
""" test graph JIT Fallback runtime feature """
import os
import math
from functools import reduce
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, tensor
from mindspore import ops
from mindspore import mutable, jit
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_tensor_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo(ms.Tensor([-1, -2, -3]))  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_list_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo([1, 2, 3, 4])  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_tuple_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "shape")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo((1, 2, 3, 4))  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_dict_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo({"1": 1, "2": 2})  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_with_scalar_input():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """
    def _check_is_inf_nan(x):
        if math.isinf(x) or math.isnan(x) or np.isinf(x) or np.isnan(x):
            return True
        return False

    class InnerNet(nn.Cell):
        def construct(self, x):
            return _check_is_inf_nan(x.shape[0])

    net = InnerNet()
    data = Tensor(np.random.randint(6, size=(2, 4, 3, 4, 5)), dtype=ms.float32)
    dyn = Tensor(shape=[None, None, None, None, None], dtype=ms.float32)
    net.set_inputs(dyn)
    ret = net(data)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_with_scalar_input_2():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """
    def _check_is_inf_nan(x):
        if math.isinf(x) or math.isnan(x) or np.isinf(x) or np.isnan(x):
            return True
        return False

    class InnerNet(nn.Cell):
        def construct(self, x):
            return _check_is_inf_nan(x)

    net = InnerNet()
    ret = net(math.inf)
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_with_scalar_input_3():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            shp = x.shape
            return all(i < 3 for i in shp)

    net = InnerNet()
    data = Tensor(np.random.randint(6, size=(2, 4, 3, 4, 5)), dtype=ms.float32)
    dyn = Tensor(shape=[None, None, None, None, None], dtype=ms.float32)
    net.set_inputs(dyn)
    ret = net(data)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_with_scalar_input_4():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            shp = x.shape
            return any(i < 3 for i in shp)

    net = InnerNet()
    data = Tensor(np.random.randint(6, size=(2, 4, 3, 4, 5)), dtype=ms.float32)
    dyn = Tensor(shape=[None, None, None, None, None], dtype=ms.float32)
    net.set_inputs(dyn)
    ret = net(data)
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_as_multitype_fg_input():
    """
    Feature: Fallback runtime.
    Description: Pyexecute node can not be used as multitype function graph.
    Expectation: No error.
    """
    class sub_class:
        def __getitem__(self, item):
            pass
        def __setitem__(self, key, target):
            pass


    class InnerNet(nn.Cell):
        def __init__(self, tuple_input):
            super(InnerNet, self).__init__()
            self.data = tuple_input

        def construct(self, start):
            return self.data[start:]

    sub_class_obj = sub_class()
    sub_class_obj[0] = [1, 2, 3, 4, 5]
    net = InnerNet(sub_class_obj)
    assert net(1) is None


def user_mul(x, y):
    return x * y


@ms.jit
def reduce_user_mul(x):
    out = reduce(user_mul, x)
    return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_with_func_graph_input():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has FuncGraph input.
    Expectation: No error.
    """
    x1 = (1, 2, 3)
    x2 = mutable((1, 2, 3), False)
    ret1 = reduce_user_mul(x1)
    ret2 = reduce_user_mul(x2)
    assert ret1 == 6
    assert ret2 == 6


@pytest.mark.skip('backend not support different type in value tuple')
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_anytype():
    """
    Feature: Fallback runtime.
    Description: test ops input is PyExecute out
    Expectation: No error.
    """

    @jit
    def func(x):
        x = x.asnumpy()
        x = ms.Tensor(x)
        x = ops.ReLU()(x)
        return x

    def func_numpy(x):
        return np.maximum(x, 0)

    x_np = np.array([1, -1])
    ms_out = func(ms.Tensor(np.array([1, -1])))
    np_out = func_numpy(x_np)
    assert np.allclose(np_out, ms_out.asnumpy())


class CreateDynTensor(nn.Cell):
    def construct(self, x):
        # @jit.typing: () -> tensor_type[int32]
        shape_tensor1 = Tensor(ms.mutable(ops.shape(x)), ms.int32)
        output1 = ops.FillV2()(shape_tensor1, Tensor(1, ms.int32))

        shape_tensor2 = Tensor(ms.mutable(ops.shape(x)), ms.int32)  # @jit.typing: () -> tensor_type[int32]
        output2 = ops.FillV2()(shape_tensor2, Tensor(1, ms.int32))
        return output1 + output2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_dynamic_shape_tensor():
    """
    Feature: Fallback runtime.
    Description: Set PyExecute output type by the annotation from comment.
    Expectation: No error.
    """
    net = CreateDynTensor()
    x = Tensor(dtype=ms.int32, input_data=[2, 2])
    out = net(x)
    return out


class CreateNotDynTensor(nn.Cell):
    def construct(self, x):
        # ops.shape(x) is a constant, should not convert to PyExecute.
        shape_tensor1 = Tensor(ops.shape(x), ms.int32)
        output1 = ops.FillV2()(shape_tensor1, Tensor(1, ms.int32))

        shape_tensor2 = Tensor(ops.shape(x), ms.int32)
        output2 = ops.FillV2()(shape_tensor2, Tensor(1, ms.int32))
        return output1 + output2


@pytest.mark.skip('ops.shape(x) is constant, not mutable.')
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_not_dynamic_shape_tensor():
    """
    Feature: Fallback runtime.
    Description: Not convert to PyExecute.
    Expectation: No error.
    """
    net = CreateNotDynTensor()
    x = Tensor(dtype=ms.int32, input_data=[2, 2])
    out = net(x)
    return out


class CreateDynTensorWithInputDtype(nn.Cell):
    def construct(self, x, dtype):
        # @jit.typing: () -> tensor_type[{dtype}]
        shape_tensor1 = Tensor(ms.mutable(ops.shape(x)), dtype)
        output1 = ops.FillV2()(shape_tensor1, Tensor(1, dtype))

        shape_tensor2 = Tensor(ms.mutable(ops.shape(x)), dtype)  # @jit.typing: () -> tensor_type[{dtype}]
        output2 = ops.FillV2()(shape_tensor2, Tensor(1, ms.int32))
        return output1 + output2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_dynamic_shape_dtype_tensor():
    """
    Feature: Fallback runtime.
    Description: Set PyExecute output type by the annotation from comment.
    Expectation: No error.
    """
    net = CreateDynTensorWithInputDtype()
    x = Tensor(dtype=ms.int32, input_data=[2, 2])
    out = net(x, ms.int32)
    return out


class MakeTensorAsConstant(ms.nn.Cell):
    def construct(self, x):
        shape_tensor1 = ms.tensor(ops.shape(x), ms.int32)
        output1 = ops.FillV2()(shape_tensor1, ms.Tensor(1, ms.int32))

        shape_tensor2 = ms.tensor(ops.shape(x), ms.int32)
        output2 = ops.FillV2()(shape_tensor2, ms.Tensor(1, ms.int32))
        return output1 + output2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_tensor_as_constant():
    """
    Feature: Fallback runtime.
    Description: Test tensor API, create constant Tensor on compile time.
    Expectation: No error.
    """
    net = MakeTensorAsConstant()
    x = ms.Tensor(dtype=ms.int32, input_data=[2, 2])
    out = net(x)
    return out


class MakeTensorWithShapeDtype(nn.Cell):
    def construct(self, x):
        dtype = ms.int32
        shape_tensor1 = ms.tensor(ms.mutable(ops.shape(x)), dtype)  # shape is mutable, so dtype is used in RT.
        output1 = ops.FillV2()(shape_tensor1, Tensor(1, dtype))

        shape_tensor2 = ms.tensor(ms.mutable(ops.shape(x)), dtype)
        output2 = ops.FillV2()(shape_tensor2, Tensor(1, ms.int32))
        return output1 + output2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_make_tensor_with_dynamic_shape_dtype():
    """
    Feature: Fallback runtime.
    Description: Test tensor API, in which the PyExecute output type is set by the annotation from comment.
    Expectation: No error.
    """
    net = MakeTensorWithShapeDtype()
    x = Tensor(dtype=ms.int32, input_data=[2, 2])
    out = net(x)
    return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_gelu():
    """
    Feature: Fallback runtime.
    Description: Set PyInterpret output type by the annotation from comment.
    Expectation: No error.
    """
    @ms.jit
    def gelu_forward_1(x):
        # @jit.typing: () -> tensor_type[float32]
        return 0.5 * x * (1 + ms.ops.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * ms.ops.pow(x, 3))))

    @ms.jit
    def gelu_forward_2(x):
        math_var = math.sqrt(2 / math.pi)
        pow_var = ms.ops.pow(x, 3)
        var1 = 0.044715 * pow_var
        var2 = x + var1
        var3 = math_var * var2  # @jit.typing: () -> tensor_type[float32]
        tanh_var = ms.ops.tanh(var3)
        return 0.5 * x * (1 + tanh_var)

    @ms.jit
    def gelu_forward_3(x):
        math_var = math.sqrt(2 / math.pi)
        pow_var = ms.ops.pow(x, 3)
        var1 = 0.044715 * pow_var
        var2 = x + var1
        var3 = math_var * var2  # No @jit.typing
        tanh_var = ms.ops.tanh(var3)
        return 0.5 * x * (1 + tanh_var)

    x = ms.Tensor(9, dtype=ms.float32)
    res1 = gelu_forward_1(x)
    res2 = gelu_forward_2(x)
    res3 = gelu_forward_3(x)
    assert np.all(res1.asnumpy() == res2.asnumpy())
    assert np.all(res1.asnumpy() == res3.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_np_save():
    """
    Feature: Fallback runtime.
    Description: Use numpy.save().
    Expectation: No error.
    """
    @ms.jit
    def func(x):
        if isinstance(x, ms.Tensor):
            np.save("x_data.npy", x.asnumpy())

    x = ms.Tensor([1, 2, 3])
    func(x)
    x_load = np.load("x_data.npy")
    assert np.all(x_load == x.asnumpy())
    os.remove("x_data.npy")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_np_save_with_args():
    """
    Feature: Fallback runtime.
    Description: Test numpy.save() and isolated side effect for top args.
    Expectation: No error.
    """
    def _save_tensor(data):
        np.save("data_from_args.npy", data.asnumpy())

    class NpSaveWithArgsNet(nn.Cell):
        def construct(self, *args):
            x = args[0]
            _save_tensor(x)
            return x

    x = ms.Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NpSaveWithArgsNet()
    output = net(x)
    print(f'output: {output}')
    x_load = np.load("data_from_args.npy")
    assert np.all(x_load == x.asnumpy())
    os.remove("data_from_args.npy")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_np_save_with_call_kw1():
    """
    Feature: Fallback runtime.
    Description: Test numpy.save() and isolated side effect for calling kw function.
    Expectation: No error.
    """
    def _save_tensor(data):
        np.save("data_from_kw.npy", data.asnumpy())

    class NpSaveWithCallKw(nn.Cell):
        def construct(self, x):
            _save_tensor(data=x)
            return x

    x = ms.Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NpSaveWithCallKw()
    output = net(x)
    print(f'output: {output}')
    x_load = np.load("data_from_kw.npy")
    assert np.all(x_load == x.asnumpy())
    os.remove("data_from_kw.npy")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_np_save_with_call_kw2():
    """
    Feature: Fallback runtime.
    Description: Test numpy.save() and isolated side effect for calling kw function with if.
    Expectation: No error.
    """
    def _save_tensor_with_if(data):
        if True:  # pylint: disable=using-constant-test
            np.save("data_from_kw_with_if.npy", data.asnumpy())

    class NpSaveWithCallKw(nn.Cell):
        def construct(self, x):
            _save_tensor_with_if(data=x)
            return x

    x = ms.Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NpSaveWithCallKw()
    output = net(x)
    print(f'output: {output}')
    x_load = np.load("data_from_kw_with_if.npy")
    assert np.all(x_load == x.asnumpy())
    os.remove("data_from_kw_with_if.npy")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_pyexecute_raise_error_with_dynamic_length_sequence():
    """
    Feature: Fallback runtime.
    Description: Pyexecute node can not be used as multitype function graph.
    Expectation: No error.
    """
    def _check_dim_shape_valid(data, tensor_index):
        if data.shape[:tensor_index.ndim] != tensor_index.shape[:]:
            raise IndexError(f"The shape of index {tensor_index.shape} does not match the shape "
                             f"of the indexed data {data.shape}")

    class InnerNet(nn.Cell):
        def construct(self, x):
            idx1 = Tensor([[True, False], [False, True], [True, True]])
            idx2 = Tensor([True, True, True, False])
            indices = idx1.nonzero()
            x1 = ops.gather_nd(x, indices)
            _check_dim_shape_valid(x1, idx2)
            return x1

    net = InnerNet()
    input_x = Tensor(np.arange(6).reshape(3, 2).astype(np.float32))
    ret = net(input_x)
    assert np.allclose(ret.asnumpy(), np.array([0.0, 3.0, 4.0, 5.0]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_pyexecute_raise_error_with_dynamic_length_sequence_2():
    """
    Feature: Fallback runtime.
    Description: Pyexecute node can not be used as multitype function graph.
    Expectation: No error.
    """
    def _check_dim_shape_valid(data, tensor_index):
        if data.shape[:tensor_index.ndim] == tensor_index.shape[:]:
            raise IndexError(f"The shape of index {tensor_index.shape} does not match the shape "
                             f"of the indexed data {data.shape}")

    class InnerNet(nn.Cell):
        def construct(self, x):
            idx1 = Tensor([[True, False], [False, True], [True, True]])
            idx2 = Tensor([True, True, True, False])
            indices = idx1.nonzero()
            x1 = ops.gather_nd(x, indices)
            _check_dim_shape_valid(x1, idx2)
            return x1

    with pytest.raises(IndexError) as err:
        net = InnerNet()
        input_x = Tensor(np.arange(6).reshape(3, 2).astype(np.float32))
        net(input_x)
    assert "does not match the shape" in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parse_subscript():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in subscript in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = [Tensor([11]), Tensor([22]), Tensor([33])]
            y = x[Tensor([0])] + x[Tensor([1])] + x[Tensor([2])]
            return y

    net = Network()
    out = net()
    assert out.asnumpy() == 66


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_func():
    """
    Feature: JIT Fallback
    Description: Test tensor function in graph mode.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    @jit
    def func(x):
        x = tensor(x.asnumpy(), x.dtype)
        return ops.Abs()(x)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.abs = ops.Abs()

        def construct(self, x, y):
            y1 = tensor(x.asnumpy() + y.asnumpy(), dtype=ms.float32)
            return self.abs(y1)

    x = Tensor([-1, 1, 2, -2], dtype=ms.float32)
    x_np = np.array([1, 1, 2, 2], dtype=np.float32)
    x = func(x)
    ms_x_np = x.asnumpy()
    assert ms_x_np.dtype == x_np.dtype
    assert np.allclose(ms_x_np, x_np)

    net = Net()
    x = ms.Tensor(-1, dtype=ms.int32)
    y = ms.Tensor(-1, dtype=ms.float32)
    result = net(x, y)
    result_ms_np = result.asnumpy()
    exp_np = np.array(2, dtype=np.float32)
    assert np.allclose(result_ms_np, exp_np)
    assert result_ms_np.dtype == exp_np.dtype


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_self_variable_as_func_args():
    """
    Feature: JIT Fallback
    Description: Use self as variable name
    Expectation: No exception
    """

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.value = 5

        def construct(self, x, y):
            return Network.func(self, x, y)

        def func(self, x, y):
            return x + y

    net = Network()
    out = net(Tensor([1], dtype=ms.float32), Tensor([2], dtype=ms.float32))
    assert np.allclose(out.asnumpy(), np.array([3], dtype=np.float32))


@pytest.mark.skip(reason="ParseCall convert whole script to pyexecute")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_tensor_with_variable_input():
    """
    Feature: JIT Fallback
    Description: Generate Tensor with graph.
    Expectation: No exception
    """

    @jit
    def foo(x):
        return Tensor([0], dtype=x.dtype)

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    ret = foo(Tensor([1, 2, 3]))
    assert ret == Tensor([0])
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_map_with_variable_input():
    """
    Feature: JIT Fallback
    Description: Generate Tensor with graph.
    Expectation: No exception
    """

    @jit
    def foo(x, y):
        m = map(lambda a, b: a + b, x.asnumpy(), y.asnumpy())
        return tuple(m)

    ret = foo(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
    assert ret == (5, 7, 9)
