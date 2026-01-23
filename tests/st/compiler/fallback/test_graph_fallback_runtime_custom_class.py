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
""" test graph JIT Fallback runtime feature """
import pytest
import random
import torch
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import ops, nn, Tensor, jit_class
import mindspore.common.dtype as mstype
from mindspore import mutable
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import match_array

ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_cust_class():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = 99
            self.attr2 = 1

        def method1(self, x):
            return x + self.attr2

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.cls = GetattrClass()

        def construct(self):
            return self.cls.method1(self.cls.attr1)

    net = GetattrClassNet()
    out = net()
    assert out == 100


class ClassTest:
    """ ClassTest definition """

    def __init__(self, name, value1):
        self.name = name
        self.value = value1

    def __call__(self, *args, **kwargs):
        pass

    def get_name(self):
        return self.name

    def get_value(self, inc):
        ret = self.value + inc
        return ret


class SelfObjectGetattrNet(ms.nn.Cell):
    """ SelfObjectGetattrNet definition """

    def __init__(self, v1, v2):
        super().__init__()
        self.relu = ms.nn.ReLU()
        self.softmax = ms.nn.Softmax(0)
        self.axis = 0
        self.test_class = ClassTest("test_class", v1)
        self.value = v2

    @ms.jit
    def construct(self, x):
        x = x + self.test_class.get_value(self.value)
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.25)
def test_call_other_object_method_runtime():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int32))
    y = ms.Tensor(np.array([[2, 3, 4], [1, 1, 2]]).astype(np.int32))
    y1 = ms.Tensor(np.array([[5, 4, 5], [1, 1, 2]]).astype(np.int32))
    z = np.array([[8, 9, 12], [3, 4, 7]]).astype(np.int32)

    net = SelfObjectGetattrNet(y, y1)
    output = net.construct(x)
    result = output.asnumpy()
    assert np.all(result == z)


# Test: call global object method(not self) on parse graph code
value = ms.Tensor(np.array([[3, 4, 5], [1, 1, 2]]).astype(np.int32))
test_class = ClassTest("test_class", value)


class GlobalObjectGetattrNet(ms.nn.Cell):
    """ GlobalObjectGetattrNet definition """

    def __init__(self, value1):
        super().__init__()
        self.value = value1

    @ms.jit
    def construct(self, x):
        x = x + test_class.get_value(self.value)
        return x

    @ms.jit
    def construct1(self, x):
        x = x + test_class.value
        x = x + self.value
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_call_no_self_other_object_method_runtime():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int32))
    y = ms.Tensor(np.array([[2, 3, 4], [1, 1, 2]]).astype(np.int32))
    z = np.array([[6, 9, 12], [3, 4, 7]]).astype(np.int32)

    net = GlobalObjectGetattrNet(y)
    output = net.construct(x)
    result = output.asnumpy()
    assert np.all(result == z)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parser_fallback_nested_class_outer():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support custom class input.
    Expectation: AttributeError.
    """

    class Inner:
        def __init__(self):
            self.number = ms.Tensor(2, dtype=ms.int32)

        def act(self, x, y):
            return self.number * (x + y)

    @ms.jit_class
    class InnerNet:
        def __init__(self):
            self.inner = Inner()

    class NestedNet(ms.nn.Cell):
        @ms.jit
        def construct(self, x, y):
            out = InnerNet().inner.act(x, y)
            return out

    x = 2
    y = 4
    net = NestedNet()
    assert net(x, y) == 12


class UserDefinedNet:
    def __init__(self):
        self.value = 10

    def __call__(self, x):
        return self.value * x


class UserDefinedMsFunctionCallNet:
    def __init__(self):
        self.value = 10

    @ms.jit
    def __call__(self, x):
        return self.value * x


class UNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        out = self.net(x)
        out = out + out
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_resolve_cust_class():
    """
    Feature: Syntax resolve.
    Description: Graph syntax resolve support custom class input.
    Expectation: No error.
    """
    net = UNet(UserDefinedNet())
    x = np.array([10], np.float32)
    output = net(ms.Tensor(x))
    assert output == 200


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_resolve_cust_ms_function_call_class():
    """
    Feature: Syntax resolve.
    Description: support device sync.
    Expectation: No error.
    """
    context.set_context(jit_level='O0')
    net = UNet(UserDefinedMsFunctionCallNet())
    x = np.array([10, 10], np.float32)
    net(ms.Tensor(x))


class OuterNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        out = self.net(x)
        return out


class UserDefinedTupleNet:
    def __init__(self):
        self.value = 10

    def __call__(self, x):
        return self.value * x, 100


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_pyexecute_with_stub_tensor():
    """
    Feature: Fallback runtime.
    Description: The output of pyexecute is not allow to have stub tensor.
    Expectation: No error.
    """
    context.set_context(jit_level='O0')
    net = OuterNet(UserDefinedTupleNet())
    x = np.array([10], np.float64)
    output = net(ms.Tensor(x))
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert output[0] == 100
    assert output[1] == 100


class UserDefinedListNet:
    def __init__(self):
        self.value = 10

    def __call__(self, x):
        return [self.value * x, 100]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_pyexecute_with_stub_tensor_2():
    """
    Feature: Fallback runtime.
    Description: The output of pyexecute is not allow to have stub tensor.
    Expectation: No error.
    """
    context.set_context(jit_level='O0')
    net = OuterNet(UserDefinedListNet())
    x = np.array([10], np.float64)
    output = net(ms.Tensor(x))
    assert isinstance(output, list)
    assert len(output) == 2
    assert output[0] == 100
    assert output[1] == 100


class UserDefinedDictNet:
    def __init__(self):
        self.value = 10

    def __call__(self, x):
        return {"100": self.value * x}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_pyexecute_with_stub_tensor_3():
    """
    Feature: Fallback runtime.
    Description: The output of pyexecute is not allow to have stub tensor.
    Expectation: No error.
    """
    context.set_context(jit_level='O0')
    net = OuterNet(UserDefinedDictNet())
    x = np.array([10], np.float64)
    output = net(ms.Tensor(x))
    assert isinstance(output, dict)
    assert output["100"] == 100


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parser_fallback_nested_class_outer_grad():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support custom class input.
    Expectation: AttributeError.
    """

    class Inner:
        def __init__(self):
            self.number = ms.Tensor(2, dtype=ms.int32)

        def act(self, x, y):
            return self.number * (x + y)

    @ms.jit_class
    class InnerNet:
        def __init__(self):
            self.inner = Inner()

    class NestedNet(ms.nn.Cell):
        @ms.jit
        def construct(self, x, y):
            out = InnerNet().inner.act(x, y)
            return out

    x = 2
    y = 4
    net = NestedNet()
    # pylint: disable=E1102
    output = ops.grad(net)(mutable(x), y)
    assert output == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_create_custom_class_default():
    """
    Feature: Create custom class instance.
    Description: Graph syntax getattr support create custom class instance.
    Expectation: No exception.
    """

    class InnerNet:
        def __init__(self):
            self.number = 2

        def act(self, x, y):
            return self.number * (x + y)

    class Net(ms.nn.Cell):
        def construct(self, x, y):
            out = InnerNet().act(x, y)
            return out

    x = ms.Tensor(1)
    y = ms.Tensor(2)
    net = Net()
    out = net(x, y)
    assert out == 6


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_create_custom_class_args():
    """
    Feature: Create custom class instance.
    Description: Graph syntax getattr support create custom class instance.
    Expectation: No exception.
    """

    class InnerNet:
        def __init__(self, number):
            self.number = number

        def act(self, x, y):
            return self.number * (x + y)

    class Net(ms.nn.Cell):
        def construct(self, x, y):
            out = InnerNet(x).act(x, y)
            return out

    x = ms.Tensor(2)
    y = ms.Tensor(4)
    net = Net()
    out = net(x, y)
    assert out == 12


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_getattr_cust_class_const():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = 99
            self.attr2 = 1

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.cls = GetattrClass()

        def construct(self, x):
            if self.cls.attr2 == 1:
                return x * 2
            return x + self.cls.attr2

    net = GetattrClassNet()
    x = 99
    out = net(x)
    assert out == 198


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_custom_class_jit():
    """
    Feature: Syntax resolve.
    Description: Graph syntax resolve support custom class input.
    Expectation: No error.
    """

    class InnerNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.value = 10

        @ms.jit
        def construct(self, x):
            return self.value + x

    class CustomNet():
        def __init__(self, model):
            super().__init__()
            self.model = model

        def __call__(self, x):
            return self.model(2 * x)

    class OutNet(ms.nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x):
            return self.net(x)

    x = ms.Tensor(2)
    call_net = InnerNet()
    custom_net = CustomNet(call_net)
    out_net = OutNet(custom_net)
    out = out_net(x)
    assert out == 14


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_kwargs_is_custom_class_attr():
    """
    Feature: Support the kwargs is any.
    Description: Graph syntax resolve support custom class input is kwargs.
    Expectation: No error.
    """

    class Config:
        def __init__(self, **kwargs):
            self.aaa = kwargs.pop("aaa", 2.0)

    class Model(ms.nn.Cell):
        def construct(self, input1, input2):
            return input1 * input2

    class Net(ms.nn.Cell):
        def __init__(self, net_config):
            super().__init__()
            self.config = net_config
            self.model = Model()

        def construct(self, x):
            return self.model(input1=x, input2=self.config.aaa)

    config = Config()
    net = Net(config)
    output = net(x=ms.Tensor(3))
    assert output == 6


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_class_getattr_and_custom_class_may_raise_exception():
    """
    Feature: attr is a custom-type object.
    Description: self.config is a custom-type object.
    Expectation: No error.
    """

    class Config:
        def __init__(self):
            self.k = 5

        def __str__(self):
            # The jit fallback may invoke the `__str__` method of a Python object.
            # The framework should catch exceptions when calling `__str__`; otherwise, the compilation will fail.
            raise NotImplementedError("Not support __str__")

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.config = Config()

        def construct(self, x: Tensor, k=None):
            k = k if k is not None else self.config.k  # self.config will be converted to InterpretedObject
            return x + k

    model = Model()
    a = Tensor([1, 2, 3])
    o = model(a)
    match_array(o, Tensor([6, 7, 8]))


class GradNet(nn.Cell):
    def __init__(self, net, grad_position=0):
        super().__init__()
        self.grad_net = ops.grad(net, grad_position=grad_position)  # pylint: disable=E1102

    def construct(self, *x):
        return self.grad_net(*x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_parse_node_conver_bit():
    """
    Feature: subclass is custom class.
    Description: Get attr form a custom class.
    Expectation: No error.
    """

    class SubClass:
        x = np.array(2)
        y = np.array(-1)

    class InnerClass(nn.Cell):
        def __init__(self, sub):
            super().__init__()
            self.subclass = sub

        def construct(self, x):
            a = x << self.subclass.x
            b = a >> self.subclass.y
            c = self.subclass.x & self.subclass.y
            d = b | c
            e = a ^ d
            return e

    class TouchNet(torch.nn.Module):
        def __init__(self, sub):
            super().__init__()
            self.subclass = sub

        def forward(self, x):
            a = x << self.subclass.x
            b = a >> self.subclass.y
            c = self.subclass.x & self.subclass.y
            d = b | c
            e = a ^ d
            return e

    x = random.choice([random.randint(-5, 10), 100])
    ms_out = InnerClass(SubClass())(x)
    torch_out = TouchNet(SubClass())(x)

    assert np.allclose(torch_out, ms_out, 1e-5, 1e-5)
    ms_grad = GradNet(InnerClass(SubClass()))(x)
    assert ms_grad == ()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_parse_node_conver_enhance_arithmetic():
    """
    Feature: subclass is custom class.
    Description: Get attr form a custom class.
    Expectation: No error.
    """

    class SubClass:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])

    class InnerClass(nn.Cell):
        def __init__(self, sub):
            super().__init__()
            self.subclass = sub

        def construct(self, x, y):
            x = x.asnumpy()
            y = y.asnumpy()
            y = y + self.subclass.x
            y = y - self.subclass.y
            z = y[0]
            return x, y, z

    class TouchNet(torch.nn.Module):
        def __init__(self, sub):
            super().__init__()
            self.subclass = sub

        def forward(self, x, y):
            y = y + self.subclass.x
            y = y - self.subclass.y
            z = y[0]
            return x, y, z

    x = np.random.randn(3).astype(np.float32)
    y = np.random.randn(3).astype(np.float32)
    context.set_context(jit_level='O0')
    ms_out = InnerClass(SubClass())(Tensor(x), Tensor(y))
    torch_out = TouchNet(SubClass())(x, y)
    for tt, mt in zip(torch_out, ms_out):
        assert np.allclose(tt, mt, 1e-5, 1e-5)
    grad_net = GradNet(InnerClass(SubClass()), grad_position=(0, 1))
    grad_net(Tensor(x), Tensor(y))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_class_abnormal_variable_input():
    """
    Feature: subclass is custom class.
    Description: Get attr form a custom class.
    Expectation: No error.
    """

    @jit_class
    class InnerNet:
        def __init__(self, val):
            self.number = val

    class Net(nn.Cell):
        def construct(self, x):
            inner_net = InnerNet(x)
            return inner_net.number

    x = Tensor(2, mstype.int32)
    net = Net()
    with pytest.raises(TypeError):
        net(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_class_abnormal_call_magic_method():
    """
    Feature: subclass is custom class.
    Description: Get attr form a custom class.
    Expectation: No error.
    """

    @jit_class
    class InnerNet:
        def __init__(self, val):
            self.val = val

    class Net(nn.Cell):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def construct(self):
            obj = InnerNet(self.val)
            out = obj.__str__()
            return obj, out

    context.set_context(jit_level='O0')
    net = Net(2)
    out = net()
    res = out[0].__str__()
    assert out[1] == res
