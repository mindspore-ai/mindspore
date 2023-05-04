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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import mutable

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
            super(GetattrClassNet, self).__init__()
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
        super(SelfObjectGetattrNet, self).__init__()
        self.relu = ms.nn.ReLU()
        self.softmax = ms.nn.Softmax(0)
        self.axis = 0
        self.test_class = ClassTest("test_class", v1)
        self.value = v2

    @ms.jit
    def construct(self, x):
        x = x + self.test_class.get_value(self.value)
        return x


@pytest.mark.skip(reason="Stuck by ScopedLongRunning() invocation in forward.cc during JIT Fallback Python running.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
        super(GlobalObjectGetattrNet, self).__init__()
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


@pytest.mark.skip(reason="Stuck by ScopedLongRunning() invocation in forward.cc during JIT Fallback Python running.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
        out = x * x
        out = self.net(x)
        out = out + out
        return out


@pytest.mark.skip(reason="No support PyExecute Add.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resolve_cust_class():
    """
    Feature: Syntax resolve.
    Description: Graph syntax resolve support custom class input.
    Expectation: No error.
    """
    net = UNet(UserDefinedNet())
    x = np.array([10], np.float32)
    output = net(ms.Tensor(x))
    print(output)
    assert output == 200


@pytest.mark.skip("PyExecute node can not be used in meta fg.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resolve_cust_ms_function_call_class():
    """
    Feature: Syntax resolve.
    Description: Graph syntax resolve support custom class input.
    Expectation: No error.
    """
    net = UNet(UserDefinedMsFunctionCallNet())
    x = np.array([10, 10], np.float32)
    with pytest.raises(RuntimeError) as err:
        net(ms.Tensor(x))
    assert "Nested execution during JIT execution is not supported." in str(err.value)


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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_stub_tensor():
    """
    Feature: Fallback runtime.
    Description: The output of pyexecute is not allow to have stub tensor.
    Expectation: No error.
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_stub_tensor_2():
    """
    Feature: Fallback runtime.
    Description: The output of pyexecute is not allow to have stub tensor.
    Expectation: No error.
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_stub_tensor_3():
    """
    Feature: Fallback runtime.
    Description: The output of pyexecute is not allow to have stub tensor.
    Expectation: No error.
    """
    net = OuterNet(UserDefinedDictNet())
    x = np.array([10], np.float64)
    output = net(ms.Tensor(x))
    assert isinstance(output, dict)
    assert output["100"] == 100


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="Cast fail from F.zeros_like")
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
    output = ops.grad(net)(mutable(x), y)
    assert output == 0
