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
import pytest
import numpy as np

import mindspore as ms
from mindspore.common.initializer import TruncatedNormal
from mindspore import ops
from mindspore import mutable

ms.set_context(mode=ms.GRAPH_MODE)


class ConstNet(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())

    def construct(self):
        a = ms.Tensor(np.array(4), ms.int32)
        b = ms.Tensor(np.array(5), ms.int32)
        return self.np_function(a, b)


class Net(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())

    def np_function2(self, a, b):
        x = np.exp(a.asnumpy())
        y = np.exp(b.asnumpy())
        return np.exp(x + y)

    def construct(self, a, b):
        return self.np_function(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net()(a, b)
    const_output = ConstNet()()
    np.testing.assert_almost_equal(output, const_output, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np_grad():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = ops.grad(Net())(a, b)
    assert output == 0


class Net1(ms.nn.Cell):
    def np_function(self, a, b):
        x = a.asnumpy()
        y = b.asnumpy()
        return np.exp(x + y)

    def construct(self, a, b):
        return self.np_function(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net1()(a, b)
    const_output = ConstNet()()
    np.testing.assert_almost_equal(output, const_output, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np_asnumpy_grad():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = ops.grad(Net1())(a, b)
    assert output == 0


@ms.jit
def tensor_asnumpy():
    tensor = ms.Tensor(np.arange(0, 6).reshape(2, 3))
    res = tensor.asnumpy()
    return res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_jit_tensor_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    res = tensor_asnumpy()
    print(res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_return_1():
    """
    Feature: Return dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_1():
        x = {'a': 'a', 'b': 'b'}
        y = x.get('a')
        z = dict(y=y)
        return z

    out = dict_net_1()
    assert out == {'y': 'a'}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_empty_dict_pyexecute():
    """
    Feature: Return empty dict
    Description: Return empty dict.
    Expectation: No error.
    """
    @ms.jit
    def dict_func():
        return {}

    with pytest.raises(RuntimeError):
        x = dict_func()
        assert x == {}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_return_2():
    """
    Feature: Return dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_2():
        x = {'a': 1, 'b': 2}
        y = x.get('a')
        y_tensor = ms.Tensor([y])
        z = dict(a=y_tensor)
        return z

    out = dict_net_2()
    assert out == {'a': ms.Tensor(np.array(1), ms.int64)}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_get_2():
    """
    Feature: Return dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_2():
        x = {'a': 1, 'b': 2}
        y = x.get('a')
        y_tensor = ms.Tensor([y])
        z = dict(a=y_tensor, b='hello', c='world')
        return z

    out = dict_net_2()
    assert out == {'a': ms.Tensor(np.array(1), ms.int64), 'b': 'hello', 'c': 'world'}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_get_3():
    """
    Feature: Return dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_3():
        x = {'a': 1, 'b': 2}
        y = x.get('a')
        y_tensor = ms.Tensor([y])
        z = dict(y=y_tensor, a='a', b='c')
        return z

    out = dict_net_3()
    assert out == {'y': ms.Tensor(np.array(1), ms.int64), 'a': 'a', 'b': 'c'}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multiple_return_contains_dict():
    """
    Feature: Return multiple outputs including dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_2():
        x = {'a': 1, 'b': 2}
        y = x.get('a')
        y_tensor = ms.Tensor([y])
        z = dict(a=y_tensor)
        return y, z, (1, 2)

    out = dict_net_2()
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] == {'a': ms.Tensor(np.array(1), ms.int64)}
    assert out[2] == (1, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multiple_return_contains_dict_2():
    """
    Feature: Return multiple outputs including dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_2(a):
        x = {'a': a, 'b': 2}
        return a, (x, (1, 2))

    out = dict_net_2(ms.Tensor([1]))
    assert len(out) == 2
    assert out[0] == ms.Tensor([1])
    assert len(out[1]) == 2
    assert out[1][0] == {'a': ms.Tensor([1], ms.int64), 'b': 2}
    assert out[1][1] == (1, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multiple_return_contains_dict_2_grad():
    """
    Feature: Return multiple outputs including dict.
    Description: Support grad for dict return.
    Expectation: Get expected gradient.
    """
    @ms.jit
    def dict_net_2(a):
        x = {'a': a, 'b': 2}
        return a, (x, (1, 2))

    out = ops.grad(dict_net_2)(ms.Tensor([1]))
    assert out == 2


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    return ms.nn.Conv2d(in_channels, out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        weight_init="ones", has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    return ms.nn.Dense(input_channels, out_channels, "ones", "ones")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net_dict_1():
    """
    Feature: Return dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    class DictLeNetNet(ms.nn.Cell):
        def __init__(self, num_class=10):
            super(DictLeNetNet, self).__init__()
            self.conv1 = conv(1, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
            self.fc2 = fc_with_initialize(120, 84)
            self.fc3 = fc_with_initialize(84, 10)
            self.relu = ms.nn.ReLU()
            self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = ms.nn.Flatten()

        def construct(self, x):
            x = self.conv1(x)
            conv1_x = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            conv2_x = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            fc_x = x
            outputs = dict(conv1=conv1_x, conv2=conv2_x, fc=fc_x)
            return outputs

    net = DictLeNetNet()
    x = ms.Tensor(np.random.rand(64, 1, 32, 32).astype(np.float32))
    outputs = net(x)
    assert outputs['conv1'].shape == (64, 6, 28, 28)
    assert outputs['conv2'].shape == (64, 16, 10, 10)
    assert outputs['fc'].shape == (64, 10)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net_dict_1_grad():
    """
    Feature: Return dict.
    Description: Support grad for dict return.
    Expectation: Get expected gradient.
    """
    class DictLeNetNet(ms.nn.Cell):
        def __init__(self, num_class=10):
            super(DictLeNetNet, self).__init__()
            self.conv1 = conv(1, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
            self.fc2 = fc_with_initialize(120, 84)
            self.fc3 = fc_with_initialize(84, 10)
            self.relu = ms.nn.ReLU()
            self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = ms.nn.Flatten()

        def construct(self, x):
            x = self.conv1(x)
            conv1_x = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            conv2_x = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            fc_x = x
            outputs = dict(conv1=conv1_x, conv2=conv2_x, fc=fc_x)
            return outputs

    net = DictLeNetNet()
    x = ms.Tensor(np.random.rand(64, 1, 32, 32).astype(np.float32))
    outputs = ops.grad(net)(x)
    assert np.all(outputs.asnumpy() == np.zeros((64, 1, 32, 32)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net_dict_2():
    """
    Feature: Return dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    class DictLeNetNet(ms.nn.Cell):
        def __init__(self, num_class=10):
            super(DictLeNetNet, self).__init__()
            self.conv1 = conv(1, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
            self.fc2 = fc_with_initialize(120, 84)
            self.fc3 = fc_with_initialize(84, 10)
            self.relu = ms.nn.ReLU()
            self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = ms.nn.Flatten()

        def construct(self, x):
            outputs = dict()
            x = self.conv1(x)
            outputs['conv1'] = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            outputs['conv2'] = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            outputs['fc'] = x
            return outputs

    net = DictLeNetNet()
    x = ms.Tensor(np.random.rand(64, 1, 32, 32).astype(np.float32))
    outputs = net(x)
    assert outputs['conv1'].shape == (64, 6, 28, 28)
    assert outputs['conv2'].shape == (64, 16, 10, 10)
    assert outputs['fc'].shape == (64, 10)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net_dict_2_grad():
    """
    Feature: Return dict.
    Description: Support grad for dict return.
    Expectation: Get expected gradients.
    """
    class LeNet(ms.nn.Cell):
        def __init__(self, num_class=10):
            super(LeNet, self).__init__()
            self.conv1 = conv(1, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
            self.fc2 = fc_with_initialize(120, 84)
            self.fc3 = fc_with_initialize(84, 10)
            self.relu = ms.nn.ReLU()
            self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = ms.nn.Flatten()

        def construct(self, x):
            x = self.conv1(x)
            output_conv1 = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            output_conv2 = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            output_fc = x
            return output_conv1, output_conv2, output_fc

    class DictLeNet(ms.nn.Cell):
        def __init__(self, num_class=10):
            super(DictLeNet, self).__init__()
            self.conv1 = conv(1, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
            self.fc2 = fc_with_initialize(120, 84)
            self.fc3 = fc_with_initialize(84, 10)
            self.relu = ms.nn.ReLU()
            self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = ms.nn.Flatten()

        def construct(self, x):
            outputs = dict()
            x = self.conv1(x)
            outputs['conv1'] = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            outputs['conv2'] = x
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            outputs['fc'] = x
            return outputs

    x = ms.Tensor(np.random.rand(64, 1, 32, 32).astype(np.float32))
    net = LeNet()
    outputs1 = ops.grad(net)(x)
    dict_lenet = DictLeNet()
    outputs2 = ops.grad(dict_lenet)(x)
    assert np.allclose(outputs1.asnumpy(), outputs2.asnumpy())


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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


class PrintPyExecuteNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        out = x * x
        print("out1:", out)
        out = self.net(x) + out
        print("out2:", out)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_pyexecute():
    """
    Feature: Side effect in Fallback runtime.
    Description: Side effect in Fallback runtime.
    Expectation: No error.
    """
    net = PrintPyExecuteNet(UserDefinedNet())
    x = np.array([10], np.float64)
    output = net(ms.Tensor(x))
    print(output)
    assert output == 200


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
