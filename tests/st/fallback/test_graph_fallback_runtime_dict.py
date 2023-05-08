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

ms.set_context(mode=ms.GRAPH_MODE)


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
def test_multiple_return_contains_dict_3():
    """
    Feature: Return multiple outputs including dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net_3():
        return None, {"a": 1}

    out = dict_net_3()
    print("out: ", out)
    assert len(out) == 2
    assert out[0] is None
    assert out[1] == {'a': 1}


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
            outputs = dict(conv1=conv1_x, conv2=conv2_x, fc=fc_x)  # @jit.typing: () -> tensor_type[float32]
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
            outputs = dict(conv1=conv1_x, conv2=conv2_x, fc=fc_x)  # @jit.typing: () -> tensor_type[float32]
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
def test_multiple_return_nested_dict1():
    """
    Feature: Return nested output of dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net1():
        return {'a': None, 'b': {'a': 1}}

    @ms.jit
    def dict_net2():
        return {'a': None, 'b': {'a': {'c': 1}}}

    out = dict_net1()
    assert out == {'a': None, 'b': {'a': 1}}
    out = dict_net2()
    assert out == {'a': None, 'b': {'a': {'c': 1}}}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multiple_return_nested_dict2():
    """
    Feature: Return nested output of dict.
    Description: Support dict return.
    Expectation: No exception.
    """
    @ms.jit
    def dict_net1():
        return {'a': None, 'b': [1, 2, None]}

    @ms.jit
    def dict_net2():
        return {'a': None, 'b': [1, 2, {'a': 1}]}

    out = dict_net1()
    assert out == {'a': None, 'b': [1, 2, None]}
    out = dict_net2()
    assert out == {'a': None, 'b': [1, 2, {'a': 1}]}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multiple_return_nested_dict3():
    """
    Feature: Return nested output of dict.
    Description: Support dict return.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net1():
        return {'a': None, 'b': (1, 2, None)}

    @ms.jit
    def dict_net2():
        return {'a': None, 'b': (1, 2, {'a': 1})}

    out = dict_net1()
    assert out == {'a': None, 'b': (1, 2, None)}
    out = dict_net2()
    assert out == {'a': None, 'b': (1, 2, {'a': 1})}
