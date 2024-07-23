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
from mindspore import ops, Parameter, Tensor, jit
import mindspore.common.dtype as mstype
from mindspore.nn import Cell
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
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


@pytest.mark.skip(reason="Pyexecute output is not any and type is wrong.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
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


@pytest.mark.skip(reason="Pyexecute output is not any and type is wrong.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict1():
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict2():
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict3():
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict_with_inputs():
    """
    Feature: Return nested output of dict.
    Description: Support dict return.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, y):
        return {'a': [x, y], 'b': (1, 2, {'a': 1})}

    x = Tensor([1], dtype=mstype.int64)
    y = Tensor([2], dtype=mstype.int64)
    out = dict_net(x, y)
    assert out == {'a': [x, y], 'b': (1, 2, {'a': 1})}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_nested_dict_with_parameter():
    """
    Feature: Support dict.
    Description: Support nested list and dict with parameter.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")

        def construct(self):
            z = [{'params': [self.x, self.y], 'a': 1, 'b': False}, {'params': self.x, 'a': 2}]
            out1 = z[0]['params']
            out2 = z[1]['a']
            return out1, out2

    net = Net()
    out1, out2 = net()
    assert out1 == [net.x, net.y]
    assert out2 == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict_with_parameter1():
    """
    Feature: Return nested output of dict with parameter.
    Description: Support dict return.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")

        def construct(self):
            z = ({'params': (self.x, self.y), 'a': 1, 'b': False}, {'params': self.x, 'a': 2})
            return z

    net = Net()
    out = net()
    assert out == ({'params': (net.x, net.y), 'a': 1, 'b': False}, {'params': net.x, 'a': 2})


@pytest.mark.skip('Not support list to PyExecute yet.')
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict_with_parameter2():
    """
    Feature: Return nested output of dict with parameter.
    Description: Support dict return.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")

        def construct(self):
            z = [{'params': [self.x, self.y], 'a': 1, 'b': False}, {'params': self.x, 'a': 2}]
            return z

    net = Net()
    out = net()
    assert out == [{'params': [net.x, net.y], 'a': 1, 'b': False}, {'params': net.x, 'a': 2}]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_nested_dict_with_parameter_constant1():
    """
    Feature: Support dict.
    Description: Support nested list and dict with parameter constant.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")
            self.z = [{'params': [self.x, self.y], 'a': 1, 'b': False}, {'params': self.x, 'a': 2}]

        def construct(self):
            out1 = self.z[0]['params']
            out2 = self.z[1]['a']
            return out1, out2

    net = Net()
    out1, out2 = net()
    assert out1 == [net.x, net.y]
    assert out2 == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_nested_dict_with_parameter_constant2():
    """
    Feature: Support dict.
    Description: Support nested list and dict with parameter constant.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")
            self.z = {'params': [self.x, self.y], 'a': 1, 'b': {'params': self.x, 'a': 2}}

        def construct(self):
            out1 = self.z['params']
            out2 = self.z['b']['params']
            return out1, out2

    net = Net()
    out1, out2 = net()
    assert out1 == [net.x, net.y]
    assert out2 == net.x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict_with_parameter_constant1():
    """
    Feature: Return nested output of dict with parameter constant.
    Description: Support dict return.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")
            self.z = [{'params': (self.x, self.y), 'a': 1, 'b': False}, {'params': self.x, 'a': 2}]

        def construct(self):
            return self.z

    net = Net()
    out = net()
    assert out == [{'params': (net.x, net.y), 'a': 1, 'b': False}, {'params': net.x, 'a': 2}]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_return_nested_dict_with_parameter_constant2():
    """
    Feature: Return nested output of dict with parameter constant.
    Description: Support dict return.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")
            self.z = [{'params': [self.x, self.y], 'a': 1, 'b': False}, {'params': self.x, 'a': 2}]

        def construct(self):
            return self.z

    net = Net()
    out = net()
    assert out == [{'params': [net.x, net.y], 'a': 1, 'b': False}, {'params': net.x, 'a': 2}]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict_with_parameter_constant3():
    """
    Feature: Return nested output of dict with parameter constant.
    Description: Support dict return.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")
            self.z = {'params': (self.x, self.y), 'a': 1, 'b': {'params': self.x, 'a': 2}}

        def construct(self):
            return self.z

    net = Net()
    out = net()
    assert out == {'params': (net.x, net.y), 'a': 1, 'b': {'params': net.x, 'a': 2}}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_nested_dict_with_parameter_constant4():
    """
    Feature: Return nested output of dict with parameter constant.
    Description: Support dict return.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Parameter(Tensor([1], dtype=mstype.int64), name="input_x")
            self.y = Parameter(Tensor([2], dtype=mstype.int64), name="input_y")
            self.z = {'params': [self.x, self.y], 'a': 1, 'b': {'params': self.x, 'a': 2}}

        def construct(self):
            return self.z

    net = Net()
    out = net()
    assert out == {'params': [net.x, net.y], 'a': 1, 'b': {'params': net.x, 'a': 2}}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_with_dict_values():
    """
    Feature: Return dict when using dict.values.
    Description: Support dict return.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x):
        d = {'a': x, 'b': 2}
        v = d.values()
        z = v[0]
        return {'x': z}

    x = Tensor([1], dtype=mstype.int64)
    out = dict_net(x)
    assert out['x'] == x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_with_dict_items():
    """
    Feature: Return dict when using dict.items.
    Description: Support dict return.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, y):
        z = {'x': x, 'y': y}
        t = ()
        for _, v in z.items():
            t += (v,)
        out = {'a': t}
        return out

    x = Tensor([1], dtype=mstype.int64)
    y = Tensor([2], dtype=mstype.int64)
    out = dict_net(x, y)
    assert out == {'a': (x, y)}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_with_empty_shape_tensor():
    """
    Feature: Return dict which contains tensor with empty shape.
    Description: Support dict return.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net():
        return {'a': Tensor(2)}

    out = dict_net()
    assert out['a'].shape == ()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_in_if_else():
    """
    Feature: Support dict return.
    Description: The if and else branch return dict.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, key):
        if key > 0:
            y = {"abc": x, "number": [1, 2, 3]}
        else:
            y = {"cba": x, "number": [3, 2, 1]}
        return y

    x = [{1: [Tensor([1]), {0: Tensor([1, 2, 3])}]}, {2: Tensor([2])}]
    out = dict_net(x, Tensor(-1))
    assert out == {"cba": x, "number": [3, 2, 1]}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_different_size_dict_in_if_else():
    """
    Feature: Support dict return.
    Description: The if and else branch return dict of different size.
    Expectation: Throw an exception.
    """

    @ms.jit
    def dict_net(x, key):
        if key < 0:
            y = {Tensor([True]): x, "number": [1, 2, 3]}
        else:
            y = {}
        return y

    x = [{1: [Tensor([1]), {0: Tensor([1, 2, 3])}]}, {2: Tensor([2])}]
    try:
        dict_net(x, Tensor(-1))
    except TypeError as e:
        assert "Cannot join the return values of different branches, perhaps you need to make them equal." in str(e)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_get_item_with_string_input():
    """
    Feature: Support string as the input of top network.
    Description: Get the element with the string input.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, input_str):
        a = {'a': x}
        return a[input_str]

    x = Tensor(2)
    out = dict_net(x, "a")
    assert out == x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_get_item_with_string_input_grad():
    """
    Feature: Support string as the input of top network.
    Description: Get the gradient for getting the element with the string input.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, input_str):
        a = {'a': x}
        return a[input_str]

    x = Tensor(2)
    out = ops.grad(dict_net, grad_position=(0, 1))(x, "a")
    assert out == Tensor(1)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_with_string_input():
    """
    Feature: Support string as the input of top network.
    Description: Return dict with the string input.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, input_str):
        return {input_str: x}

    x = Tensor(2)
    out = dict_net(x, "a")
    assert out == {"a": x}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_with_string_input_grad():
    """
    Feature: Support string as the input of top network.
    Description: Get the gradient for returning dict with the string input..
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(x, input_str):
        return {input_str: x}

    x = Tensor(2)
    out = ops.grad(dict_net, grad_position=(0, 1))(x, "a")
    assert out == Tensor(1)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_dict_with_different_size_branch():
    """
    Feature: Support dict return.
    Description: The if and else branch return dict of different size.
    Expectation: Return the correct dict.
    """
    class InnerNet(Cell):
        def construct(self, x, y):
            return [x, y]

    class DictNet(Cell):
        def __init__(self):
            super().__init__()
            self.obj = InnerNet()

        def construct(self, z):
            x = [1, 2, 3]
            y = [4, 5, 6]
            if z >= 0:
                ret = {k: v for k, v in zip(x, y)}
            else:
                d = [[i, sum(self.obj(x, y)[i])] for i in range(2)]
                ret = {k: v for k, v in d}
            return ret

    ms_net = DictNet()
    z = Tensor(0)
    ms_out = ms_net(z)
    assert ms_out == {1: 4, 2: 5, 3: 6}

    z = Tensor(-1)
    ms_out = ms_net(z)
    assert ms_out == {0: 6, 1: 15}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_dict_inner_method_overrrided_1():
    """
    Feature: Support overriding dict getitem.
    Description: Make overriding __getitem__ works in graph mode
    Expectation: Return the correct value.
    """
    class Tmp(dict):
        def __getitem__(self, x):
            return x

    obj = Tmp({"aaa": 100})
    @jit
    def foo():
        return obj["aaa"]
    ms_out = foo()
    assert ms_out == 'aaa'


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inner_method_overrrided_2():
    """
    Feature: Support overriding dict getattr.
    Description: Make overriding __getattr__ works in graph mode
    Expectation: Return the correct value.
    """
    class Tmp(dict):
        __getattr__ = dict.__getitem__

    obj = Tmp({"aaa": 100})

    @jit
    def foo():
        return obj.aaa
    ms_out = foo()
    assert ms_out == 100


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inner_method_overrrided_3():
    """
    Feature: Support getattr from overridden dict.
    Description: Support getattr from overridden dict in graph mode
    Expectation: Return the correct value.
    """
    class Tmp(dict):
        def __getitem__(self, k):
            inner_dict = dict(self.items())
            return inner_dict[k]

        def to_tuple(self):
            return tuple(self[k] for k in self.keys())

    obj = Tmp({"a": 1, "b": 2})

    @jit
    def foo():
        return obj.to_tuple()
    ms_out = foo()
    assert ms_out == (1, 2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_pynative_jit_dict_grad():
    """
    Feature: Return dict in forward graph.
    Description: Support grad for dict return in pynative mode.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(a):
        x = {'a': a, 'b': 2}
        return x

    ms.set_context(mode=ms.PYNATIVE_MODE)
    out = ops.grad(dict_net)(ms.Tensor([1]))
    assert out == 1
    ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_pynative_jit_dict_grad_2():
    """
    Feature: Return dict in forward graph.
    Description: Support grad for dict return in pynative mode.
    Expectation: No exception.
    """

    @ms.jit
    def dict_net(a):
        x = {'a': a, 'b': 2}
        return x

    ms.set_context(mode=ms.PYNATIVE_MODE)
    grad = ops.GradOperation()
    out = grad(dict_net)(ms.Tensor([1]))
    assert out == 1
    ms.set_context(mode=ms.GRAPH_MODE)


def test_jitclass_grad():
    """
    Feature: Support grad with custom class in jit in pynative mode.
    Description: Support grad with custom class in jit in pynative mode.
    Expectation: No exception.
    """
    class GradNet(Cell):
        def __init__(self, net, grad_position=0):
            super().__init__()
            self.grad = ops.grad
            self.grad_net = self.grad(net, grad_position=grad_position)

        def construct(self, *x):
            return self.grad_net(*x)


    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

    obj = Net()

    class ModNet(Cell):
        def construct(self, y):
            self._mod_x()
            return ops.mul(obj.x, y)

        @jit
        def _mod_x(self):
            obj.x = -1*obj.x


    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms_net = ModNet()
    x_before = obj.x
    y = Tensor(16)
    ms_out = ms_net(y)
    ms_out2 = ms_net(y)
    ms_grad = GradNet(ms_net)(y)
    ms.set_context(mode=ms.GRAPH_MODE)

    assert ms_out == -16
    assert ms_out2 == 16
    assert x_before == 1
    assert ms_grad == -1
