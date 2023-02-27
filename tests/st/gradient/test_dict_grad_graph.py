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
"""test grad for dict in graph mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops as ops
from mindspore.ops.composite import GradOperation
from mindspore import Tensor
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_backward_return_dict():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network with tensor input and backward returns dict.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            result = self.relu(x)
            return result

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return {'a': out}

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert isinstance(output, dict)
    assert len(output.keys()) == 1
    assert np.allclose(output['a'].asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_contain_make_dict_and_dict_getitem():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network which contains make_dict and dict_getitem.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = {'a': x}
            z = y['a']
            result = self.relu(z)
            return result

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return out

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_return_dict():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network which returns make_dict.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            result = {'a': y}
            return result

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return out

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_return_dict_backward_return_dict():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network which returns make_dict, and the backward returns make_dict.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            result = {'a': y}
            return result

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return {'a': out}

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert isinstance(output, dict)
    assert len(output.keys()) == 1
    assert np.allclose(output['a'].asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_contain_make_dict_and_dict_getitem_backward_return_dict():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network which contains make_dict and dict_getitem,
                 and the backward returns make_dict.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            result = {'a': y}
            return result['a']

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return {'a': out}

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert isinstance(output, dict)
    assert len(output.keys()) == 1
    assert np.allclose(output['a'].asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_contain_make_dict_and_dict_setitem_backward_return_dict1():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network which contains make_dict and dict_setitem,
                 and the backward returns make_dict.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            result = {'a': y}
            result['a'] = x
            return result

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return {'a': out}

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert isinstance(output, dict)
    assert len(output.keys()) == 1
    assert np.allclose(output['a'].asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_contain_make_dict_and_dict_setitem_backward_return_dict2():
    """
    Feature: Grad for dict.
    Description: Get gradient for the network which contains make_dict and dict_setitem,
                 and the backward returns make_dict.
    Expectation: Get the correct output.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            y = self.relu(x)
            result = {'a': y}
            result['a'] = x
            return result['a']

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x):
            grad_func = self.grad_op(self.net)
            out = grad_func(x)
            return {'a': out}

    net = Net()
    grad_net = GradNetWrtX(net)
    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    output = grad_net(x)
    expect = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32)
    assert isinstance(output, dict)
    assert len(output.keys()) == 1
    assert np.allclose(output['a'].asnumpy(), expect)
