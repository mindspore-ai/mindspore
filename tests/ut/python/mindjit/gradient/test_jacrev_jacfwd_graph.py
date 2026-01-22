# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test function jacfwd and jacrev in graph mode"""
import numpy as np
import pytest
from mindspore import Tensor, ops, nn, context
from mindspore.ops.functional import jacfwd, jacrev
from mindspore.graph.api import _pynative_executor

context.set_context(mode=context.GRAPH_MODE)

class NetMul(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        out = self.mul(x, y)
        return out

class JacFwdNet(nn.Cell):
    def __init__(self, net, grad_position=0, has_aux=False):
        super().__init__()
        self.net = net
        self.grad_position = grad_position
        self.has_aux = has_aux

    def construct(self, *x):
        jac_net = jacfwd(self.net, self.grad_position, self.has_aux)
        return jac_net(*x)


class JacRevNet(nn.Cell):
    def __init__(self, net, grad_position=0, has_aux=False):
        super().__init__()
        self.net = net
        self.grad_position = grad_position
        self.has_aux = has_aux

    def construct(self, *x):
        jac_net = jacrev(self.net, self.grad_position, self.has_aux)
        return jac_net(*x)


def test_jacfwd_position_duplicate():
    """
    Features: Function jacfwd.
    Description: Test jacfwd invalid position
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacFwdNet(net, grad_position=(0, 0, 1))
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    with pytest.raises(ValueError) as e:
        grad_func(x, y)
        _pynative_executor.sync()
    assert "duplicate" in str(e.value)


@pytest.mark.parametrize('grad_position_error', [(-1, ValueError), ((0, -1), ValueError),
                                                 (2, ValueError), ((0, 2), ValueError),
                                                 (1.0, TypeError), ((0, 1.0), TypeError),
                                                 ([0, 1], TypeError), (("has", "aux"), TypeError)])
def test_jacfwd_position_neg_int(grad_position_error):
    """
    Features: Function jacfwd.
    Description: Test jacfwd invalid grad position
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacFwdNet(net, grad_position=grad_position_error[0])
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    error_type = grad_position_error[1]
    with pytest.raises(error_type):
        grad_func(x, y)
        _pynative_executor.sync()


def test_jacfwd_has_aux_true_out1():
    """
    Features: Function jacfwd.
    Description: Test jacfwd grad invalid has_aux
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacFwdNet(net, grad_position=(0, 1), has_aux=True)
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    with pytest.raises(ValueError) as e:
        grad_func(x, y)
        _pynative_executor.sync()
    assert "When 'has_aux' is True, origin 'fn' requires more than one outputs" in str(e.value)


def test_jacfwd_has_aux_int():
    """
    Features: Function jacfwd.
    Description: Test jacfwd grad invalid has_aux
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacFwdNet(net, grad_position=(0, 1), has_aux=2)
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    with pytest.raises(TypeError) as e:
        grad_func(x, y)
        _pynative_executor.sync()
    assert "The 'has_aux' must be bool type" in str(e.value)


def test_jacrev_position_duplicate():
    """
    Features: Function jacrev.
    Description: Test jacrev invalid position
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacRevNet(net, grad_position=(0, 0, 1))
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    with pytest.raises(ValueError) as e:
        grad_func(x, y)
        _pynative_executor.sync()
    assert "duplicate" in str(e.value)


@pytest.mark.parametrize('grad_position_error', [(-1, ValueError), ((0, -1), ValueError),
                                                 (2, ValueError), ((0, 2), ValueError),
                                                 (1.0, TypeError), ((0, 1.0), TypeError),
                                                 ([0, 1], TypeError), (("has", "aux"), TypeError)])
def test_jacrev_position_neg_int(grad_position_error):
    """
    Features: Function jacrev.
    Description: Test jacrev invalid grad position
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacRevNet(net, grad_position=grad_position_error[0])
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    error_type = grad_position_error[1]
    with pytest.raises(error_type):
        grad_func(x, y)
        _pynative_executor.sync()


def test_jacrev_has_aux_true_out1():
    """
    Features: Function jacrev.
    Description: Test jacrev grad invalid has_aux
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacRevNet(net, grad_position=(0, 1), has_aux=True)
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    with pytest.raises(ValueError) as e:
        grad_func(x, y)
        _pynative_executor.sync()
    assert "When 'has_aux' is True, origin 'fn' requires more than one outputs" in str(e.value)


def test_jacrev_has_aux_int():
    """
    Features: Function jacrev.
    Description: Test jacrev grad invalid has_aux
    Expectation: Raise exception.
    """
    net = NetMul()
    grad_func = JacRevNet(net, grad_position=(0, 1), has_aux=2)
    x = Tensor(np.random.randn(5).astype(np.float32))
    y = Tensor(np.random.randn(5).astype(np.float32))
    with pytest.raises(TypeError) as e:
        grad_func(x, y)
        _pynative_executor.sync()
    assert "The 'has_aux' must be bool type" in str(e.value)
