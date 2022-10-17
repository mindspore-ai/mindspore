# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class TestNet(nn.Cell):
    def __init__(self, var_np, m_np, v_np):
        super(TestNet, self).__init__()
        self.net = P.FusedSparseLazyAdam()
        self.var = Parameter(var_np, name="var")
        self.m = Parameter(m_np, name="m")
        self.v = Parameter(v_np, name="v")
        self.beta1_power = 0.9
        self.beta2_power = 0.999
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def construct(self, grad, indices):
        out = self.net(self.var, self.m, self.v, self.beta1_power, self.beta2_power, self.lr, self.beta1,
                       self.beta2, self.epsilon, grad, indices)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_shape_var_m_not_match():
    """
    Feature: FusedSparseLazyAdam
    Description: var and m shape not same
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 2, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        #  ValueError: For primitive[FusedSparseLazyAdam], the var_shape must be equal to m_shape
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_shape_var_v_not_match():
    """
    Feature: FusedSparseLazyAdam
    Description: var and v shape not same
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 2, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseLazyAdam], the var_shape must be equal to v_shape
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_shape_grad_indices_not_match():
    """
    Feature: FusedSparseLazyAdam
    Description: when grad_shape[0] != indices_shape[0]
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseLazyAdam], the grad_shape[0] must be equal to indices_shape[0]
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_shape_indices_rank_invalid():
    """
    Feature: FusedSparseLazyAdam
    Description: indices rank size != 1
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor(0, mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseLazyAdam], the indices rank must be equal to 1
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_shape_grad_rank_invalid():
    """
    Feature: FusedSparseLazyAdam
    Description: grad rank size <= 0
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(3, mstype.float32)
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseLazyAdam], the grad rank must be greater than or equal to 1
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_shape_indices_grad_not_match():
    """
    Feature: FusedSparseLazyAdam
    Description: when grad_shape[1:] != var_shape[1:]
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(np.ones([3, 2, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For 'FusedSparseLazyAdam', the shape of updates must be [] or grad_shape = indices_shape +
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_type_indices_invalid():
    """
    Feature: FusedSparseLazyAdam
    Description: invalid indices data type
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.float32)
    try:
        net(gradient, indices)
        assert False
    except TypeError:
        # TypeError: For primitive[FusedSparseLazyAdam], the input argument[indices] must be a type of
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_type_indices_invalid2():
    """
    Feature: FusedSparseLazyAdam
    Description: list tensor, invalid indices data type
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, [indices, indices])
        assert False
    except TypeError:
        # TypeError: For Primitive[FusedSparseLazyAdam], the input argument[indices] must be a Tensor
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_invalid_input_type_gradient_invalid():
    """
    Feature: FusedSparseLazyAdam
    Description: list tensor, invalid gradient data type
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    m_np = np.ones([3, 3, 3]).astype(np.float32)
    v_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, m_np, v_np)
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net([gradient, gradient], indices)
        assert False
    except TypeError:
        # TypeError: The primitive[FusedSparseLazyAdam]'s input arguments[grad, m, v, var] must be all tensor
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_lazy_adam_dynamic():
    """
    Feature: FusedSparseLazyAdam
    Description: dynamic inputs
    Expectation: the result meets the expectation
    """

    class DynamicNet(nn.Cell):
        def __init__(self):
            super(DynamicNet, self).__init__()
            self.unique = P.Unique()
            self.gather = P.Gather()
            self.axis = 0

            self.net = P.FusedSparseLazyAdam()
            self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
            self.m = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="m")
            self.v = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="v")

            self.beta1_power = 0.9
            self.beta2_power = 0.999
            self.lr = 0.001
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

        def construct(self, grad, indices, indices_dy):
            indices_dy, _ = self.unique(indices_dy)
            grad = self.gather(grad, indices_dy, self.axis)
            indices = self.gather(indices, indices_dy, self.axis)
            out = self.net(self.var, self.m, self.v, self.beta1_power, self.beta2_power, self.lr, self.beta1,
                           self.beta2, self.epsilon, grad, indices)
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = DynamicNet()
    gradient = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]], [[0.1, 0.1]]]), mstype.float32)
    indices = Tensor([0, 1, 2], mstype.int32)

    indices_dy = Tensor([0, 1], mstype.int32)
    net(gradient, indices, indices_dy)
    print(net.var.data)
    expect_var = np.array([[[0.9997121, 0.9997121]],
                           [[0.9997121, 0.9997121]],
                           [[1, 1]]]).astype(np.float32)
    assert np.allclose(net.var.data.asnumpy(), expect_var)
