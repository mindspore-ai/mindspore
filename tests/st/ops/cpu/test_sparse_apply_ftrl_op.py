# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sparse_apply_ftrl = P.FusedSparseFtrl(lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5)
        self.var = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="accum")
        self.linear = Parameter(Tensor(np.ones([3, 3, 3]).astype(np.float32)), name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


class TestNet(nn.Cell):
    def __init__(self, var_np, accum_np, linear_np, lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5):
        super(TestNet, self).__init__()
        self.sparse_apply_ftrl = P.FusedSparseFtrl(lr=lr, l1=l1, l2=l2, lr_power=lr_power)
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")
        self.linear = Parameter(Tensor(linear_np), name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


class SparseApplyFtrlNet(nn.Cell):
    def __init__(self, var, accum, linear, lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5):
        super(SparseApplyFtrlNet, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=lr, l1=l1, l2=l2, lr_power=lr_power)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    """
    Feature: FusedSparseFtrl
    Description: normal params, attr and input
    Expectation: the result meet expectation
    """
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    sparse_apply_ftrl = Net()
    sparse_apply_ftrl(gradient, indices)
    print(sparse_apply_ftrl.var.data)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float32)
    assert np.all(sparse_apply_ftrl.var.data.asnumpy() == expect_var)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_shape_var_accum_not_match():
    """
    Feature: FusedSparseFtrl
    Description: var and accum shape not same
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 2, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseFtrl], the var shape must be equal to accum shape
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_shape_var_linear_not_match():
    """
    Feature: FusedSparseFtrl
    Description: var and linear shape not same
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 2, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseFtrl], the var shape must be equal to linear shape
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_shape_grad_indices_not_match():
    """
    Feature: FusedSparseFtrl
    Description: when grad_shape[0] != indices_shape[0]
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseFtrl], the grad_shape[0] must be equal to indices_shape[0]
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_shape_indices_rank_invalid():
    """
    Feature: FusedSparseFtrl
    Description: indices rank size != 1
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor(0, mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseFtrl], the indices rank must be equal to 1" in str(e)
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_shape_grad_rank_invalid():
    """
    Feature: FusedSparseFtrl
    Description: grad rank size <= 0
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(3, mstype.float32)
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseFtrl], the grad rank must be greater than or equal to 1
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_shape_indices_grad_not_match():
    """
    Feature: FusedSparseFtrl
    Description: when grad_shape[1:] != var_shape[1:]
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # not match when grad_shape[1:] != var_shape[1:]
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(np.ones([3, 2, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, indices)
        assert False
    except ValueError:
        # ValueError: For primitive[FusedSparseFtrl], the var_shape[1:] must be equal to grad_shape[1:]
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_type_indices_invalid():
    """
    Feature: FusedSparseFtrl
    Description: invalid indices data type
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)

    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.float32)
    try:
        net(gradient, indices)
        assert False
    except TypeError:
        # TypeError: For primitive[FusedSparseFtrl], the input argument[indices] must be a type of
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_type_indices_invalid2():
    """
    Feature: FusedSparseFtrl
    Description: list tensor, invalid indices data type
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net(gradient, [indices, indices])
        assert False
    except TypeError:
        # TypeError: For Primitive[FusedSparseFtrl], the input argument[indices] must be a Tensor
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_invalid_input_type_gradient_invalid():
    """
    Feature: FusedSparseFtrl
    Description: list tensor, invalid gradient data type
    Expectation: raise expected exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var_np = np.ones([3, 3, 3]).astype(np.float32)
    accum_np = np.ones([3, 3, 3]).astype(np.float32)
    linear_np = np.ones([3, 3, 3]).astype(np.float32)
    net = TestNet(var_np, accum_np, linear_np)
    gradient = Tensor(np.ones([3, 3, 3]).astype(np.float32))
    indices = Tensor([0, 1, 2], mstype.int32)
    try:
        net([gradient, gradient], indices)
        assert False
    except TypeError:
        # TypeError: The primitive[FusedSparseFtrl]'s input arguments[accum, grad, linear, var] must be all tensor
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_sparse_ftrl_dynamic():
    """
    Feature: FusedSparseFtrl
    Description: dynamic inputs
    Expectation: the result meets the expectation
    """

    class DynamicNet(nn.Cell):
        def __init__(self):
            super(DynamicNet, self).__init__()
            self.unique = P.Unique()
            self.gather = P.Gather()
            self.axis = 0

            self.sparse_apply_ftrl = P.FusedSparseFtrl(lr=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
            self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
            self.accum = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="accum")
            self.linear = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="linear")

        def construct(self, grad, indices, indices_dy):
            indices_dy, _ = self.unique(indices_dy)
            grad = self.gather(grad, indices_dy, self.axis)
            indices = self.gather(indices, indices_dy, self.axis)
            out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    grad = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]], [[0.1, 0.1]]]).astype(np.float32))
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))
    indices_dy = Tensor([0, 1], mstype.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = DynamicNet()
    net(grad, indices, indices_dy)
    print(net.var.data)
    expect_var = np.array([[[-0.00598256, -0.00598256]],
                           [[-0.00598256, -0.00598256]],
                           [[1., 1.]]]).astype(np.float32)
    assert np.allclose(net.var.data.asnumpy(), expect_var)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_apply_ftrl():
    """
    Feature: SparseApplyFtrl
    Description: normal params, attr and input
    Expectation: the result meet expectation
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    grad_np = np.ones([3, 3, 3])
    indice_np = [0, 1, 2]
    var_np = np.ones([3, 3, 3])
    accum_np = np.ones([3, 3, 3])
    linear_np = np.ones([3, 3, 3])

    # test1: var/accum/linear/gradient are float32 and indices is int32.
    gradient = Tensor(grad_np, dtype=mstype.float32)
    indices = Tensor(indice_np, dtype=mstype.int32)
    var = Tensor(var_np, dtype=mstype.float32)
    accum = Tensor(accum_np, dtype=mstype.float32)
    linear = Tensor(linear_np, dtype=mstype.float32)
    sparse_apply_ftrl = SparseApplyFtrlNet(var, accum, linear)
    out = sparse_apply_ftrl(gradient, indices)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float32)
    assert np.all(out[0].asnumpy() == expect_var)

    # test2: var/accum/linear/gradient are float16 and indices is int32.
    gradient = Tensor(grad_np, dtype=mstype.float16)
    indices = Tensor(indice_np, dtype=mstype.int32)
    var = Tensor(var_np, dtype=mstype.float16)
    accum = Tensor(accum_np, dtype=mstype.float16)
    linear = Tensor(linear_np, dtype=mstype.float16)
    sparse_apply_ftrl = SparseApplyFtrlNet(var, accum, linear)
    out = sparse_apply_ftrl(gradient, indices)
    expect_var = np.array([[[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]],
                           [[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]],
                           [[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]]]).astype(np.float16)
    assert np.all(out[0].asnumpy() == expect_var)


class VmapNetSparseApplyFtrl(nn.Cell):
    def __init__(self, lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5):
        super(VmapNetSparseApplyFtrl, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=lr, l1=l1, l2=l2, lr_power=lr_power)

    def construct(self, var, accum, linear, grad, indices):
        out = self.sparse_apply_ftrl(var, accum, linear, grad, indices)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_sparseapplyftrl():
    """
    Feature: Vmap feature on SparseApplyFtrl cpu op
    Description: Compare the vmap result with the manually batch result.
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var = Parameter(Tensor(np.array([[2.0], [0.1]]).astype(np.float32)))
    accum = Parameter(Tensor(np.array([[1.0], [0.1]]).astype(np.float32)))
    linear = Parameter(Tensor(np.array([[0.2], [0.1]]).astype(np.float32)))
    gradient = Tensor(np.array([[0.5], [0.1]]).astype(np.float32))
    indices = Tensor([[0], [0]], mstype.int32)
    sparse_apply_adagrad = VmapNetSparseApplyFtrl()
    return_vmap = F.vmap(sparse_apply_adagrad, in_axes=(0, 0, 0, 0, 0))(var, accum, linear, gradient, indices)

    expect_var = np.array([[2.10519552e-01], [4.05071862e-03]]).astype(np.float32)
    expect_accum = np.array([[1.25000000e+00], [1.09999999e-01]]).astype(np.float32)
    expect_linear = np.array([[-2.35367996e+02], [-1.34347117e+00]]).astype(np.float32)
    assert len(return_vmap) == 3
    assert np.all(return_vmap[0].asnumpy() == expect_var)
    assert np.all(return_vmap[1].asnumpy() == expect_accum)
    assert np.all(return_vmap[2].asnumpy() == expect_linear)
