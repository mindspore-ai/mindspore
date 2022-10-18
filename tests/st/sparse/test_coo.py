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
"""smoke tests for COO operations"""

import pytest
import numpy as np

from mindspore import Tensor, COOTensor, jit, nn, ops
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F

from .sparse_utils import get_platform, compare_res, compare_coo


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_make_coo():
    """
    Feature: Test COOTensor Constructor in Graph and PyNative.
    Description: Test COOTensor(indices, values, shape) and COOTensor(COOTensor)
    Expectation: Success.
    """
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    dense_shape = (3, 4)

    def test_pynative():
        return COOTensor(indices, values, dense_shape)
    test_graph = jit(test_pynative)

    coo1 = test_pynative()
    coo2 = test_graph()
    compare_coo(coo1, coo2)
    coo3 = COOTensor(coo_tensor=coo2)
    compare_coo(coo3, coo2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coo_tensor_with_control_if():
    """
    Feature: Test COOTensor in if.
    Description: Test COOTensor computation in while loop.
    Expectation: Success.
    """
    class COOTensorValuesDouble(nn.Cell):

        def construct(self, x):
            indices = x.indices
            values = x.values * 2
            shape = x.shape
            return COOTensor(indices, values, shape)

    class COOTensorValuesAdd2(nn.Cell):

        def construct(self, x):
            indices = x.indices
            values = x.values + 2
            shape = x.shape
            return COOTensor(indices, values, shape)

    class COOTensorWithControlIf(nn.Cell):
        def __init__(self, shape):
            super(COOTensorWithControlIf, self).__init__()
            self.op1 = COOTensorValuesDouble()
            self.op2 = COOTensorValuesAdd2()
            self.shape = shape

        def construct(self, a, b, indices, values):
            x = COOTensor(indices, values, self.shape)
            if a > b:
                x = self.op1(x)
            else:
                x = self.op2(x)
            return x.indices, x.values, x.shape

    a = Tensor(0, mstype.int32)
    b = Tensor(2, mstype.int32)
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (3, 4)
    net = COOTensorWithControlIf(shape)
    out = net(a, b, indices, values)
    assert np.allclose(out[0].asnumpy(), indices.asnumpy(), .0, .0)
    assert np.allclose(out[1].asnumpy(), values.asnumpy() + 2, .0, .0)
    assert out[2] == shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coo_tensor_in_while():
    """
    Feature: Test COOTensor in while loop.
    Description: Test COOTensor computation in while loop.
    Expectation: Success.
    """
    class COOTensorWithControlWhile(nn.Cell):
        def __init__(self, shape):
            super(COOTensorWithControlWhile, self).__init__()
            self.shape = shape

        @jit
        def construct(self, a, b, indices, values):
            x = COOTensor(indices, values, self.shape)
            while a > b:
                x = COOTensor(indices, values, self.shape)
                b = b + 1
            return x
    a = Tensor(3, mstype.int32)
    b = Tensor(0, mstype.int32)
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (3, 4)
    net = COOTensorWithControlWhile(shape)
    out = net(a, b, indices, values)
    assert np.allclose(out.indices.asnumpy(), indices.asnumpy(), .0, .0)
    assert np.allclose(out.values.asnumpy(), values.asnumpy(), .0, .0)
    assert out.shape == shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_coo_method():
    """
    Feature: Test coo tensor methods.
    Description: Test coo_tensor.to_csr(), coo_tensor.to_dense().
    Expectation: Success.
    """
    if get_platform() != "linux":
        return

    class COOToCSRNet(nn.Cell):
        def construct(self, coo_tensor):
            return coo_tensor.to_csr()

    class COOToDenseNet(nn.Cell):
        def construct(self, coo_tensor):
            return coo_tensor.to_dense()

    indices = Tensor([[1, 2], [0, 1]], dtype=mstype.int32)
    values = Tensor([2, 1], dtype=mstype.float32)
    shape = (3, 4)
    coo_tensor = COOTensor(indices, values, shape)

    to_csr_output = COOToCSRNet()(coo_tensor)
    to_csr_expect_1 = np.array([0, 1, 2, 2], dtype=np.int32)
    to_csr_expect_2 = np.array([1, 2], dtype=np.int32)
    to_csr_expect_3 = np.array([1, 2], dtype=np.float32)
    assert np.allclose(to_csr_output.indptr.asnumpy(), to_csr_expect_1)
    assert np.allclose(to_csr_output.indices.asnumpy(), to_csr_expect_2)
    assert np.allclose(to_csr_output.values.asnumpy(), to_csr_expect_3)

    to_dense_output = COOToDenseNet()(coo_tensor)
    to_dense_expect = np.array(
        [[0., 1., 0., 0.], [0., 0., 2., 0.], [0., 0., 0., 0.]], dtype=np.float32)
    assert np.allclose(to_dense_output.asnumpy(), to_dense_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coo_coalesce():
    """
    Feature: Test coo tensor coalesce methods.
    Description: Test coo_tensor.coalesce()
    Expectation: Success.
    """
    if get_platform() != "linux":
        return

    class COOCoalesce(nn.Cell):
        def construct(self, coo_tensor):
            return coo_tensor.coalesce()

    indices = Tensor([[0, 0, 1], [1, 1, 2]], dtype=mstype.int64)
    values = Tensor([1, 5, 4], dtype=mstype.float32)
    shape = (3, 3)
    coo_tensor = COOTensor(indices.transpose(), values, shape)

    coalesce_output = COOCoalesce()(coo_tensor)
    res_coo = coo_tensor.coalesce()

    expect_indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    expect_values = np.array([6., 4.], dtype=np.float32)
    assert np.allclose(expect_indices, res_coo.indices.asnumpy())
    assert np.allclose(expect_values, res_coo.values.asnumpy())
    assert np.allclose(expect_indices, coalesce_output.indices.asnumpy())
    assert np.allclose(expect_values, coalesce_output.values.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dtype_coo_tensor():
    """
    Feature: Test F.dtype with COOTensor.
    Description: Test: F.dtype(x), x.dtype.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (3, 4)

    def pynative_test():
        x = COOTensor(indices, values, shape)
        return F.dtype(x), x.dtype
    graph_test = jit(pynative_test)

    out1, out2 = pynative_test()
    out3, out4 = graph_test()
    assert out1 in [mstype.float32]
    assert out2 in [mstype.float32]
    assert out3 in [mstype.float32]
    assert out4 in [mstype.float32]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_coo_attr():
    """
    Feature: Test COOTensor GetAttr in Graph and PyNative.
    Description: Test COOTensor.indices, COOTensor.values, COOTensor.shape.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (3, 4)
    coo = COOTensor(indices, values, shape)

    def test_pynative_1():
        return coo.indices, coo.values, coo.shape

    def test_pynative_2():
        return coo.astype(mstype.int32)

    def test_pynative_3():
        return coo.to_tuple()

    test_graph_1 = jit(test_pynative_1)
    test_graph_2 = jit(test_pynative_2)
    test_graph_3 = jit(test_pynative_3)

    py_indices, py_values, py_shape = test_pynative_1()
    py_coo = test_pynative_2()
    py_tuple = test_pynative_3()

    g_indices, g_values, g_shape = test_graph_1()
    g_coo = test_graph_2()
    g_tuple = test_graph_3()

    coo1 = COOTensor(py_indices, py_values, py_shape)
    coo2 = COOTensor(g_indices, g_values, g_shape)
    # check coo attr
    compare_coo(coo1, coo2)
    # check astype
    compare_coo(py_coo, g_coo)
    # check to_tuple
    assert len(py_tuple) == len(g_tuple)
    for i, _ in enumerate(py_tuple):
        if isinstance(py_tuple[i], Tensor):
            assert (py_tuple[i].asnumpy() == g_tuple[i].asnumpy()).all()
        else:
            assert py_tuple[i] == g_tuple[i]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_coo_bprop():
    """
    Feature: Test back-propagation with COO-related Ops.
    Description: Test back-propagation of make_coo, coo.attributes, coo.methods().
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    grad_op = ops.GradOperation(get_all=True)
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int32)
    values = Tensor([-1, 2], dtype=mstype.float32)
    dense_shape = (3, 4)

    @grad_op
    @jit
    def test_coo_tensor(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor

    @grad_op
    @jit
    def test_coo_indices(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.indices

    @grad_op
    @jit
    def test_coo_values(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.values

    @grad_op
    @jit
    def test_coo_shape(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.shape

    @grad_op
    @jit
    def test_coo_cast(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.astype(mstype.int32)

    @grad_op
    @jit
    def test_coo_dtype(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.dtype

    @grad_op
    @jit
    def test_coo_to_tuple(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.to_tuple()

    @grad_op
    @jit
    def test_coo_to_abs(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.abs()

    @grad_op
    @jit
    def test_coo_to_csr(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.to_csr()

    @grad_op
    @jit
    def test_coo_to_dense(indices, values, dense_shape):
        coo_tensor = COOTensor(indices, values, dense_shape)
        return coo_tensor.to_dense()

    all_zero = (np.zeros(indices.shape, np.int32), np.zeros(values.shape, np.float32))
    values_on = (np.zeros(indices.shape, np.int32), np.ones(values.shape, np.float32))
    values_absgrad = (np.zeros(indices.shape, np.int32), np.sign(values.asnumpy()))

    compare_res(test_coo_tensor(indices, values, dense_shape), values_on)
    compare_res(test_coo_indices(indices, values, dense_shape), all_zero)
    compare_res(test_coo_values(indices, values, dense_shape), values_on)
    compare_res(test_coo_shape(indices, values, dense_shape), all_zero)
    compare_res(test_coo_cast(indices, values, dense_shape), values_on)
    compare_res(test_coo_dtype(indices, values, dense_shape), all_zero)
    compare_res(test_coo_to_tuple(indices, values, dense_shape), values_on)
    compare_res(test_coo_to_abs(indices, values, dense_shape), values_absgrad)
    compare_res(test_coo_to_csr(indices, values, dense_shape), values_on)
    compare_res(test_coo_to_dense(indices, values, dense_shape), values_on)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dense_to_coo():
    """
    Feature: Test dense tensor to coo methods.
    Description: Test tensor.to_coo().
    Expectation: Success.
    """
    dense_tensor = Tensor([[0, 1, 2, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=mstype.float32)

    def test_to_coo(dense_tensor):
        return dense_tensor.to_coo()

    coo_tensor = test_to_coo(dense_tensor)
    coo_tensor_graph = jit(test_to_coo)(dense_tensor)
    expect = COOTensor(Tensor([[0, 1], [0, 2], [2, 0]], dtype=mstype.int32),
                       Tensor([1, 2, 1], dtype=mstype.float32),
                       (3, 4))
    assert isinstance(coo_tensor, COOTensor)
    assert isinstance(coo_tensor, COOTensor)
    compare_coo(coo_tensor, expect)
    compare_coo(coo_tensor_graph, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_coo_magic_methods():
    """
    Feature: Test coo magic methods.
    Description: Test COOTensor.__neg__, COOTensor.__add__, COOTensor.__sub__, COOTensor.__mul__, COOTensor.__div__.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    dense = Tensor([[0, 1, 2, 0], [0, 0, 2, 0], [1, 0, 0, 0]], dtype=mstype.float32)

    indices_2 = Tensor([[0, 2], [1, 2], [2, 3]], dtype=mstype.int64)
    values_2 = Tensor([3, -2, 1], dtype=mstype.float32)

    def test_coo_neg(indices, values, shape):
        coo_tensor = COOTensor(indices, values, shape)
        return -coo_tensor

    def test_coo_add_coo(indices, indices_2, values, values_2, shape):
        coo_tensor_1 = COOTensor(indices, values, shape)
        coo_tensor_2 = COOTensor(indices_2, values_2, shape)
        return coo_tensor_1 + coo_tensor_2

    def test_coo_add_dense(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return coo_tensor + dense

    def test_dense_add_coo(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return dense + coo_tensor

    def test_coo_sub_coo(indices, indices_2, values, values_2, shape):
        coo_tensor_1 = COOTensor(indices, values, shape)
        coo_tensor_2 = COOTensor(indices_2, values_2, shape)
        return coo_tensor_1 - coo_tensor_2

    def test_coo_sub_dense(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return coo_tensor - dense

    def test_dense_sub_coo(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return dense - coo_tensor

    def test_coo_mul_dense(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return coo_tensor * dense

    def test_dense_mul_coo(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return dense * coo_tensor

    def test_coo_div_dense(indices, values, shape, dense):
        coo_tensor = COOTensor(indices, values, shape)
        return coo_tensor / dense

    neg_expect = COOTensor(indices, Tensor([1, -2], mstype.float32), shape)
    neg_output = test_coo_neg(indices, values, shape)
    compare_coo(neg_output, neg_expect)
    neg_output = jit(test_coo_neg)(indices, values, shape)
    compare_coo(neg_output, neg_expect)

    coo_add_coo_expect = COOTensor(Tensor([[0, 1], [0, 2], [1, 2], [2, 3]], mstype.int64),
                                   Tensor([-1, 3, 0, 1], mstype.float32), shape)
    coo_add_coo_output = test_coo_add_coo(indices, indices_2, values, values_2, shape)
    compare_coo(coo_add_coo_output, coo_add_coo_expect)
    coo_add_coo_output = jit(test_coo_add_coo)(indices, indices_2, values, values_2, shape)
    compare_coo(coo_add_coo_output, coo_add_coo_expect)

    coo_add_dense_expect = np.array([[0, 0, 2, 0], [0, 0, 4, 0,], [1, 0, 0, 0,]], np.int64)
    coo_add_dense_output = test_coo_add_dense(indices, values, shape, dense)
    assert np.allclose(coo_add_dense_expect, coo_add_dense_output.asnumpy())
    coo_add_dense_output = jit(test_coo_add_dense)(indices, values, shape, dense)
    assert np.allclose(coo_add_dense_expect, coo_add_dense_output.asnumpy())

    dense_add_coo_output = test_dense_add_coo(indices, values, shape, dense)
    assert np.allclose(coo_add_dense_expect, dense_add_coo_output.asnumpy())
    dense_add_coo_output = jit(test_dense_add_coo)(indices, values, shape, dense)
    assert np.allclose(coo_add_dense_expect, dense_add_coo_output.asnumpy())

    coo_sub_coo_expect = COOTensor(Tensor([[0, 1], [0, 2], [1, 2], [2, 3]], mstype.int64),
                                   Tensor([-1, -3, 4, -1], mstype.float32), shape)
    coo_sub_coo_output = test_coo_sub_coo(indices, indices_2, values, values_2, shape)
    compare_coo(coo_sub_coo_output, coo_sub_coo_expect)
    coo_sub_coo_output = jit(test_coo_sub_coo)(indices, indices_2, values, values_2, shape)
    compare_coo(coo_sub_coo_output, coo_sub_coo_expect)

    coo_sub_dense_expect = np.array([[0, -2, -2, 0], [0, 0, 0, 0], [-1, 0, 0, 0]], np.int32)
    coo_sub_dense_output = test_coo_sub_dense(indices, values, shape, dense)
    assert np.allclose(coo_sub_dense_expect, coo_sub_dense_output.asnumpy())
    coo_sub_dense_output = jit(test_coo_sub_dense)(indices, values, shape, dense)
    assert np.allclose(coo_sub_dense_expect, coo_sub_dense_output.asnumpy())

    dense_sub_coo_expect = np.array([[0, 2, 2, 0], [0, 0, 0, 0], [1, 0, 0, 0]], np.int64)
    dense_sub_coo_output = test_dense_sub_coo(indices, values, shape, dense)
    assert np.allclose(dense_sub_coo_expect, dense_sub_coo_output.asnumpy())
    dense_sub_coo_output = jit(test_dense_sub_coo)(indices, values, shape, dense)
    assert np.allclose(dense_sub_coo_expect, dense_sub_coo_output.asnumpy())

    coo_mul_dense_expect = COOTensor(indices, Tensor([-1, 4], mstype.float32), shape)
    coo_mul_dense_output = test_coo_mul_dense(indices, values, shape, dense)
    compare_coo(coo_mul_dense_output, coo_mul_dense_expect)
    coo_mul_dense_output = jit(test_coo_mul_dense)(indices, values, shape, dense)
    compare_coo(coo_mul_dense_output, coo_mul_dense_expect)

    dense_mul_coo_output = test_dense_mul_coo(indices, values, shape, dense)
    compare_coo(dense_mul_coo_output, coo_mul_dense_expect)
    dense_mul_coo_output = jit(test_dense_mul_coo)(indices, values, shape, dense)
    compare_coo(dense_mul_coo_output, coo_mul_dense_expect)

    coo_div_dense_expect = COOTensor(indices, Tensor([-1, 1], mstype.float32), shape)
    coo_div_dense_output = test_coo_div_dense(indices, values, shape, dense)
    compare_coo(coo_div_dense_output, coo_div_dense_expect)
    coo_div_dense_output = jit(test_coo_div_dense)(indices, values, shape, dense)
    compare_coo(coo_div_dense_output, coo_div_dense_expect)
