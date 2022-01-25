# Copyright 2021 Huawei Technologies Co., Ltd
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
"""smoke tests for CSR operations"""

import os
import pytest
import numpy as np

from mindspore import Tensor, CSRTensor, ms_function, nn, context, ops
from mindspore.ops.operations import _csr_ops
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export, load
from mindspore.ops import functional as F


context.set_context(mode=context.GRAPH_MODE)

def compare_csr(csr1, csr2):
    assert isinstance(csr1, CSRTensor)
    assert isinstance(csr2, CSRTensor)
    assert (csr1.indptr.asnumpy() == csr2.indptr.asnumpy()).all()
    assert (csr1.indices.asnumpy() == csr2.indices.asnumpy()).all()
    assert (csr1.values.asnumpy() == csr2.values.asnumpy()).all()
    assert csr1.shape == csr2.shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_make_csr():
    """
    Feature: Test CSRTensor Constructor in Graph and PyNative.
    Description: Test CSRTensor(indptr, indices, values, shape) and CSRTensor(CSRTensor)
    Expectation: Success.
    """
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 6)
    def test_pynative():
        return CSRTensor(indptr, indices, values, shape)
    test_graph = ms_function(test_pynative)

    csr1 = test_pynative()
    csr2 = test_graph()
    compare_csr(csr1, csr2)
    csr3 = CSRTensor(csr_tensor=csr2)
    compare_csr(csr3, csr2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_attr():
    """
    Feature: Test CSRTensor GetAttr in Graph and PyNative.
    Description: Test CSRTensor.indptr, CSRTensor.indices, CSRTensor.values, CSRTensor.shape.
    Expectation: Success.
    """
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 6)
    csr = CSRTensor(indptr, indices, values, shape)

    def test_pynative_1():
        return csr.indptr, csr.indices

    def test_pynative_2():
        return csr.values, csr.shape

    def test_pynative_3():
        return csr.astype(mstype.int32)

    def test_pynative_4():
        return csr.to_tuple()

    test_graph_1 = ms_function(test_pynative_1)
    test_graph_2 = ms_function(test_pynative_2)
    test_graph_3 = ms_function(test_pynative_3)
    test_graph_4 = ms_function(test_pynative_4)

    py_indptr, py_indices = test_pynative_1()
    py_values, py_shape = test_pynative_2()
    py_csr = test_pynative_3()
    py_tuple = test_pynative_4()

    g_indptr, g_indices = test_graph_1()
    g_values, g_shape = test_graph_2()
    g_csr = test_graph_3()
    g_tuple = test_graph_4()

    csr1 = CSRTensor(py_indptr, py_indices, py_values, py_shape)
    csr2 = CSRTensor(g_indptr, g_indices, g_values, g_shape)
    # check csr attr
    compare_csr(csr1, csr2)
    # check astype
    compare_csr(py_csr, g_csr)
    # check to_tuple
    assert len(py_tuple) == len(g_tuple)
    for i, _ in enumerate(py_tuple):
        if isinstance(py_tuple[i], Tensor):
            assert (py_tuple[i].asnumpy() == g_tuple[i].asnumpy()).all()
        else:
            assert py_tuple[i] == g_tuple[i]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_tensor_in_while():
    """
    Feature: Test CSRTensor in while loop.
    Description: Test CSRTensor computation in while loop.
    Expectation: Success.
    """
    class CSRTensorValuesDouble(nn.Cell):

        def construct(self, x):
            indptr = x.indptr
            indices = x.indices
            values = x.values * 2
            shape = x.shape
            return CSRTensor(indptr, indices, values, shape)

    class CSRTensorValuesAdd2(nn.Cell):

        def construct(self, x):
            indptr = x.indptr
            indices = x.indices
            values = x.values + 2
            shape = x.shape
            return CSRTensor(indptr, indices, values, shape)

    class CSRTensorWithControlWhile(nn.Cell):
        def __init__(self, shape):
            super(CSRTensorWithControlWhile, self).__init__()
            self.op1 = CSRTensorValuesDouble()
            self.op2 = CSRTensorValuesAdd2()
            self.shape = shape

        @ms_function
        def construct(self, a, b, indptr, indices, values):
            x = CSRTensor(indptr, indices, values, self.shape)
            x = self.op2(x)
            while a > b:
                x = self.op1(x)
                b = b + 1
            return x
    a = Tensor(3, mstype.int32)
    b = Tensor(0, mstype.int32)
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 6)
    net = CSRTensorWithControlWhile(shape)
    out = net(a, b, indptr, indices, values)
    assert np.allclose(out.indptr.asnumpy(), indptr.asnumpy(), .0, .0)
    assert np.allclose(out.indices.asnumpy(), indices.asnumpy(), .0, .0)
    assert np.allclose((values.asnumpy() + 2) * 8, out.values.asnumpy(), .0, .0)
    assert shape == out.shape

    # Test Export MindIR
    file_name = "csrtensor_with_control_while_net"
    export(net, a, b, indptr, indices, values, file_name=file_name, file_format="MINDIR")
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(a, b, indptr, indices, values)
    assert np.allclose(out.indptr.asnumpy(), outputs_after_load.indptr.asnumpy())
    assert np.allclose(out.indices.asnumpy(), outputs_after_load.indices.asnumpy())
    assert np.allclose(out.values.asnumpy(), outputs_after_load.values.asnumpy())
    assert out.shape == outputs_after_load.shape


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_tensor_in_while_cpu():
    """
    Feature: Test CSRTensor in while loop.
    Description: Test CSRTensor computation in while loop.
    Expectation: Success.
    """
    class CSRTensorValuesDouble(nn.Cell):

        def construct(self, x):
            indptr = x.indptr
            indices = x.indices
            values = x.values * 2
            shape = x.shape
            return CSRTensor(indptr, indices, values, shape)

    class CSRTensorValuesAdd2(nn.Cell):

        def construct(self, x):
            indptr = x.indptr
            indices = x.indices
            values = x.values + 2
            shape = x.shape
            return CSRTensor(indptr, indices, values, shape)

    class CSRTensorWithControlWhile(nn.Cell):
        def __init__(self, shape):
            super(CSRTensorWithControlWhile, self).__init__()
            self.op1 = CSRTensorValuesDouble()
            self.op2 = CSRTensorValuesAdd2()
            self.shape = shape

        @ms_function
        def construct(self, a, b, indptr, indices, values):
            x = CSRTensor(indptr, indices, values, self.shape)
            x = self.op2(x)
            while a > b:
                x = self.op1(x)
                b = b + 1
            return x
    a = Tensor(3, mstype.int32)
    b = Tensor(0, mstype.int32)
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 6)
    net = CSRTensorWithControlWhile(shape)
    out = net(a, b, indptr, indices, values)
    assert np.allclose(out.indptr.asnumpy(), indptr.asnumpy(), .0, .0)
    assert np.allclose(out.indices.asnumpy(), indices.asnumpy(), .0, .0)
    assert np.allclose((values.asnumpy() + 2) * 8, out.values.asnumpy(), .0, .0)
    assert shape == out.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_ops():
    """
    Feature: Test CSR-related Ops.
    Description: Test CSRReduceSum, CSRMul, CSRMV.
    Expectation: Success.
    """
    csr_reducesum = _csr_ops.CSRReduceSum()
    csrmv = _csr_ops.CSRMV()

    indptr = Tensor([0, 1, 2], dtype=mstype.int32)
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([2, 1], dtype=mstype.float32)
    dense_shape = (2, 4)

    dense_tensor = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
    dense_vector = Tensor([[1.], [1], [1], [1]], dtype=mstype.float32)
    csr_tensor = CSRTensor(indptr, indices, values, dense_shape)

    def test_ops_pynative_dense():
        dense1 = csr_reducesum(csr_tensor, 1)
        dense2 = csrmv(csr_tensor, dense_vector)
        return dense1, dense2

    def test_ops_pynative_sparse():
        sparse1 = csr_tensor * dense_tensor
        sparse2 = dense_tensor * csr_tensor
        sparse3 = csr_tensor / dense_tensor
        return sparse1, sparse2, sparse3

    test_ops_graph_dense = ms_function(test_ops_pynative_dense)
    test_ops_graph_sparse = ms_function(test_ops_pynative_sparse)

    pynative_res_dense = test_ops_pynative_dense()
    graph_res_dense = test_ops_graph_dense()
    expect1 = np.array([[2.], [1.]], dtype=np.float32)
    expect2 = np.array([[2.], [1.]], dtype=np.float32)
    assert np.allclose(pynative_res_dense[0].asnumpy(), expect1)
    assert np.allclose(pynative_res_dense[1].asnumpy(), expect2)
    assert np.allclose(graph_res_dense[0].asnumpy(), expect1)
    assert np.allclose(graph_res_dense[1].asnumpy(), expect2)

    pynative_res_sparse = test_ops_pynative_sparse()
    graph_res_sparse = test_ops_graph_sparse()
    expect3 = np.array([2., 1.], dtype=np.float32)
    assert np.allclose(pynative_res_sparse[0].values.asnumpy(), expect3)
    assert np.allclose(pynative_res_sparse[1].values.asnumpy(), expect3)
    assert np.allclose(pynative_res_sparse[2].values.asnumpy(), expect3)
    assert np.allclose(graph_res_sparse[0].values.asnumpy(), expect3)
    assert np.allclose(graph_res_sparse[1].values.asnumpy(), expect3)
    assert np.allclose(graph_res_sparse[2].values.asnumpy(), expect3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csrtensor_export_and_import_mindir():
    """
    Feature: Test exporting and loading CSRTensor MindIR.
    Description: Test export and load.
    Expectation: Success.
    """
    class TestCSRTensor(nn.Cell):
        def __init__(self, shape):
            super(TestCSRTensor, self).__init__()
            self.shape = shape

        def construct(self, indptr, indices, values):
            return CSRTensor(indptr, indices, values, self.shape)

    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([2, 1], dtype=mstype.float32)
    shape = (2, 4)
    net = TestCSRTensor(shape)

    file_name = "csrtensor_net"
    export(net, indptr, indices, values, file_name=file_name, file_format="MINDIR")
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    out = net(indptr, indices, values)
    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(indptr, indices, values)
    assert np.allclose(out.indptr.asnumpy(), outputs_after_load.indptr.asnumpy())
    assert np.allclose(out.indices.asnumpy(), outputs_after_load.indices.asnumpy())
    assert np.allclose(out.values.asnumpy(), outputs_after_load.values.asnumpy())
    assert out.shape == outputs_after_load.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csrops_export_and_import_mindir():
    """
    Feature: Test exporting and loading CSRTensor MindIR in a net.
    Description: Test export and load.
    Expectation: Success.
    """
    class TestCSRNet(nn.Cell):
        def __init__(self, shape):
            super(TestCSRNet, self).__init__()
            self.shape = shape
            self.csr_reducesum = _csr_ops.CSRReduceSum()
            self.csr_mv = _csr_ops.CSRMV()

        def construct(self, indptr, indices, values, dence_tensor, dense_vector):
            csr_tensor = CSRTensor(indptr, indices, values, self.shape)
            dense1 = self.csr_reducesum(csr_tensor, 1)
            dense2 = self.csr_mv(csr_tensor, dense_vector)
            dense3 = dense1 * dense2
            sparse1 = csr_tensor * dence_tensor
            sparse2 = dence_tensor * csr_tensor
            return dense1, dense2, dense3, sparse1, sparse2

    indptr = Tensor([0, 1, 2], dtype=mstype.int32)
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([2, 1], dtype=mstype.float32)
    shape = (2, 4)
    dense_tensor = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
    dense_vector = Tensor([[1.], [1], [1], [1]], dtype=mstype.float32)

    net = TestCSRNet(shape)
    file_name = "csrops_net"
    export(net, indptr, indices, values, dense_tensor, dense_vector, file_name=file_name, file_format="MINDIR")
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    out = net(indptr, indices, values, dense_tensor, dense_vector)
    expect0 = np.array([[2.], [1.]], dtype=np.float32)
    expect1 = np.array([[2.], [1.]], dtype=np.float32)
    expect2 = np.array([[4.], [1.]], dtype=np.float32)
    expect3 = np.array([2., 1.], dtype=np.float32)
    assert np.allclose(out[0].asnumpy(), expect0)
    assert np.allclose(out[1].asnumpy(), expect1)
    assert np.allclose(out[2].asnumpy(), expect2)
    assert np.allclose(out[3].values.asnumpy(), expect3)
    assert np.allclose(out[4].values.asnumpy(), expect3)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(indptr, indices, values, dense_tensor, dense_vector)
    assert np.allclose(out[0].asnumpy(), outputs_after_load[0].asnumpy())
    assert np.allclose(out[1].asnumpy(), outputs_after_load[1].asnumpy())
    assert np.allclose(out[2].asnumpy(), outputs_after_load[2].asnumpy())
    assert np.allclose(out[3].values.asnumpy(), outputs_after_load[3].values.asnumpy())
    assert np.allclose(out[4].values.asnumpy(), outputs_after_load[4].values.asnumpy())
    assert out[3].shape == outputs_after_load[3].shape
    assert out[4].shape == outputs_after_load[4].shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isinstance_csr_tensor():
    """
    Feature: Test isinstance.
    Description: Test: isinstance(x, CSRTensor).
    Expectation: Success.
    """
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([2, 1], dtype=mstype.float32)
    shape = (2, 4)

    def pynative_test_csr_tensor():
        x = CSRTensor(indptr, indices, values, shape)
        # Test input CSRTensor
        is_tensor = isinstance(x, Tensor)
        is_bool = isinstance(x, bool)
        is_float = isinstance(x, float)
        is_tuple = isinstance(x, (Tensor, CSRTensor, int, float))
        is_csr_tensor = isinstance(x, CSRTensor)

        # Test input Tensor
        is_tensor_2 = isinstance(indptr, CSRTensor)
        is_tuple_2 = isinstance(indptr, (Tensor, CSRTensor))
        return is_tensor, is_bool, is_float, is_tuple, is_csr_tensor, is_tensor_2, is_tuple_2
    graph_test_csr_tensor = ms_function(pynative_test_csr_tensor)

    out1 = pynative_test_csr_tensor()
    out2 = graph_test_csr_tensor()
    assert out1 == (False, False, False, True, True, False, True)
    assert out2 == (False, False, False, True, True, False, True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dtype_csr_tensor():
    """
    Feature: Test F.dtype with CSRTensor.
    Description: Test: F.dtype(x).
    Expectation: Success.
    """
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([2, 1], dtype=mstype.float32)
    shape = (2, 4)

    def pynative_test():
        x = CSRTensor(indptr, indices, values, shape)
        return F.dtype(x), x.dtype
    graph_test = ms_function(pynative_test)

    out1, out2 = pynative_test()
    out3, out4 = graph_test()
    assert out1 in [mstype.float32]
    assert out2 in [mstype.float32]
    assert out3 in [mstype.float32]
    assert out4 in [mstype.float32]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_bprop():
    """
    Feature: Test back-propagation with CSR-related Ops.
    Description: Test CSRReduceSum, CSRMul, CSRMV, CSRTensor.to_coo(), CSRTensor.to_dense().
    Expectation: Success.
    """
    csr_reduce_sum = _csr_ops.CSRReduceSum()
    csrmv = _csr_ops.CSRMV()
    grad_op = ops.GradOperation(get_all=True)

    def test_csr_mul(csr_tensor, dense):
        return csr_tensor * dense

    def test_csr_reduce_sum(csr_tensor, axis):
        return csr_reduce_sum(csr_tensor, axis)

    def test_csrmv(csr_tensor, dense):
        return csrmv(csr_tensor, dense)

    test_csr_mul_grad_pynative = grad_op(test_csr_mul)
    test_csr_mul_grad_graph = ms_function(test_csr_mul_grad_pynative)
    test_csr_reduce_sum_grad_pynative = grad_op(test_csr_reduce_sum)
    test_csr_reduce_sum_grad_graph = ms_function(test_csr_reduce_sum_grad_pynative)
    test_csrmv_grad_pynative = grad_op(test_csrmv)
    test_csrmv_grad_graph = ms_function(test_csrmv_grad_pynative)

    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6), dtype=mstype.float32)
    dense_shape = (3, 4)
    csr_tensor = CSRTensor(indptr, indices, values, dense_shape)

    csr_mv_arg = Tensor([[1], [2], [3], [4]], dtype=mstype.float32)
    csr_mv_expect_1 = np.array([4, 1, 2, 3, 2, 4], dtype=np.float32)
    csr_mv_expect_2 = np.array([[1], [6], [3], [5]], dtype=np.float32)
    csr_mv_output_1, csr_mv_output_2 = test_csrmv_grad_pynative(csr_tensor, csr_mv_arg)
    assert np.allclose(csr_mv_output_1.values.asnumpy(), csr_mv_expect_1)
    assert np.allclose(csr_mv_output_2.asnumpy(), csr_mv_expect_2)
    csr_mv_output_1, csr_mv_output_2 = test_csrmv_grad_graph(csr_tensor, csr_mv_arg)
    assert np.allclose(csr_mv_output_1.values.asnumpy(), csr_mv_expect_1)
    assert np.allclose(csr_mv_output_2.asnumpy(), csr_mv_expect_2)

    csr_reduce_sum_expect = np.ones(6, dtype=np.float32)
    csr_reduce_sum_output = test_csr_reduce_sum_grad_pynative(csr_tensor, 1)
    assert np.allclose(csr_reduce_sum_output[0].values.asnumpy(), csr_reduce_sum_expect)
    csr_reduce_sum_output = test_csr_reduce_sum_grad_graph(csr_tensor, 1)
    assert np.allclose(csr_reduce_sum_output[0].values.asnumpy(), csr_reduce_sum_expect)

    csr_mul_arg_1 = Tensor([[1], [2], [3]], dtype=mstype.float32)
    csr_mul_expect_1_1 = np.array([1, 2, 2, 2, 3, 3], dtype=np.float32)
    csr_mul_expect_1_2 = np.array([[0], [6], [9]], dtype=np.float32)
    csr_mul_output_1_1, csr_mul_output_1_2 = test_csr_mul_grad_pynative(csr_tensor, csr_mul_arg_1)
    assert np.allclose(csr_mul_output_1_1.values.asnumpy(), csr_mul_expect_1_1)
    assert np.allclose(csr_mul_output_1_2.asnumpy(), csr_mul_expect_1_2)
    csr_mul_output_1_1, csr_mul_output_1_2 = test_csr_mul_grad_graph(csr_tensor, csr_mul_arg_1)
    assert np.allclose(csr_mul_output_1_1.values.asnumpy(), csr_mul_expect_1_1)
    assert np.allclose(csr_mul_output_1_2.asnumpy(), csr_mul_expect_1_2)

    csr_mul_arg_2 = Tensor(np.arange(12).reshape(3, 4), dtype=mstype.float32)
    csr_mul_expect_2_1 = np.array([3, 4, 5, 6, 9, 11], dtype=np.float32)
    csr_mul_expect_2_2 = np.array([[0, 0, 0, 0], [1, 2, 3, 0], [0, 4, 0, 5]], np.float32)
    csr_mul_output_2_1, csr_mul_output_2_2 = test_csr_mul_grad_pynative(csr_tensor, csr_mul_arg_2)
    assert np.allclose(csr_mul_output_2_1.values.asnumpy(), csr_mul_expect_2_1)
    assert np.allclose(csr_mul_output_2_2.asnumpy(), csr_mul_expect_2_2)
    csr_mul_output_2_1, csr_mul_output_2_2 = test_csr_mul_grad_graph(csr_tensor, csr_mul_arg_2)
    assert np.allclose(csr_mul_output_2_1.values.asnumpy(), csr_mul_expect_2_1)
    assert np.allclose(csr_mul_output_2_2.asnumpy(), csr_mul_expect_2_2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_method():
    """
    Feature: Test csr tensor methods.
    Description: Test csr_tensor.to_coo(), csr_tensor.to_dense().
    Expectation: Success.
    """
    class CSRToCOONet(nn.Cell):
        def construct(self, csr_tensor):
            return csr_tensor.to_coo()

    class CSRToDenseNet(nn.Cell):
        def construct(self, csr_tensor):
            return csr_tensor.to_dense()

    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6), dtype=mstype.float32)
    dense_shape = (3, 4)
    csr_tensor = CSRTensor(indptr, indices, values, dense_shape)

    to_coo_output = CSRToCOONet()(csr_tensor)
    to_coo_expect_1 = np.array([[0, 3], [1, 0], [1, 1], [1, 2], [2, 1], [2, 3]], dtype=np.int32)
    to_coo_expect_2 = np.arange(6).astype(np.float32)
    assert np.allclose(to_coo_output.indices.asnumpy(), to_coo_expect_1)
    assert np.allclose(to_coo_output.values.asnumpy(), to_coo_expect_2)

    to_dense_output = CSRToDenseNet()(csr_tensor)
    to_dense_expect = np.array([[0, 0, 0, 0], [1, 2, 3, 0], [0, 4, 0, 5]], np.float32)
    assert np.allclose(to_dense_output.asnumpy(), to_dense_expect)
