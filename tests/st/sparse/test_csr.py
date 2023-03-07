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

from mindspore import Tensor, CSRTensor, jit, nn, ops
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export, load
from mindspore.ops import functional as F
from mindspore.ops.operations import _csr_ops

from .sparse_utils import get_platform, compare_res, compare_csr


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
    if get_platform() != "linux":
        return
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 6)

    def test_pynative():
        return CSRTensor(indptr, indices, values, shape)
    test_graph = jit(test_pynative)

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
def test_make_csr_empty():
    """
    Feature: Test CSRTensor Constructor in Graph and PyNative.
    Description: Test CSRTensor(indptr, indices, values, shape) and CSRTensor(CSRTensor)
    Expectation: Success.
    """
    indptr = Tensor([], dtype=mstype.int32)
    indices = Tensor([], dtype=mstype.int32)
    values = Tensor([], dtype=mstype.float32)
    shape = (2, 6)

    def test_pynative():
        return CSRTensor(indptr, indices, values, shape)
    test_graph = jit(test_pynative)

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
    if get_platform() != "linux":
        return
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

    test_graph_1 = jit(test_pynative_1)
    test_graph_2 = jit(test_pynative_2)
    test_graph_3 = jit(test_pynative_3)
    test_graph_4 = jit(test_pynative_4)

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

        @jit
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

        @jit
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
def test_batch_csr_ops():
    """
    Feature: Test Batch CSR-related Ops.
    Description: Test CSRReduceSum, CSRMul, CSRGather.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    csr_gather = _csr_ops.CSRGather()

    indptr = Tensor([0, 1, 1, 2, 2], dtype=mstype.int32)
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([[2, 1, 3], [2, 1, 3]], dtype=mstype.float32)
    dense_shape = (4, 2, 3)
    dense_tensor = Tensor(
        [[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]],
        dtype=mstype.float32)
    csr_tensor = CSRTensor(indptr, indices, values, dense_shape)

    def test_ops_pynative_gather():
        dense = csr_gather(indptr, indices, dense_tensor, dense_shape)
        return dense

    def test_ops_pynative_reducesum():
        dense = F.csr_reduce_sum(csr_tensor, 1)
        return dense

    def test_ops_pynative_sparse_elemwise():
        sparse1 = csr_tensor * dense_tensor
        sparse2 = csr_tensor / dense_tensor
        return sparse1, sparse2

    # TODO(PyTrace): PyTrace Async bug.
    test_ops_graph_reducesum = jit(test_ops_pynative_reducesum)
    graph_res_reducesum = test_ops_graph_reducesum()
    res_reducesum = test_ops_pynative_reducesum()
    expect1 = np.array([[2., 1., 3.]], dtype=np.float32)
    expect2 = np.array([[2., 1., 3.]], dtype=np.float32)
    assert np.allclose(res_reducesum[0].asnumpy(), expect1)
    assert np.allclose(res_reducesum[2].asnumpy(), expect2)
    assert np.allclose(graph_res_reducesum[0].asnumpy(), expect1)
    assert np.allclose(graph_res_reducesum[2].asnumpy(), expect2)

    # TODO(PyTrace): PyTrace Async bug.
    test_ops_graph_elemwise = jit(test_ops_pynative_sparse_elemwise)
    graph_res_elemwise = test_ops_graph_elemwise()
    res_elemwise = test_ops_pynative_sparse_elemwise()
    expect3 = np.array([[2., 1., 3.], [4., 2., 6.]], dtype=np.float32)
    expect4 = np.array([[2., 1., 3.], [1., 0.5, 1.5]], dtype=np.float32)
    assert np.allclose(res_elemwise[0].values.asnumpy(), expect3)
    assert np.allclose(res_elemwise[1].values.asnumpy(), expect4)
    assert np.allclose(graph_res_elemwise[0].values.asnumpy(), expect3)
    assert np.allclose(graph_res_elemwise[1].values.asnumpy(), expect4)

    expect5 = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    res_gather = test_ops_pynative_gather()
    test_ops_graph_gather = jit(test_ops_pynative_gather)
    graph_res_gather = test_ops_graph_gather()
    assert np.allclose(res_gather.asnumpy(), expect5)
    assert np.allclose(graph_res_gather.asnumpy(), expect5)


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
    if get_platform() != "linux":
        return

    indptr = Tensor([0, 1, 2], dtype=mstype.int32)
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([2, 1], dtype=mstype.float32)
    dense_shape = (2, 4)

    dense_tensor = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
    dense_vector = Tensor([[1.], [1], [1], [1]], dtype=mstype.float32)
    csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
    dense_matrix = Tensor([[1., 2.], [1, 2.], [1, 2.], [1., 2.]], dtype=mstype.float32)

    def test_ops_pynative_dense():
        dense1 = F.csr_reduce_sum(csr_tensor, 1)
        dense2 = F.csr_mv(csr_tensor, dense_vector)
        dense3 = csr_tensor.mm(dense_matrix)
        return dense1, dense2, dense3

    def test_ops_pynative_sparse():
        sparse1 = csr_tensor * dense_tensor
        sparse2 = dense_tensor * csr_tensor
        sparse3 = csr_tensor / dense_tensor
        return sparse1, sparse2, sparse3

    test_ops_graph_dense = jit(test_ops_pynative_dense)
    test_ops_graph_sparse = jit(test_ops_pynative_sparse)

    # TODO(PyTrace): PyTrace async bug.
    graph_res_dense = test_ops_graph_dense()
    pynative_res_dense = test_ops_pynative_dense()
    expect1 = np.array([[2.], [1.]], dtype=np.float32)
    expect2 = np.array([[2.], [1.]], dtype=np.float32)
    expect3 = np.array([[2., 4.], [1., 2.]], dtype=np.float32)
    assert np.allclose(pynative_res_dense[0].asnumpy(), expect1)
    assert np.allclose(pynative_res_dense[1].asnumpy(), expect2)
    assert np.allclose(pynative_res_dense[2].asnumpy(), expect3)
    assert np.allclose(graph_res_dense[0].asnumpy(), expect1)
    assert np.allclose(graph_res_dense[1].asnumpy(), expect2)
    assert np.allclose(graph_res_dense[2].asnumpy(), expect3)

    # TODO(PyTrace): PyTrace async bug.
    graph_res_sparse = test_ops_graph_sparse()
    pynative_res_sparse = test_ops_pynative_sparse()
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
    if get_platform() != "linux":
        return

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

        def construct(self, indptr, indices, values, dense_tensor, dense_vector):
            csr_tensor = CSRTensor(indptr, indices, values, self.shape)
            dense1 = F.csr_reduce_sum(csr_tensor, 1)
            dense2 = F.csr_mv(csr_tensor, dense_vector)
            dense3 = dense1 * dense2
            sparse1 = csr_tensor * dense_tensor
            sparse2 = dense_tensor * csr_tensor
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
    if get_platform() != "linux":
        return
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
    graph_test_csr_tensor = jit(pynative_test_csr_tensor)

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
    if get_platform() != "linux":
        return
    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([2, 1], dtype=mstype.float32)
    shape = (2, 4)

    def pynative_test():
        x = CSRTensor(indptr, indices, values, shape)
        return F.dtype(x), x.dtype
    graph_test = jit(pynative_test)

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
def test_bprop():
    """
    Feature: Test back-propagation with CSR-related Ops.
    Description: Test CSRReduceSum, CSRMul, CSRDiv, CSRMV.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    grad_op = ops.GradOperation(get_all=True)

    @grad_op
    @jit
    def test_csr_mul(indptr, indices, values, shape, dense):
        csr_tensor = CSRTensor(indptr, indices, values, shape)
        return csr_tensor * dense

    @grad_op
    @jit
    def test_csr_div(indptr, indices, values, shape, dense):
        csr_tensor = CSRTensor(indptr, indices, values, shape)
        return csr_tensor / dense

    @grad_op
    @jit
    def test_csr_reduce_sum(indptr, indices, values, shape, axis):
        csr_tensor = CSRTensor(indptr, indices, values, shape)
        return F.csr_reduce_sum(csr_tensor, axis)

    @grad_op
    @jit
    def test_csrmv(indptr, indices, values, shape, dense):
        csr_tensor = CSRTensor(indptr, indices, values, shape)
        return F.csr_mv(csr_tensor, dense)

    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6), dtype=mstype.float32)
    dense_shape = (3, 4)

    csr_mv_arg = Tensor([[1], [2], [3], [4]], dtype=mstype.float32)
    csr_mv_expect_1 = np.array([4, 1, 2, 3, 2, 4], dtype=np.float32)
    csr_mv_expect_2 = np.array([[1], [6], [3], [5]], dtype=np.float32)
    csr_mv_output = test_csrmv(indptr, indices, values, dense_shape, csr_mv_arg)
    # indptr, indices, values, dense_grad
    assert len(csr_mv_output) == 4
    assert np.allclose(csr_mv_output[2].asnumpy(), csr_mv_expect_1)
    assert np.allclose(csr_mv_output[3].asnumpy(), csr_mv_expect_2)

    csr_reduce_sum_expect_1 = np.ones(6, dtype=np.float32)
    csr_reduce_sum_output_1 = test_csr_reduce_sum(indptr, indices, values, dense_shape, 1)
    assert len(csr_reduce_sum_output_1) == 3
    assert np.allclose(csr_reduce_sum_output_1[2].asnumpy(), csr_reduce_sum_expect_1)

    csr_mul_arg_1 = Tensor([[1], [2], [3]], dtype=mstype.float32)
    csr_mul_expect_1_1 = np.array([1, 2, 2, 2, 3, 3], dtype=np.float32)
    csr_mul_expect_1_2 = np.array([[0], [6], [9]], dtype=np.float32)
    csr_mul_output_1 = test_csr_mul(indptr, indices, values, dense_shape, csr_mul_arg_1)
    assert len(csr_mul_output_1) == 4
    assert np.allclose(csr_mul_output_1[2].asnumpy(), csr_mul_expect_1_1)
    assert np.allclose(csr_mul_output_1[3].asnumpy(), csr_mul_expect_1_2)

    csr_mul_arg_2 = Tensor(np.arange(12).reshape(3, 4), dtype=mstype.float32)
    csr_mul_expect_2_1 = np.array([3, 4, 5, 6, 9, 11], dtype=np.float32)
    csr_mul_expect_2_2 = np.array([[0, 0, 0, 0], [1, 2, 3, 0], [0, 4, 0, 5]], np.float32)
    csr_mul_output_2 = test_csr_mul(indptr, indices, values, dense_shape, csr_mul_arg_2)
    assert len(csr_mul_output_2) == 4
    assert np.allclose(csr_mul_output_2[2].asnumpy(), csr_mul_expect_2_1)
    assert np.allclose(csr_mul_output_2[3].asnumpy(), csr_mul_expect_2_2)

    csr_div_expect_1_1 = np.array([1, 0.5, 0.5, 0.5, 0.3333333, 0.3333333], dtype=np.float32)
    csr_div_expect_1_2 = np.array([[0], [-1.5], [-1]], dtype=np.float32)
    csr_div_arg_1 = Tensor([[1], [2], [3]], dtype=mstype.float32)
    csr_div_output_1 = test_csr_div(indptr, indices, values, dense_shape, csr_div_arg_1)
    assert len(csr_div_output_1) == 4
    assert np.allclose(csr_div_output_1[2].asnumpy(), csr_div_expect_1_1)
    assert np.allclose(csr_div_output_1[3].asnumpy(), csr_div_expect_1_2)

    csr_div_arg_2 = Tensor(np.arange(1, 13).reshape(3, 4), dtype=mstype.float32)
    csr_div_expect_2_1 = np.array([0.25, 0.2, 0.16666667, 0.14285715, 0.1, 0.0833333], dtype=np.float32)
    csr_div_expect_2_2 = np.array(
        [[0, 0, 0, 0], [-0.04, -0.05555556, -0.06122449, 0], [0, -0.04, 0, -0.03472222]], dtype=np.float32)
    csr_div_output_2 = test_csr_div(indptr, indices, values, dense_shape, csr_div_arg_2)
    assert len(csr_div_output_2) == 4
    assert np.allclose(csr_div_output_2[2].asnumpy(), csr_div_expect_2_1)
    assert np.allclose(csr_div_output_2[3].asnumpy(), csr_div_expect_2_2)


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
    if get_platform() != "linux":
        return

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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bprop2():
    """
    Feature: Test back-propagation with CSR-related Ops.
    Description: Test back-propagation of make_csr, csr.attributes, csr.methods().
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    grad_op = ops.GradOperation(get_all=True)
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    dense_shape = (3, 4)

    @grad_op
    @jit
    def test_csr_tensor(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor

    @grad_op
    @jit
    def test_csr_indptr(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.indptr

    @grad_op
    @jit
    def test_csr_indices(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.indices

    @grad_op
    @jit
    def test_csr_values(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.values

    @grad_op
    @jit
    def test_csr_shape(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.shape

    @grad_op
    @jit
    def test_csr_cast(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.astype(mstype.int32)

    @grad_op
    @jit
    def test_csr_dtype(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.dtype

    @grad_op
    @jit
    def test_csr_to_tuple(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.to_tuple()

    @grad_op
    @jit
    def test_csr_to_abs(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.abs()

    @grad_op
    @jit
    def test_csr_to_coo(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.to_coo()

    @grad_op
    @jit
    def test_csr_to_dense(indptr, indices, values, dense_shape):
        csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
        return csr_tensor.to_dense()

    all_zero = (np.zeros(indptr.shape, np.int32), np.zeros(indices.shape, np.int32), np.zeros(values.shape, np.float32))
    values_on = (np.zeros(indptr.shape, np.int32), np.zeros(indices.shape, np.int32), np.ones(values.shape, np.float32))
    values_absgrad = (np.zeros(indptr.shape, np.int32), np.zeros(indices.shape, np.int32), np.sign(values.asnumpy()))
    compare_res(test_csr_tensor(indptr, indices, values, dense_shape), values_on)
    compare_res(test_csr_indptr(indptr, indices, values, dense_shape), all_zero)
    compare_res(test_csr_indices(indptr, indices, values, dense_shape), all_zero)
    compare_res(test_csr_values(indptr, indices, values, dense_shape), values_on)
    compare_res(test_csr_cast(indptr, indices, values, dense_shape), values_on)
    compare_res(test_csr_shape(indptr, indices, values, dense_shape), all_zero)
    compare_res(test_csr_dtype(indptr, indices, values, dense_shape), all_zero)
    compare_res(test_csr_to_tuple(indptr, indices, values, dense_shape), values_on)
    compare_res(test_csr_to_abs(indptr, indices, values, dense_shape), values_absgrad)
    compare_res(test_csr_to_coo(indptr, indices, values, dense_shape), values_on)
    compare_res(test_csr_to_dense(indptr, indices, values, dense_shape), values_on)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dense_to_csr():
    """
    Feature: Test dense tensor to csr methods.
    Description: Test tensor.to_csr().
    Expectation: Success.
    """
    dense_tensor = Tensor([[0, 1, 2, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=mstype.float32)
    grad_op = ops.GradOperation(get_all=True, sens_param=True)
    def test_to_csr(dense_tensor):
        return dense_tensor.to_csr()

    csr_tensor = test_to_csr(dense_tensor)
    csr_tensor_graph = jit(test_to_csr)(dense_tensor)
    expect = CSRTensor(Tensor([0, 2, 2, 3], dtype=mstype.int32),
                       Tensor([1, 2, 0], dtype=mstype.int32),
                       Tensor([1, 2, 1], dtype=mstype.float32),
                       (3, 4))
    assert isinstance(csr_tensor, CSRTensor)
    assert isinstance(csr_tensor_graph, CSRTensor)
    compare_csr(csr_tensor, expect)
    compare_csr(csr_tensor_graph, expect)

    dense_tensor_grad = grad_op(test_to_csr)(dense_tensor, expect)
    assert (dense_tensor_grad[0].asnumpy() == np.array([[0, 1, 2, 0], [0, 0, 0, 0], [1, 0, 0, 0]])).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_magic_methods():
    """
    Feature: Test csr magic methods.
    Description: Test CSRTensor.__neg__, CSRTensor.__add__, CSRTensor.__sub__.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)

    indptr_2 = Tensor([0, 2, 3, 4], dtype=mstype.int32)
    indices_2 = Tensor([2, 3, 0, 1], dtype=mstype.int32)
    values_2 = Tensor(np.arange(4) - 2.5, dtype=mstype.float32)

    def test_csr_neg(indptr, indices, values, shape):
        csr_tensor = CSRTensor(indptr, indices, values, shape)
        return -csr_tensor

    def test_csr_add(indptr, indptr_2, indices, indices_2, values, values_2, shape):
        csr_tensor_1 = CSRTensor(indptr, indices, values, shape)
        csr_tensor_2 = CSRTensor(indptr_2, indices_2, values_2, shape)
        return csr_tensor_1 + csr_tensor_2

    def test_csr_sub(indptr, indptr_2, indices, indices_2, values, values_2, shape):
        csr_tensor_1 = CSRTensor(indptr, indices, values, shape)
        csr_tensor_2 = CSRTensor(indptr_2, indices_2, values_2, shape)
        return csr_tensor_1 - csr_tensor_2

    neg_expect = CSRTensor(indptr, indices, Tensor([3.5, 2.5, 1.5, 0.5, -0.5, -1.5], mstype.float32), shape)
    neg_output = test_csr_neg(indptr, indices, values, shape)
    compare_csr(neg_output, neg_expect)
    neg_output = jit(test_csr_neg)(indptr, indices, values, shape)
    compare_csr(neg_output, neg_expect)

    add_expect = CSRTensor(Tensor([0, 2, 5, 7], mstype.int32), Tensor([2, 3, 0, 1, 2, 1, 3], mstype.int32),
                           Tensor([-2.5, -5, -3, -1.5, -0.5, 1, 1.5], mstype.float32), shape)
    add_output = test_csr_add(indptr, indptr_2, indices, indices_2, values, values_2, shape)
    compare_csr(add_output, add_expect)
    add_output = jit(test_csr_add)(indptr, indptr_2, indices, indices_2, values, values_2, shape)
    compare_csr(add_output, add_expect)

    sub_expect = CSRTensor(Tensor([0, 2, 5, 7], mstype.int32), Tensor([2, 3, 0, 1, 2, 1, 3], mstype.int32),
                           Tensor([2.5, -2, -2, -1.5, -0.5, 0, 1.5], mstype.float32), shape)
    sub_output = test_csr_sub(indptr, indptr_2, indices, indices_2, values, values_2, shape)
    compare_csr(sub_output, sub_expect)

    sub_output = jit(test_csr_sub)(indptr, indptr_2, indices, indices_2, values, values_2, shape)
    compare_csr(sub_output, sub_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_csr_add_dynamic_shape_methods():
    """
    Feature: Test csr add dynamic shape methods.
    Description: Test csr_add.
    Expectation: Success.
    """
    if get_platform() != "linux":
        return

    class Net(nn.Cell):
        def construct(self, x, y, z):
            return -x + y + z

    indptr = Tensor([0, 1, 2, 4, 5], dtype=mstype.int32)
    indices = Tensor([4, 4, 1, 2, 2], dtype=mstype.int32)
    shape = (4, 5)
    values = Tensor(np.arange(5) - 2.5, dtype=mstype.float32)

    def test_csr_add(indptr, indices, values, shape):
        x = CSRTensor(indptr, indices, values, shape)
        net = Net()
        return net(x, x, x)

    add_expect = CSRTensor(indptr, indices, Tensor(
        [-2.5, -1.5, -0.5, 0.5, 1.5], mstype.float32), shape)
    add_output = test_csr_add(indptr, indices, values, shape)
    compare_csr(add_output, add_expect)
    add_output = jit(test_csr_add)(indptr, indices, values, shape)
    compare_csr(add_output, add_expect)
