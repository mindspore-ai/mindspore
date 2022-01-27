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

from mindspore import Tensor, CSRTensor, ms_function, nn, context
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
    def test_pynative():
        csr = CSRTensor(indptr, indices, values, shape)
        return csr.indptr, csr.indices, csr.values, csr.shape
    test_graph = ms_function(test_pynative)

    csr1_tuple = test_pynative()
    csr2_tuple = test_graph()

    csr1 = CSRTensor(*csr1_tuple)
    csr2 = CSRTensor(*csr2_tuple)
    compare_csr(csr1, csr2)


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
            super().__init__()
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
            super().__init__()
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
@pytest.mark.env_onecard
def test_csr_ops():
    """
    Feature: Test CSR-related Ops.
    Description: Test CSRReduceSum, CSRMul, CSRMV.
    Expectation: Success.
    """
    csr_reducesum = _csr_ops.CSRReduceSum()
    csrmv = _csr_ops.CSRMV()

    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([2, 1], dtype=mstype.float32)
    dense_shape = (2, 4)

    dense_tensor = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
    dense_vector = Tensor([[1.], [1], [1], [1]], dtype=mstype.float32)
    csr_tensor = CSRTensor(indptr, indices, values, dense_shape)

    def test_ops_pynative():
        dense1 = csr_reducesum(csr_tensor, 1)
        dense2 = csrmv(csr_tensor, dense_vector)
        sparse1 = csr_tensor * dense_tensor
        sparse2 = dense_tensor * csr_tensor
        return dense1, dense2, sparse1, sparse2

    test_ops_graph = ms_function(test_ops_pynative)

    pynative_res = test_ops_pynative()
    graph_res = test_ops_graph()
    expect1 = np.array([[2.], [1.]], dtype=np.float32)
    expect2 = np.array([[2.], [1.]], dtype=np.float32)
    expect3 = np.array([2., 1.], dtype=np.float32)
    assert np.allclose(pynative_res[0].asnumpy(), expect1)
    assert np.allclose(pynative_res[1].asnumpy(), expect2)
    assert np.allclose(pynative_res[2].values.asnumpy(), expect3)
    assert np.allclose(pynative_res[3].values.asnumpy(), expect3)
    assert np.allclose(graph_res[0].asnumpy(), expect1)
    assert np.allclose(graph_res[1].asnumpy(), expect2)
    assert np.allclose(graph_res[2].values.asnumpy(), expect3)
    assert np.allclose(graph_res[3].values.asnumpy(), expect3)


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
            super().__init__()
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
            super().__init__()
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

    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
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
        return F.dtype(x)
    graph_test = ms_function(pynative_test)

    out1 = pynative_test()
    out2 = graph_test()
    assert out1 in [mstype.float32]
    assert out2 in [mstype.float32]
