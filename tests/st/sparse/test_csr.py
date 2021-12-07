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

import pytest
import numpy as np

from mindspore import Tensor, CSRTensor, ms_function, nn, context
from mindspore.ops.operations import _csr_ops
from mindspore.common import dtype as mstype

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


@pytest.mark.level1
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
    class CSRReduceSumNet(nn.Cell):
        def __init__(self):
            super(CSRReduceSumNet, self).__init__()
            self.op = _csr_ops.CSRReduceSum()

        def construct(self, indptr, indices, values, dense_shape, axis):
            csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
            return self.op(csr_tensor, axis)

    class CSRMulNet(nn.Cell):
        def __init__(self):
            super(CSRMulNet, self).__init__()
            self.op = _csr_ops.CSRMul()

        def construct(self, indptr, indices, values, dense_shape, dense):
            csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
            return self.op(csr_tensor, dense)

    class CSRMVNet(nn.Cell):
        def __init__(self):
            super(CSRMVNet, self).__init__()
            self.op = _csr_ops.CSRMV()

        def construct(self, indptr, indices, values, dense_shape, dense):
            csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
            return self.op(csr_tensor, dense)

    indptr = Tensor([0, 1, 2])
    indices = Tensor([0, 1])
    values = Tensor([2, 1], dtype=mstype.float32)
    dense_shape = (2, 4)
    dense_tensor = Tensor([[1., 1, 1, 1], [1, 1, 1, 1]], dtype=mstype.float32)
    dense_vector = Tensor([[1.], [1], [1], [1]], dtype=mstype.float32)

    net1 = CSRReduceSumNet()
    out1 = net1(indptr, indices, values, dense_shape, 1)
    expect1 = np.array([[2.], [1.]], dtype=np.float32)
    assert np.allclose(out1.asnumpy(), expect1)

    net2 = CSRMulNet()
    out2 = net2(indptr, indices, values, dense_shape, dense_tensor)
    expect2 = np.array([2., 1.], dtype=np.float32)
    assert np.allclose(out2.asnumpy(), expect2)

    net3 = CSRMVNet()
    out3 = net3(indptr, indices, values, dense_shape, dense_vector)
    expect3 = np.array([[2.], [1.]], dtype=np.float32)
    assert np.allclose(out3.asnumpy(), expect3)
