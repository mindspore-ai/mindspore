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

from mindspore import Tensor, COOTensor, ms_function, nn, context
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F


context.set_context(mode=context.GRAPH_MODE)


def compare_coo(coo1, coo2):
    assert isinstance(coo1, COOTensor)
    assert isinstance(coo2, COOTensor)
    assert (coo1.indices.asnumpy() == coo2.indices.asnumpy()).all()
    assert (coo1.values.asnumpy() == coo2.values.asnumpy()).all()
    assert coo1.shape == coo2.shape


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
    test_graph = ms_function(test_pynative)

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

        @ms_function
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
def test_dtype_coo_tensor():
    """
    Feature: Test F.dtype with COOTensor.
    Description: Test: F.dtype(x), x.dtype.
    Expectation: Success.
    """
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (3, 4)

    def pynative_test():
        x = COOTensor(indices, values, shape)
        return F.dtype(x), x.dtype
    graph_test = ms_function(pynative_test)

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

    test_graph_1 = ms_function(test_pynative_1)
    test_graph_2 = ms_function(test_pynative_2)
    test_graph_3 = ms_function(test_pynative_3)

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
