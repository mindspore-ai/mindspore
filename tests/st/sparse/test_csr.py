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
from mindspore import Tensor, CSRTensor, ms_function
from mindspore.common import dtype as mstype


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
