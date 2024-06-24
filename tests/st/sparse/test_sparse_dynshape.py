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
"""smoke tests for sparse dynamic shape operations"""
import numpy as np
from mindspore import Tensor, nn, CSRTensor, ops
from mindspore.common import dtype as mstype

from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.grad.test_grad_of_dynamic import TestDynamicGrad


class NetDenseToCSR(nn.Cell):

    def construct(self, x):
        csr = x.to_csr()
        return csr.indptr, csr.indices, csr.values


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_dense_to_csr():
    """
    Feature: Test tensor.to_csr in Graph and PyNative.
    Description: Test tensor.to_csr in dynamic rank and dynamic shape.
    Expectation: Success.
    """
    test_dynamic = TestDynamicGrad(NetDenseToCSR())
    x = Tensor(np.array([[2, 0, -1], [0, 0, 1]]), mstype.float32)
    test_dynamic.test_dynamic_grad_net((x))
    test_dynamic.test_dynamic_grad_net((x), is_dynamic_rank=True)


class NetDenseToCOO(nn.Cell):

    def construct(self, x):
        coo = x.to_coo()
        return coo.indices, coo.values


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_dense_to_coo():
    """
    Feature: Test tensor.to_coo in Graph and PyNative.
    Description: Test tensor.to_coo in dynamic rank and dynamic shape.
    Expectation: Success.
    """
    test_dynamic = TestDynamicGrad(NetDenseToCOO())
    x = Tensor(np.array([[2, 0, -1], [0, 0, 1]]), mstype.float32)
    test_dynamic.test_dynamic_grad_net((x))
    test_dynamic.test_dynamic_grad_net((x), is_dynamic_rank=True)


class NetCSRToCOO(nn.Cell):
    def construct(self, indptr, indices, values, shape):
        csr = CSRTensor(indptr, indices, values, shape)
        coo = ops.csr_to_coo(csr)
        return coo.indices, coo.values


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_to_coo():
    """
    Feature: Test ops.csr_to_coo in Graph and PyNative.
    Description: Test ops.csr_to_coo in dynamic rank and dynamic shape.
    Expectation: Success.
    """
    test_dynamic = TestDynamicGrad(NetCSRToCOO())
    x = Tensor(np.array([[2, 0, -1], [0, 0, 1]]), mstype.float32).to_csr()
    args = (x.indptr, x.indices, x.values, x.shape)
    test_dynamic.test_dynamic_grad_net(args)
    test_dynamic.test_dynamic_grad_net(args, is_dynamic_rank=True)


class NetCSRToDense(nn.Cell):

    def construct(self, indptr, indices, values, dense_shape):
        x = CSRTensor(indptr, indices, values, dense_shape)
        return x.to_dense()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_to_dense_dshape():
    """
    Feature: Test csr_tensor.to_dense in Graph and PyNative.
    Description: Test csr_tensor.to_dense in dynamic shape.
    Expectation: Success.
    """
    test_dynamic = TestDynamicGrad(NetCSRToDense())
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(1, 7), dtype=mstype.float32)
    dense_shape = (3, 4)
    x = (indptr, indices, values, dense_shape)
    test_dynamic.test_dynamic_grad_net(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_csr_to_dense_drank():
    """
    Feature: Test csr_tensor.to_dense in Graph and PyNative.
    Description: Test csr_tensor.to_dense in dynamic rank.
    Expectation: Success.
    """
    test_dynamic = TestDynamicGrad(NetCSRToDense())
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(1, 7), dtype=mstype.float32)
    dense_shape = (3, 4)
    x = (indptr, indices, values, dense_shape)
    test_dynamic.test_dynamic_grad_net(x, is_dynamic_rank=True)
