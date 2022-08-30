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
"""common utils for sparse tests"""
import platform
from mindspore import Tensor, CSRTensor, COOTensor, context, ops
import mindspore.common.dtype as mstype


def get_platform():
    return platform.system().lower()


def compare_res(tensor_tup, numpy_tup):
    assert len(tensor_tup) == len(numpy_tup)
    for item in zip(tensor_tup, numpy_tup):
        assert (item[0].asnumpy() == item[1]).all()


def compare_csr(csr1, csr2):
    assert isinstance(csr1, CSRTensor)
    assert isinstance(csr2, CSRTensor)
    assert (csr1.indptr.asnumpy() == csr2.indptr.asnumpy()).all()
    assert (csr1.indices.asnumpy() == csr2.indices.asnumpy()).all()
    assert (csr1.values.asnumpy() == csr2.values.asnumpy()).all()
    assert csr1.shape == csr2.shape


def compare_coo(coo1, coo2):
    assert isinstance(coo1, COOTensor)
    assert isinstance(coo2, COOTensor)
    assert (coo1.indices.asnumpy() == coo2.indices.asnumpy()).all()
    assert (coo1.values.asnumpy() == coo2.values.asnumpy()).all()
    assert coo1.shape == coo2.shape


def get_csr_tensor():
    indptr = Tensor([0, 1, 2], dtype=mstype.int32)
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 4)
    return CSRTensor(indptr, indices, values, shape)


def get_csr_components():
    csr = get_csr_tensor()
    res = (csr.indptr, csr.indices, csr.values, csr.shape)
    return res


def get_csr_from_scalar(x):
    indptr = Tensor([0, 1, 1], dtype=mstype.int32)
    indices = Tensor([2], dtype=mstype.int32)
    shape = (2, 3)
    return CSRTensor(indptr, indices, x.reshape(1), shape)


def csr_add(csr_tensor, x):
    return CSRTensor(csr_tensor.indptr, csr_tensor.indices, csr_tensor.values + x, csr_tensor.shape)


def forward_grad_net(net, *inputs, mode=context.GRAPH_MODE):
    context.set_context(mode=mode)
    forward = net(*inputs)
    grad = ops.GradOperation(get_all=True)(net)(*inputs)
    return forward, grad
