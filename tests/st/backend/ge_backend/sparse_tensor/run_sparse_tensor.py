# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor, COOTensor, CSRTensor, nn, ops

class GradOfAllInputs(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, *inputs):
        grad_net = self.grad_op(self.net)
        return grad_net(*inputs)


class COOTensorNet(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def construct(self, indices, values):
        x = COOTensor(indices, values, self.shape)
        out = x.abs()
        return out


def coo_data_generate(values_shape):
    x = np.random.randint(2, 1024, size=(1,)).astype(np.int32)
    y = np.random.randint(np.ceil(values_shape / x).astype(np.int32), 1024, size=(1,)).astype(np.int32)
    shape = (int(x[0]), int(y[0]))
    row = np.random.randint(0, shape[0], size=(values_shape,)).astype(np.int32)
    now = np.random.randint(0, shape[1], size=(values_shape,)).astype(np.int32)
    indices = np.stack((row, now), axis=1)
    return shape, row, now, indices


def test_coo_tensor():
    """
    Feature: COOTensor implement
    Description: test COOTensor with ge backend.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    # construct input data: indices, values, shape
    values_shape = 1024
    shape, _, _, indices = coo_data_generate(values_shape)
    values = np.random.randn(values_shape,).astype(np.float32)

    net = COOTensorNet(shape)
    out = net(Tensor(indices), Tensor(values))
    assert np.allclose(out.values.asnumpy(), np.abs(values), 1.0e-4, 1.0e-4)
    net_grad = GradOfAllInputs(net)
    grad = net_grad(Tensor(indices), Tensor(values))
    assert len(grad) == 2


class CSRTensorNet(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def construct(self, indptr, indices, values):
        x = CSRTensor(indptr, indices, values, self.shape)
        out = x.abs()
        return out


def csr_data_generate(values_shape):
    x = np.random.randint(2, 1024, size=(1,)).astype(np.int32)
    y = np.random.randint(np.ceil(values_shape / x).astype(np.int32), 1024, size=(1,)).astype(np.int32)
    shape = (int(x[0]), int(y[0]))
    indptr = np.sort(np.random.choice(values_shape, x[0] + 1, replace=False)).astype(np.int32)
    if indptr[0] != np.array([0]).astype(np.int32):
        indptr[0] = np.array([0]).astype(np.int32)
    if indptr[-1] != np.array([0]).astype(np.int32):
        indptr[-1] = np.array([0]).astype(np.int32)
    indices = np.random.randint(0, y[0], size=(values_shape,)).astype(np.int32)
    return shape, indptr, indices


def test_csr_tensor():
    """
    Feature: CSRTensor implement
    Description: test CSRTensor with ge backend.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    # construct input data: indices, values, shape
    values_shape = 1024
    shape, indptr, indices = csr_data_generate(values_shape)
    values = np.random.randn(values_shape,).astype(np.float32)

    net = CSRTensorNet(shape)
    out = net(Tensor(indptr), Tensor(indices), Tensor(values))
    assert np.allclose(out.values.asnumpy(), np.abs(values), 1.0e-4, 1.0e-4)
    net_grad = GradOfAllInputs(net)
    grad = net_grad(Tensor(indptr), Tensor(indices), Tensor(values))
    assert len(grad) == 3


if __name__ == "__main__":
    test_coo_tensor()
    test_csr_tensor()
