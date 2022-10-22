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

import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SparseToDenseNet(nn.Cell):

    def __init__(self):
        super(SparseToDenseNet, self).__init__()
        self.sparse_to_dense = P.SparseToDense()

    def construct(self, indices, values, sparse_shape):
        return self.sparse_to_dense(indices, values, sparse_shape)


class GradNet(nn.Cell):

    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, indices, values, sparse_shape, sense):
        return self.grad(self.network)(indices, values, sparse_shape, sense)


def judge_result_correct(result, expect):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)


def sparse_to_dense_int(i_type, v_type):
    indices = np.array([[0, 1], [1, 2]]).astype(i_type)
    values = np.array([7, 8]).astype(v_type)
    sparse_shape = (3, 3)
    forward_net = SparseToDenseNet()
    forward_output = forward_net(Tensor(indices), Tensor(values), sparse_shape)
    expect_forward_output = np.array([[0, 7, 0], [0, 0, 8],
                                      [0, 0, 0]]).astype(v_type)
    judge_result_correct(forward_output.asnumpy(), expect_forward_output)

    grad_net = GradNet(forward_net)
    sense = Tensor(np.arange(9).reshape((3, 3)).astype(v_type))
    grad_output = grad_net(Tensor(indices), Tensor(values), sparse_shape,
                           sense)
    expect_grad_output = np.array([1, 5]).astype(v_type)
    judge_result_correct(grad_output[1].asnumpy(), expect_grad_output)


def sparse_to_dense_float(i_type, v_type):
    indices = np.array([[0, 1, 0], [1, 2, 1], [2, 3, 2], [0, 2,
                                                          3]]).astype(i_type)
    values = np.array([6.5, 7.5, 9.5, 10.5]).astype(v_type)
    sparse_shape = (3, 4, 4)
    forward_net = SparseToDenseNet()
    forward_output = forward_net(Tensor(indices), Tensor(values), sparse_shape)
    expect_forward_output = np.array([[[0, 0, 0, 0], [6.5, 0, 0, 0],
                                       [0, 0, 0, 10.5], [0, 0, 0, 0]],
                                      [[0, 0, 0, 0], [0, 0, 0, 0],
                                       [0, 7.5, 0, 0], [0, 0, 0, 0]],
                                      [[0, 0, 0, 0], [0, 0, 0,
                                                      0], [0, 0, 0, 0],
                                       [0, 0, 9.5, 0]]]).astype(v_type)
    judge_result_correct(forward_output.asnumpy(), expect_forward_output)

    grad_net = GradNet(forward_net)
    sense = Tensor(np.arange(48).reshape((3, 4, 4)).astype(v_type) + 0.8)
    grad_output = grad_net(Tensor(indices), Tensor(values), sparse_shape,
                           sense)
    expect_grad_output = np.array([4.8, 25.8, 46.8, 11.8]).astype(v_type)
    judge_result_correct(grad_output[1].asnumpy(), expect_grad_output)


def sparse_to_dense_1D(i_type, v_type):
    indices = np.array([[8], [2], [6], [4]]).astype(i_type)
    values = np.array([6.5, 7.5, 9.5, 10.5]).astype(v_type)
    sparse_shape = (10,)
    forward_net = SparseToDenseNet()
    forward_output = forward_net(Tensor(indices), Tensor(values), sparse_shape)
    expect_forward_output = np.array([0, 0, 7.5, 0, 10.5, 0, 9.5, 0, 6.5,
                                      0]).astype(v_type)
    judge_result_correct(forward_output.asnumpy(), expect_forward_output)

    grad_net = GradNet(forward_net)
    sense = Tensor(np.arange(10).astype(v_type) + 0.8)
    grad_output = grad_net(Tensor(indices), Tensor(values), sparse_shape,
                           sense)
    expect_grad_output = np.array([8.8, 2.8, 6.8, 4.8]).astype(v_type)
    judge_result_correct(grad_output[1].asnumpy(), expect_grad_output)


indices_types = (np.int32, np.int64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_to_dense_dyn():
    """
    Feature: test SparseToDense ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = SparseToDenseNet()

    indices_dyn = Tensor(shape=[None, 2], dtype=ms.int32)
    values_dyn = Tensor(shape=[None], dtype=ms.float32)
    sparse_shape = (3, 4)
    net.set_inputs(indices_dyn, values_dyn, sparse_shape)

    indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
    values = Tensor([1, 2], dtype=ms.float32)
    out = net(indices, values, sparse_shape)
    print(out)

    expect_shape = (3, 4)
    assert out.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_to_dense_int():
    values_types = (np.bool_, np.uint8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64)
    for i_type in indices_types:
        for v_type in values_types:
            sparse_to_dense_int(i_type, v_type)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_to_dense_float():
    values_types = (np.float16, np.float32, np.float64)
    for i_type in indices_types:
        for v_type in values_types:
            sparse_to_dense_float(i_type, v_type)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_to_dense_1D():
    values_types = (np.float16, np.float32, np.float64)
    for i_type in indices_types:
        for v_type in values_types:
            sparse_to_dense_1D(i_type, v_type)
