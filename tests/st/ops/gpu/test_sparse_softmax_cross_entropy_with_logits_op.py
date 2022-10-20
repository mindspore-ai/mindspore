# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore.ops.operations.nn_ops import SparseSoftmaxCrossEntropyWithLogits
from mindspore import Tensor


class NetSparseSoftmaxCrossEntropyWithLogits(nn.Cell):

    def __init__(self):
        super(NetSparseSoftmaxCrossEntropyWithLogits, self).__init__()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, logits, labels):
        return self.loss(logits, labels)


class Net(nn.Cell):

    def __init__(self, is_grad) -> None:
        super(Net, self).__init__()
        self.loss = SparseSoftmaxCrossEntropyWithLogits(is_grad=is_grad)

    def construct(self, logits, labels):
        return self.loss(logits, labels)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_sparse_softmax_cross_entropy_with_logits_dyn():
    """
    Feature: test SparseSoftmaxCrossEntropyWithLogits ops in gpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    for i in range(0, 2, 1):
        net = Net(is_grad=bool(i))
        logits_dyn = Tensor(shape=[None, 5], dtype=ms.float32)
        labels_dyn = Tensor(shape=[None], dtype=ms.int32)
        net.set_inputs(logits_dyn, labels_dyn)
        logits = Tensor(
            np.array([[2, 3, 1, 4, 5], [2, 1, 2, 4, 3]]).astype(np.float32))
        labels = Tensor(np.array([0, 1]).astype(np.int32))
        loss = net(logits, labels)
        expect_shape = () if i == 0 else (2, 5)
        assert loss.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_softmax_cross_entropy_with_logits():
    logits = Tensor(
        np.array([[1, 1, 10], [1, 10, 1], [10, 1, 1]]).astype(np.float32))
    labels = Tensor(np.array([2, 1, 0]).astype(np.int32))
    expect_loss = [0.00024673, 0.00024673, 0.00024673]

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    sparse_softmax_cross_entropy_with_logits = NetSparseSoftmaxCrossEntropyWithLogits(
    )
    output = sparse_softmax_cross_entropy_with_logits(logits, labels)
    error0 = 1.0e-6
    diff0 = output.asnumpy() - expect_loss
    assert np.all(abs(diff0) < error0)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    sparse_softmax_cross_entropy_with_logits = NetSparseSoftmaxCrossEntropyWithLogits(
    )
    output = sparse_softmax_cross_entropy_with_logits(logits, labels)
    error0 = 1.0e-6
    diff0 = output.asnumpy() - expect_loss
    assert np.all(abs(diff0) < error0)
