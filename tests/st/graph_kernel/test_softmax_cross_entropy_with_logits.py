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
from tests.mark_utils import arg_mark

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetSoftmaxCrossEntropyWithLogits(nn.Cell):
    def __init__(self):
        super(NetSoftmaxCrossEntropyWithLogits, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits()

    def construct(self, logits, labels):
        return self.loss(logits, labels)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softmax_cross_entropy_with_logits():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    logits = Tensor(np.array([[1, 1, 10],
                              [1, 10, 1],
                              [10, 1, 1]]).astype(np.float32))
    labels = Tensor(np.array([[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0]]).astype(np.float32))

    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    softmax_cross_entropy_with_logits = NetSoftmaxCrossEntropyWithLogits()
    result_open_gk = softmax_cross_entropy_with_logits(logits, labels)

    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=False)
    softmax_cross_entropy_with_logits_beta = NetSoftmaxCrossEntropyWithLogits()
    result_close_gk = softmax_cross_entropy_with_logits_beta(logits, labels)

    error0 = 1.0e-6
    diff0 = result_open_gk[0].asnumpy() - result_close_gk[0].asnumpy()
    diff1 = result_open_gk[1].asnumpy() - result_close_gk[1].asnumpy()
    assert np.all(abs(diff0) < error0)
    assert np.all(abs(diff1) < error0)
