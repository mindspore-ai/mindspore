# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.nn import TrainOneStepCell, WithLossCell
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


class NetWithEmbeddingLookUp(nn.Cell):
    def __init__(self, vocab_size, embedding_size, target="CPU"):
        super(NetWithEmbeddingLookUp, self).__init__()
        self.embedding_lookup =  \
                        nn.EmbeddingLookup(vocab_size=vocab_size,
                                           embedding_size=embedding_size,
                                           param_init="ones", target=target)

    def construct(self, indices):
        out = self.embedding_lookup(indices)
        return out


def test_sit_embedding_lookup_net():
    """
    Feature: Dynamic shape.
    Description: Test dynamic shape ops.
    Expectation: No exception.
    """
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))
    label = Tensor(np.random.randn(3, 8).astype(np.float32))

    net1 = NetWithEmbeddingLookUp(vocab_size=8, embedding_size=8, target="CPU")
    loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    optimizer1 = nn.Adam(params=net1.trainable_params(), learning_rate=0.1)
    optimizer1.unique = True
    train_network1 = TrainOneStepCell(WithLossCell(net1, loss), optimizer1)
    train_network1.set_train()
    out1 = train_network1(indices, label)

    net2 = NetWithEmbeddingLookUp(vocab_size=8, embedding_size=8, target="CPU")
    optimizer2 = nn.Adam(params=net2.trainable_params(), learning_rate=0.1)
    optimizer2.unique = False
    optimizer2.target = "CPU"
    train_network2 = TrainOneStepCell(WithLossCell(net2, loss), optimizer2)
    train_network2.set_train()
    out2 = train_network2(indices, label)

    assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)
