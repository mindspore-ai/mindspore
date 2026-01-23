# Copyright 2025 Huawei Technologies Co., Ltd
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

import numpy as np
from mindspore import context, nn, Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell
from tests.mark_utils import arg_mark


class EmbeddingLookUpNet(nn.Cell):
    def __init__(self, vocab_size, embedding_size, target='CPU'):
        super().__init__()
        self.embedding_loopup = nn.EmbeddingLookup(vocab_size=vocab_size,
                                                   embedding_size=embedding_size,
                                                   param_init="ones", target=target)

    def construct(self, indices):
        return self.embedding_loopup(indices)


def embedding_lookup_cpu():
    inputs = Tensor(np.array([0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 2]).astype(np.float32))
    net = EmbeddingLookUpNet(4, 2, target="CPU")

    criterion = SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=0.1)
    optimizer.sparse_opt.add_prim_attr("primitive_target", "CPU")
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    for _ in range(2):
        train_network(inputs, label)
    return net(inputs)


def embedding_lookup_ascend():
    inputs = Tensor(np.array([0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 2]).astype(np.float32))
    net = EmbeddingLookUpNet(4, 2, target="DEVICE")

    criterion = SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=0.1)
    optimizer.sparse_opt.add_prim_attr("primitive_target", "CPU")
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    for _ in range(2):
        train_network(inputs, label)
    return net(inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_heter_dvm():
    """
    Feature: Runtime special output.
    Description: Heter input for packet op.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O1"})
    out_cpu = embedding_lookup_cpu()
    out_ascend = embedding_lookup_ascend()
    assert np.allclose(out_cpu.asnumpy(), out_ascend.asnumpy())
