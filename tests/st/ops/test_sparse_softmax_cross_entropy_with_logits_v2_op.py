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

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations.nn_ops import SparseSoftmaxCrossEntropyWithLogitsV2
from tests.mark_utils import arg_mark


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = SparseSoftmaxCrossEntropyWithLogitsV2()

    def construct(self, logits, labels):
        return self.op(logits, labels)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_sparse_softmax_cross_entropy_with_logits_v2_dyn():
    """
    Feature: test SparseSoftmaxCrossEntropyWithLogitsV2 ops.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()

    logits_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    labels_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(logits_dyn, labels_dyn)

    logits = Tensor([[2, 3, 1, 4, 5], [2, 1, 2, 4, 3]], dtype=ms.float32)
    labels = Tensor([0, 1], dtype=ms.int32)
    loss, back_prop = net(logits, labels)

    expect_loss_shape = (2,)
    expect_back_prop_shape = (2, 5)
    assert loss.asnumpy().shape == expect_loss_shape
    assert back_prop.asnumpy().shape == expect_back_prop_shape
