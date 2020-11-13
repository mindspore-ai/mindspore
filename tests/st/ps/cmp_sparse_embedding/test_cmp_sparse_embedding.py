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

import os
import sys
import argparse
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam
from mindspore.common import set_seed
from mindspore.ops import operations as P
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_worker

parser = argparse.ArgumentParser(description="test_sparse_embedding")
parser.add_argument("--device_target", type=str, default="Ascend")
args, _ = parser.parse_known_args()
device_target = args.device_target
context.set_context(
    mode=context.GRAPH_MODE, device_target=device_target, enable_sparse=True
)
context.set_ps_context(enable_ps=True)


class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.cast = P.Cast()
        self.flatten = nn.Flatten()
        self.embedding = nn.EmbeddingLookup(16, 4)
        self.relu = nn.ReLU()
        self.fc = nn.Dense(12, num_class)

    def construct(self, x):
        x = self.cast(x, mstype.int32)
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def do_sparse_embedding(ps=False):
    epoch = 10
    net = LeNet5(10)
    if ps:
        net.embedding.embedding_table.set_param_ps()

    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()))
    optimizer.target = 'CPU'
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    losses = []
    for _ in range(epoch):
        data = Tensor(np.random.randint(-5, 15, (32, 3), np.int32))
        label = Tensor(np.random.randint(0, 9, (32), np.int32))
        if _is_role_pserver():
            train_network(data, label)
            sys.exit()
        else:
            loss = train_network(data, label).asnumpy()
            losses.append(loss)
    print(losses)
    return losses


envs = os.environ
if __name__ == "__main__":
    set_seed(0)
    ps_loss = do_sparse_embedding(True)

    if _is_role_worker():
        context.reset_ps_context()
        set_seed(0)
        no_ps_loss = do_sparse_embedding()
        context.set_ps_context(enable_ps=True)

    assert np.allclose(ps_loss, no_ps_loss, rtol=1.0e-6, atol=1.0e-6)
