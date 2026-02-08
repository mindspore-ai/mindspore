# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.communication.management as D
from mindspore import context, Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell, Momentum


grad = C.GradOperation(get_all=True)


class Net(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(Net, self).__init__()
        self.layer_norm = nn.LayerNorm((hidden_size,))
        self.dense1 = nn.Dense(in_channels=hidden_size, out_channels=hidden_size).to_float(mstype.float16)
        self.dense2 = nn.Dense(in_channels=hidden_size, out_channels=hidden_size).to_float(mstype.float16)
        self.gelu = P.Gelu()

        self.layer_norm.layer_norm.shard(((dp * mp, 1), (1,), (1,)))
        self.dense1.matmul.shard(((dp, 1), (mp, 1)))
        self.dense1.bias_add.shard(((dp, mp), (mp,)))
        self.gelu.shard(((dp, mp),))
        self.dense2.matmul.shard(((dp, mp), (1, mp)), out_strategy=((dp * mp, 1),))
        self.dense2.bias_add.shard(((dp * mp, 1), (1,)))

    def construct(self, x):
        out = self.layer_norm(x)
        out = self.dense1(out)
        out = self.gelu(out)
        out = self.dense2(out)
        return out


def run_mc2():
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    context.set_context(ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_mc2.json"})
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 2, 4
    input_ = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)
    label_ = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    net = Net(seq_len, hidden_size, dp, mp)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net(input_, label_)

run_mc2()
