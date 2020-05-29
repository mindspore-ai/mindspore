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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Dense, SoftmaxCrossEntropyWithLogits
from mindspore.nn import TrainOneStepCell, WithLossCell

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", impl_type="tbe")


class Adam:
    def __init__(self, batch_num, input_channels, output_channels, epoch, lr, weight_decay, epsilon):
        self.batch_num = batch_num
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.epsilon = epsilon

    def train_mindspore_impl(self):
        input_ = Tensor(np.random.randn(self.batch_num, self.input_channels).astype(np.float32))
        weight_np = Tensor(np.random.randn(self.output_channels, self.input_channels).astype(np.float32))
        bias = Tensor(np.random.randn(self.output_channels).astype(np.float32))

        label_np = np.random.randint(self.output_channels, size=self.batch_num)
        label_np_onehot = np.zeros(shape=(self.batch_num, self.output_channels)).astype(np.float32)
        label_np_onehot[np.arange(self.batch_num), label_np] = 1.0
        label = Tensor(label_np_onehot)

        ms_dense = Dense(in_channels=self.input_channels,
                         out_channels=self.output_channels,
                         weight_init=weight_np,
                         bias_init=bias, has_bias=True)
        criterion = SoftmaxCrossEntropyWithLogits()
        optimizer = nn.Adam(ms_dense.trainable_params(),
                            learning_rate=1e-3,
                            beta1=0.9, beta2=0.999, eps=self.epsilon,
                            use_locking=False,
                            use_nesterov=False, weight_decay=0.0,
                            loss_scale=1.0)

        net_with_criterion = WithLossCell(ms_dense, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()

        print('MS Initialized!')
        for _ in range(self.epoch):
            train_network(input_, label)
        output = ms_dense(input_)
        print("===============output=================", output)
        return output.asnumpy()


def test_adam():
    fact = Adam(batch_num=8, input_channels=20, output_channels=5, epoch=5, lr=0.1, weight_decay=0.0, epsilon=1e-8)
    fact.train_mindspore_impl()
