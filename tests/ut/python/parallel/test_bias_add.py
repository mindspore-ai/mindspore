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

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.train.model import Model


class CrossEntropyLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.reduce_mean = P.ReduceMean()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits()
        self.reduction = reduction

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)
        if self.reduction == 'mean':
            loss = self.reduce_mean(loss, (-1,))
        return loss


class DatasetLenet():
    def __init__(self, predict, label, length=3):
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, pad_mode='valid',
                              has_bias=True, weight_init='ones', bias_init='ones')
        self.conv.conv2d.shard(((8, 1, 1, 1), (1, 1, 1, 1)))
        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((1, 1, 1, 8),))
        self.flat = nn.Flatten()

    def construct(self, inputs):
        x = self.conv(inputs)
        x = self.reduce_mean(x, -1)
        x = self.flat(x)
        return x


def test_bias_add():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8)
    input_np = np.ones([16, 3, 32, 32]).astype(np.float32)
    label_np = np.zeros([16, 2048]).astype(np.float32)
    dataset = DatasetLenet(Tensor(input_np), Tensor(label_np), 1)
    net = Net()
    loss = CrossEntropyLoss()
    opt = nn.Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    model.train(epoch=1, train_dataset=dataset, dataset_sink_mode=False)
