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
"""train_utils."""

import os
from mindspore import nn, Tensor
from mindspore.common.parameter import ParameterTuple


def train_wrap(net, loss_fn=None, optimizer=None, weights=None):
    """train_wrap"""
    if loss_fn is None:
        loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    loss_net = nn.WithLossCell(net, loss_fn)
    loss_net.set_train()
    if weights is None:
        weights = ParameterTuple(net.trainable_params())
    if optimizer is None:
        optimizer = nn.Adam(weights, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                            use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    return train_net


def save_t(t, file):
    x = t.asnumpy()
    x.tofile(file)


def train_and_save(name, net, net_train, x, l, epoch):
    """train_and_save"""
    net.set_train(True)
    for i in range(epoch):
        net_train(x, l)

    net.set_train(False)
    y = net(x)
    if isinstance(y, tuple):
        i = 1
        for t in y:
            with os.fdopen(name + "_output" + str(i) + ".bin", 'w') as f:
                for j in t.asnumpy().flatten():
                    f.write(str(j)+' ')
            i = i + 1
    else:
        y_name = name + "_output1.bin"
        save_t(y, y_name)


def save_inout(name, x, l, net, net_train, sparse=False, epoch=1):
    """save_inout"""
    x_name = name + "_input1.bin"
    if sparse:
        x_name = name + "_input2.bin"
    save_t(Tensor(x.asnumpy().transpose(0, 2, 3, 1)), x_name)

    l_name = name + "_input2.bin"
    if sparse:
        l_name = name + "_input1.bin"
    save_t(l, l_name)

    net.set_train(False)
    net(x)

    train_and_save(name, net, net_train, x, l, epoch)


def save_inout_transfer(name, x, l, net_bb, net, net_train, sparse=False, epoch=1):
    """save_inout_transfer"""
    x_name = name + "_input1.bin"
    if sparse:
        x_name = name + "_input2.bin"
    save_t(Tensor(x.asnumpy().transpose(0, 2, 3, 1)), x_name)

    l_name = name + "_input2.bin"
    if sparse:
        l_name = name + "_input1.bin"
    save_t(l, l_name)

    net_bb.set_train(False)
    x1 = net_bb(x)
    net.set_train(False)
    net(x1)

    train_and_save(name, net, net_train, x1, l, epoch)
