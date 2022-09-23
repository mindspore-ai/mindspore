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
"""Test network turn on mix_precision and heterogeneous_excutor."""

import pytest
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import amp
from mindspore import Tensor
from mindspore import context
from mindspore.train.loss_scale_manager import FixedLossScaleManager


class Net(nn.Cell):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=in_c,
                                  gamma_init='ones',
                                  beta_init='zeros',
                                  moving_mean_init='zeros',
                                  moving_var_init='ones')
        self.bn2 = nn.BatchNorm2d(num_features=out_c,
                                  gamma_init='ones',
                                  beta_init='zeros',
                                  moving_mean_init='zeros',
                                  moving_var_init='ones')
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=True,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones')
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_heterogeneous_excutor():
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float64)
    label_data = np.random.randn(32, 10).astype(np.float32)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(3, 10)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001,
                      momentum=0.0009, weight_decay=0.001, loss_scale=0.0001)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network = amp.build_train_network(net, opt, loss, level="O3",
                                            loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    out = train_network(Tensor(input_data), Tensor(label_data))

    # heterogeneous_excutor
    net_heter = Net(3, 10)
    net_heter.relu.relu.set_device("CPU")
    net_heter.conv.conv2d.set_device("CPU")

    opt_heter = nn.Momentum(params=net_heter.trainable_params(),
                            learning_rate=0.001, momentum=0.0009,
                            weight_decay=0.001, loss_scale=0.0001)
    loss_heter = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network_heter = amp.build_train_network(net_heter, opt_heter, loss_heter, level="O3",
                                                  loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    out_heter = train_network_heter(Tensor(input_data), Tensor(label_data))
    assert np.allclose(out.asnumpy(), out_heter.asnumpy(), 0.001, 0.001)
