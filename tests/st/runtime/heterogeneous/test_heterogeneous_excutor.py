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

import numpy as np
from mindspore import nn, jit
from mindspore import ops
from mindspore import amp
from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    """
    Heterogeneous net cell.
    """
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


class MulRelu(Cell):
    """
    Heterogeneous mul relu cell.
    """
    def __init__(self):
        super().__init__()
        self.relu1 = ops.ReLU()
        self.relu2 = ops.ReLU()
        self.mul = ops.Mul()

    @jit
    def construct(self, inp1, inp2):
        x1 = self.relu1(inp1)
        x2 = self.relu2(inp2)
        y = self.mul(x1, x2)
        return y


@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level1", card_mark="onecard",
          essential_mark="essential", mem_peak=2048.0)
def test_heterogeneous_excutor():
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float64)
    label_data = np.random.randn(32, 10).astype(np.float32)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
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


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
def test_heterogeneous_default_cpu_prim_ascend_in_graph_mode():
    """
    Feature: KBK heterogeneous.
    Description: Default device target is CPU, the relu1 set to Ascend.
    Expectation: The output of device is equal to the output of heterogeneous.
    """
    context.set_context(device_target="CPU")
    net = MulRelu()
    inp1 = Tensor(np.random.randn(2, 2).astype(np.float32))
    inp2 = Tensor(np.random.randn(2, 2).astype(np.float32))
    output_device = net(inp1, inp2)
    net.relu1.set_device("Ascend")
    output_heter = net(inp1, inp2)
    assert np.allclose(output_device.asnumpy(), output_heter.asnumpy(), 1e-6, 1e-6)
