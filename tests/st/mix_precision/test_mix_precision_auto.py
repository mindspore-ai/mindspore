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
"""Test network turn on mix_precision with auto mode."""

import pytest
import numpy as np
import mindspore as ms
from mindspore.amp import auto_mixed_precision, build_train_network
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore import context


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
                              has_bias=False,
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


class Net_FP16(nn.Cell):

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
                              has_bias=False,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones').to_float(ms.float16)
        self.mean = ops.ReduceMean(keep_dims=False)
        self.cast = ops.Cast()

    def construct(self, x):
        x = self.cast(x, ms.float16)
        x = self.relu(x)
        x = self.cast(x, ms.float32)
        x = self.bn1(x)
        x = self.cast(x, ms.float16)
        x = self.conv(x)
        x = self.cast(x, ms.float32)
        x = self.bn2(x)
        x = self.cast(x, ms.float16)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.cast(x, ms.float32)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_infer_auto():
    """
    Feature: auto mixed precision auto mode.
    Description: test network infer result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    net_pynative = Net(3, 10)
    net_pynative = auto_mixed_precision(net_pynative, amp_level="auto", dtype=ms.float16)
    out_pynative = net_pynative(Tensor(input_data))

    # manual mixed precision
    net_pynative2 = Net_FP16(3, 10)
    out_pynative2 = net_pynative2(Tensor(input_data))

    assert np.allclose(out_pynative.asnumpy(), out_pynative2.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_train_auto():
    """
    Feature: auto mixed precision auto mode.
    Description: test network train result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    label_data = np.random.randn(32, 10).astype(np.float32)

    # auto mixed precision
    net_pynative = Net(3, 10)
    net_pynative = auto_mixed_precision(net_pynative, amp_level="auto", dtype=ms.float16)
    opt_pynative = nn.Momentum(params=net_pynative.trainable_params(),
                               learning_rate=0.001,
                               momentum=0.0009,
                               weight_decay=0.001,
                               loss_scale=0.0001)
    loss_pynative = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network_pynative = build_train_network(net_pynative,
                                                 opt_pynative,
                                                 loss_pynative,
                                                 level="O0",
                                                 loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss_pynative = train_network_pynative(Tensor(input_data), Tensor(label_data))

    # manual mixed precision
    net_pynative2 = Net_FP16(3, 10)
    opt_pynative2 = nn.Momentum(params=net_pynative2.trainable_params(),
                                learning_rate=0.001,
                                momentum=0.0009,
                                weight_decay=0.001,
                                loss_scale=0.0001)
    loss_pynative2 = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network_pynative2 = build_train_network(net_pynative2,
                                                  opt_pynative2,
                                                  loss_pynative2,
                                                  level="O0",
                                                  loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss_pynative2 = train_network_pynative2(Tensor(input_data), Tensor(label_data))

    assert np.allclose(loss_pynative.asnumpy(), loss_pynative2.asnumpy(), 0.0001, 0.0001)


def func_for_amp(x, in_c, out_c):
    """function for test amp in auto mode"""
    bn1 = nn.BatchNorm2d(num_features=in_c,
                         gamma_init='ones',
                         beta_init='zeros',
                         moving_mean_init='zeros',
                         moving_var_init='ones')
    bn2 = nn.BatchNorm2d(num_features=out_c,
                         gamma_init='ones',
                         beta_init='zeros',
                         moving_mean_init='zeros',
                         moving_var_init='ones')
    x = ops.relu(x)
    x = bn1(x)
    x = ops.conv2d(x, ops.ones((out_c, in_c, 3, 3), ms.float32), ops.ones((out_c), ms.float32), 1, 'same')
    x = bn2(x)
    x = ops.relu(x)
    x = ops.ReduceMean(keep_dims=False)(x, (2, 3))
    return x


def func_for_amp_fp16(x, in_c, out_c):
    """function for test amp in auto mode"""
    bn1 = nn.BatchNorm2d(num_features=in_c,
                         gamma_init='ones',
                         beta_init='zeros',
                         moving_mean_init='zeros',
                         moving_var_init='ones')
    bn2 = nn.BatchNorm2d(num_features=out_c,
                         gamma_init='ones',
                         beta_init='zeros',
                         moving_mean_init='zeros',
                         moving_var_init='ones')
    x = ops.cast(x, ms.float16)
    x = ops.relu(x)
    x = ops.cast(x, ms.float32)
    x = bn1(x)
    x = ops.cast(x, ms.float16)
    x = ops.conv2d(x, ops.ones((out_c, in_c, 3, 3), ms.float16), None, 1, 'same')
    x = ops.cast(x, ms.float32)
    x = ops.BiasAdd()(x, ops.ones((out_c), ms.float32))
    x = bn2(x)
    x = ops.cast(x, ms.float16)
    x = ops.relu(x)
    x = ops.ReduceMean(keep_dims=False)(x, (2, 3))
    x = ops.cast(x, ms.float32)
    return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_infer_func_auto():
    """
    Feature: auto mixed precision auto mode.
    Description: test function infer result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    func_pynative = auto_mixed_precision(func_for_amp, amp_level="auto", dtype=ms.float16)
    out_pynative = func_pynative(Tensor(input_data), 3, 10)

    # manual mixed precision
    out_pynative2 = func_for_amp_fp16(Tensor(input_data), 3, 10)

    assert np.allclose(out_pynative.asnumpy(), out_pynative2.asnumpy(), 0.0001, 0.0001)


class SubNet(nn.Cell):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_c,
                                 gamma_init='ones',
                                 beta_init='zeros',
                                 moving_mean_init='zeros',
                                 moving_var_init='ones')
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=False,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones')

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NetWithSubNet(nn.Cell):

    def __init__(self, sub_net, in_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=in_c,
                                 gamma_init='ones',
                                 beta_init='zeros',
                                 moving_mean_init='zeros',
                                 moving_var_init='ones')
        self.sub_net = sub_net
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.relu(x)
        x = self.bn(x)
        x = self.sub_net(x)
        x = self.mean(x, (2, 3))
        return x


class SubNet_FP16(nn.Cell):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_c,
                                 gamma_init='ones',
                                 beta_init='zeros',
                                 moving_mean_init='zeros',
                                 moving_var_init='ones')
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=False,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones').to_float(ms.float16)

    def construct(self, x):
        x = ops.cast(x, ms.float16)
        x = self.conv(x)
        x = ops.cast(x, ms.float32)
        x = self.bn(x)
        x = ops.cast(x, ms.float16)
        x = self.relu(x)
        x = ops.cast(x, ms.float32)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_infer_subnet_auto():
    """
    Feature: auto mixed precision auto mode.
    Description: test subnet infer result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    sub_net = SubNet(3, 10)
    sub_net = auto_mixed_precision(sub_net, amp_level="auto", dtype=ms.float16)
    net_pynative = NetWithSubNet(sub_net, 3)
    out_pynative = net_pynative(Tensor(input_data))

    # manual mixed precision
    sub_net_fp16 = SubNet_FP16(3, 10)
    net_pynative2 = NetWithSubNet(sub_net_fp16, 3)
    out_pynative2 = net_pynative2(Tensor(input_data))

    assert np.allclose(out_pynative.asnumpy(), out_pynative2.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_train_subnet_auto():
    """
    Feature: auto mixed precision auto mode.
    Description: test subnet train result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    label_data = np.random.randn(32, 10).astype(np.float32)

    # auto mixed precision
    sub_net = SubNet(3, 10)
    sub_net = auto_mixed_precision(sub_net, amp_level="auto", dtype=ms.float16)
    net_pynative = NetWithSubNet(sub_net, 3)
    opt_pynative = nn.Momentum(params=net_pynative.trainable_params(),
                               learning_rate=0.001,
                               momentum=0.0009,
                               weight_decay=0.001,
                               loss_scale=0.0001)
    loss_pynative = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network_pynative = build_train_network(net_pynative,
                                                 opt_pynative,
                                                 loss_pynative,
                                                 level="O0",
                                                 loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss_pynative = train_network_pynative(Tensor(input_data), Tensor(label_data))

    # manual mixed precision
    sub_net_fp16 = SubNet_FP16(3, 10)
    net_pynative2 = NetWithSubNet(sub_net_fp16, 3)
    opt_pynative2 = nn.Momentum(params=net_pynative2.trainable_params(),
                                learning_rate=0.001,
                                momentum=0.0009,
                                weight_decay=0.001,
                                loss_scale=0.0001)
    loss_pynative2 = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network_pynative2 = build_train_network(net_pynative2,
                                                  opt_pynative2,
                                                  loss_pynative2,
                                                  level="O0",
                                                  loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss_pynative2 = train_network_pynative2(Tensor(input_data), Tensor(label_data))

    assert np.allclose(loss_pynative.asnumpy(), loss_pynative2.asnumpy(), 0.0001, 0.0001)


class NetWithRecompute(nn.Cell):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_c,
                                 gamma_init='ones',
                                 beta_init='zeros',
                                 moving_mean_init='zeros',
                                 moving_var_init='ones')
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=False,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones')
        self.conv.recompute()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_recompute():
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using network with recompute.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float16)

    # net with amp should run success
    net = NetWithRecompute(3, 10)
    net = auto_mixed_precision(net, amp_level="auto", dtype=ms.float16)
    grad_net = ops.GradOperation()(net)
    grad_val = grad_net(Tensor(input_data))
    _ = grad_val.asnumpy()
