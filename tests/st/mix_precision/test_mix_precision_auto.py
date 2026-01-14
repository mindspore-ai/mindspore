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
from mindspore.train.amp import auto_mixed_precision, build_train_network, _OutputTo32
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore import context
from mindspore.ops import auto_generate as gen
from mindspore.common.parameter import Parameter
from tests.mark_utils import arg_mark

context.set_context(jit_level='O0')


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
        x = self.relu(x)
        x = self.cast(x, ms.float32)
        x = self.bn1(x)
        x = self.cast(x, ms.float16)
        x = self.conv(x)
        x = self.cast(x, ms.float32)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_infer_auto(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test network infer result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    net = Net(3, 10)
    net = auto_mixed_precision(net, amp_level="auto", dtype=ms.float16)
    out = net(Tensor(input_data))

    # manual mixed precision
    net2 = Net_FP16(3, 10)
    out2 = net2(Tensor(input_data))

    assert np.allclose(out.asnumpy(), out2.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_train_auto(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test network train result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    label_data = np.random.randn(32, 10).astype(np.float32)

    # auto mixed precision
    net = Net(3, 10)
    opt = nn.Momentum(params=net.trainable_params(),
                      learning_rate=0.001,
                      momentum=0.0009,
                      weight_decay=0.001,
                      loss_scale=0.0001)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network = build_train_network(net,
                                        opt,
                                        loss,
                                        level="auto",
                                        loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss = train_network(Tensor(input_data), Tensor(label_data))

    # manual mixed precision
    net2 = Net_FP16(3, 10)
    net2 = _OutputTo32(net2)
    opt2 = nn.Momentum(params=net2.trainable_params(),
                       learning_rate=0.001,
                       momentum=0.0009,
                       weight_decay=0.001,
                       loss_scale=0.0001)
    loss2 = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network2 = build_train_network(net2,
                                         opt2,
                                         loss2,
                                         level="O0",
                                         loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss2 = train_network2(Tensor(input_data), Tensor(label_data))

    assert np.allclose(loss.asnumpy(), loss2.asnumpy(), 0.0001, 0.0001)


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
    x = ops.relu(x)
    x = ops.ReduceMean(keep_dims=False)(x, (2, 3))
    return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_infer_func_auto(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test function infer result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=mode, jit_config={"jit_level": "O1"})
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    func = auto_mixed_precision(func_for_amp, amp_level="auto", dtype=ms.float16)
    out = func(Tensor(input_data), 3, 10)

    # manual mixed precision
    out2 = func_for_amp_fp16(Tensor(input_data), 3, 10)

    assert np.allclose(out.asnumpy(), out2.asnumpy(), 0.0001, 0.0001)


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
        x = self.relu(x)
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_infer_subnet_auto(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test subnet infer result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    sub_net = SubNet(3, 10)
    sub_net = auto_mixed_precision(sub_net, amp_level="auto", dtype=ms.float16)
    net = NetWithSubNet(sub_net, 3)
    out = net(Tensor(input_data))

    # manual mixed precision
    sub_net_fp16 = SubNet_FP16(3, 10)
    net2 = NetWithSubNet(sub_net_fp16, 3)
    out2 = net2(Tensor(input_data))

    assert np.allclose(out.asnumpy(), out2.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_train_subnet_auto(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test subnet train result of amp auto mode compared with manual mixed precision.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    label_data = np.random.randn(32, 10).astype(np.float32)

    # auto mixed precision
    sub_net = SubNet(3, 10)
    sub_net = auto_mixed_precision(sub_net, amp_level="auto", dtype=ms.float16)
    net = NetWithSubNet(sub_net, 3)
    net = _OutputTo32(net)
    opt = nn.Momentum(params=net.trainable_params(),
                      learning_rate=0.001,
                      momentum=0.0009,
                      weight_decay=0.001,
                      loss_scale=0.0001)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network = build_train_network(net,
                                        opt,
                                        loss,
                                        level="O0",
                                        loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss = train_network(Tensor(input_data), Tensor(label_data))

    # manual mixed precision
    sub_net_fp16 = SubNet_FP16(3, 10)
    net2 = NetWithSubNet(sub_net_fp16, 3)
    net2 = _OutputTo32(net2)
    opt2 = nn.Momentum(params=net2.trainable_params(),
                       learning_rate=0.001,
                       momentum=0.0009,
                       weight_decay=0.001,
                       loss_scale=0.0001)
    loss2 = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network2 = build_train_network(net2,
                                         opt2,
                                         loss2,
                                         level="O0",
                                         loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    loss2 = train_network2(Tensor(input_data), Tensor(label_data))

    assert np.allclose(loss.asnumpy(), loss2.asnumpy(), 0.0001, 0.0001)


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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_recompute(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using network with recompute.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float16)

    # net with amp should run success
    net = NetWithRecompute(3, 10)
    net = auto_mixed_precision(net, amp_level="auto", dtype=ms.float16)
    grad_net = ops.GradOperation()(net)
    grad_val = grad_net(Tensor(input_data))
    _ = grad_val.asnumpy()


class NetWithToFloat(nn.Cell):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=False,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones')
        self.conv.to_float(ms.float32)

    def construct(self, x):
        x = self.conv(x)
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_with_to_float(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using network with to_float.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float16)
    input_data = Tensor(input_data)

    # net with amp should run success
    net = NetWithToFloat(3, 10)
    net = auto_mixed_precision(net, amp_level="auto", dtype=ms.float16)
    out = net(input_data)
    assert out.dtype == ms.float32


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_with_keyword_arguments(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using network with keyword arguments.
    Expectation: success.
    """
    class SoftmaxNet(nn.Cell):
        def __init__(self, weight_shape, input_type_np=np.float32):
            super().__init__()
            self.weight = Parameter(Tensor(np.random.randn(*weight_shape).astype(input_type_np)),
                                    name="weight")
            self.matmul = ops.MatMul()
            self.softmax = ops.Softmax(axis=1)

        def construct(self, x):
            out1 = self.matmul(x, self.weight)
            out2 = self.softmax(out1)
            return out1, out2

    context.set_context(mode=mode)
    net = SoftmaxNet((16, 16))
    net = auto_mixed_precision(net, amp_level="auto", dtype=ms.float16)
    input_x = Tensor(np.random.randn(16, 16).astype(np.float32))
    out1, out2 = net(input_x)
    assert out1.dtype == ms.float16 and out2.dtype == ms.float32


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
def test_auto_mix_precision_with_AddN(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode with AddN.
    Expectation: success.
    """
    class AddnNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.addn = ops.AddN()

        def construct(self, *z):
            return self.addn(z)

    context.set_context(mode=mode)
    x = Tensor(np.array([1, 2, 3]), ms.float16)
    y = Tensor(np.array([4, 5, 6]), ms.float32)
    net = AddnNet()
    net = auto_mixed_precision(net, amp_level="auto")
    out = net(x, y, x, y)
    assert out.dtype == ms.float32


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_amp_single_func():
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using single function.
    Expectation: success.
    """
    def func(x, y):
        return ops.matmul(x, y)

    x = ms.Tensor(np.ones([1, 2]), ms.float32)
    y = ms.Tensor(np.ones([2,]), ms.float32)
    amp_func = auto_mixed_precision(func, amp_level="auto", dtype=ms.float16)
    # Pynative mode
    assert amp_func(x, y).dtype == ms.float16
    # Graph mode
    assert ms.jit(amp_func)(x, y).dtype == ms.float16


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_amp_inner_func():
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using nested function.
    Expectation: success.
    """
    def inner_func(x, y):
        return ops.matmul(x, y)

    inner_func = auto_mixed_precision(inner_func, amp_level="auto", dtype=ms.float16)

    def func(x, y):
        return inner_func(x, y) + 1

    x = ms.Tensor(np.ones([1, 2]), ms.float32)
    y = ms.Tensor(np.ones([2,]), ms.float32)
    # Pynative mode
    assert func(x, y).dtype == ms.float16
    # Graph mode
    assert ms.jit(func)(x, y).dtype == ms.float16


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_amp_nested_func():
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using nested function.
    Expectation: success.
    """
    def inner_func(x, y):
        return ops.matmul(x, y)

    def func1(x, y):
        return inner_func(x, y)

    func1 = auto_mixed_precision(func1, amp_level="auto", dtype=ms.float16)

    def func2(x, y):
        return inner_func(x, y)

    def func(x, y):
        return func1(x, y), func2(x, y)

    x = ms.Tensor(np.ones([1, 2]), ms.float32)
    y = ms.Tensor(np.ones([2,]), ms.float32)
    # Pynative mode
    out_pynative = func(x, y)
    assert out_pynative[0].dtype == ms.float16 and out_pynative[1].dtype == ms.float32
    # Graph mode
    out_graph = ms.jit(func)(x, y)
    assert out_graph[0].dtype == ms.float16 and out_graph[1].dtype == ms.float32


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_amp_jit_func():
    """
    Feature: auto mixed precision auto mode.
    Description: test amp auto mode using jit function.
    Expectation: success.
    """
    @ms.jit
    def graph_func(x, y):
        return ops.matmul(x, y)

    def pynative_func(x, y):
        return ops.matmul(x, y)

    def func(x, y):
        return graph_func(x, y), pynative_func(x, y)

    func = auto_mixed_precision(func, amp_level="auto", dtype=ms.float16)
    x = ms.Tensor(np.ones([1, 2]), ms.float32)
    y = ms.Tensor(np.ones([2,]), ms.float32)
    out_graph, out_pynative = func(x, y)
    assert out_graph.dtype == ms.float16 and out_pynative.dtype == ms.float16


class NormNet(nn.Cell):

    def __init__(self):
        super().__init__()
        self.norm = gen.Norm()

    def construct(self, x):
        return self.norm(x)


def func_norm(x):
    return gen.Norm()(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_amp_auto_black_list_norm():
    """
    Feature: auto mixed precision auto mode.
    Description: test if prim in black list(Norm) can run in fp16.
    Expectation: success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    input_data = Tensor(np.ones([1, 1]), dtype=ms.float16)
    # test with net
    net = NormNet()
    net = auto_mixed_precision(net, "auto")
    out = net(input_data)
    assert out.dtype == ms.float32
    # test with func
    net_func = auto_mixed_precision(func_norm, "auto")
    out = net_func(input_data)
    assert out.dtype == ms.float32
