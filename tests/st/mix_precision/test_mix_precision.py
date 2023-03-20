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
"""Test network turn on mix_precision."""

import os
import re
import pytest
import numpy as np
from mindspore.common import dtype
from mindspore import nn
from mindspore import ops
from mindspore import amp
from mindspore import Tensor
from mindspore import context
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train import Model
from utils import FakeData
from utils import allclose_nparray
from utils import FakeDataInitMode
from utils import find_newest_validateir_file
from utils import clean_all_ir_files
from tests.security_utils import security_off_wrap

def read_validateir_file(path_folder):
    filename = find_newest_validateir_file(path_folder)
    with open(os.path.join(filename), 'r') as f:
        contend = f.read()
    return contend


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
def test_sit_auto_mix_precision_train_o3():
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float64)
    label_data = np.random.randn(32, 10).astype(np.float32)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(3, 10)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.0009, weight_decay=0.001,
                      loss_scale=0.0001)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network = amp.build_train_network(net, opt, loss, level="O3",
                                            loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False))
    out = train_network(Tensor(input_data), Tensor(label_data))

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = Net(3, 10)
    opt_pynative = nn.Momentum(params=net_pynative.trainable_params(), learning_rate=0.001, momentum=0.0009,
                               weight_decay=0.001,
                               loss_scale=0.0001)
    loss_pynative = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    train_network_pynative = amp.build_train_network(net_pynative, opt_pynative, loss_pynative, level="O3",
                                                     loss_scale_manager=FixedLossScaleManager(
                                                         drop_overflow_update=False))
    out_pynative = train_network_pynative(Tensor(input_data), Tensor(label_data))
    assert np.allclose(out.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_sit_auto_mix_precision_model_o0():
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    dataset1 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    dataset1.set_label_data_type(np.float16)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(save_graphs=3, save_graphs_path='./test_amp_o0')
    net = Net(3, 10)
    net.to_float(dtype.float16)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model = Model(net, loss, opt, amp_level="O0")
    model.train(1, dataset1, dataset_sink_mode=False)
    contend = read_validateir_file('./test_amp_o0/')
    castnum = re.findall(r"Cast\(", contend)
    assert len(castnum) == 5
    clean_all_ir_files('./test_amp_o0')
    model.predict(Tensor(input_data))
    contend = read_validateir_file('./test_amp_o0/')
    castnum = re.findall(r"Cast\(", contend)
    assert len(castnum) == 11
    clean_all_ir_files('./test_amp_o0/')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_sit_auto_mix_precision_model_o2():
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    dataset1 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    dataset2 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(save_graphs=3, save_graphs_path='./test_amp_o2')
    net = Net(3, 10)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model = Model(net, loss, opt, amp_level="O2")
    model.train(1, dataset1, dataset_sink_mode=False)
    contend = read_validateir_file('./test_amp_o2/')
    castnum = re.findall(r"Cast\(", contend)
    assert len(castnum) == 14
    clean_all_ir_files('./test_amp_o2/')
    out_graph = model.predict(Tensor(input_data))

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = Net(3, 10)
    opt_pynative = nn.Momentum(params=net_pynative.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss_pynative = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model_pynative = Model(net_pynative, loss_pynative, opt_pynative, amp_level="O2")
    model_pynative.train(1, dataset2, dataset_sink_mode=False)
    out_pynative = model_pynative.predict(Tensor(input_data))
    allclose_nparray(out_graph.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_sit_auto_mix_precision_model_o1():
    """
    Feature: Test the O1 level auto mixed precision
    Description: input O1 level to Model interface
    Expectation: success.
    """
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    dataset1 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    dataset2 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(save_graphs=3, save_graphs_path='./test_amp_o1')
    net = Net(3, 10)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model = Model(net, loss, opt, amp_level="O1")
    model.train(1, dataset1, dataset_sink_mode=False)
    clean_all_ir_files('./test_amp_o1/')
    out_graph = model.predict(Tensor(input_data))

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = Net(3, 10)
    opt_pynative = nn.Momentum(params=net_pynative.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss_pynative = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model_pynative = Model(net_pynative, loss_pynative, opt_pynative, amp_level="O1")
    model_pynative.train(1, dataset2, dataset_sink_mode=False)
    out_pynative = model_pynative.predict(Tensor(input_data))
    allclose_nparray(out_graph.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_custom_mix_precision():
    """
    Feature: Test custom mixed precision
    Description: Test custom mixed precision
    Expectation: success.
    """
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)
    dataset1 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    dataset2 = FakeData(size=32,
                        batch_size=32,
                        image_size=(3, 224, 224),
                        num_classes=10,
                        fakedata_mode=FakeDataInitMode.OnesInit)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(3, 10)
    white_list = amp.get_white_list()
    white_list.clear()
    white_list.append(nn.ReLU)
    white_list.append(nn.Conv2d)
    net = amp.custom_mixed_precision(net, white_list=white_list)

    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model = Model(net, loss, opt, amp_level="O0")
    model.train(1, dataset1, dataset_sink_mode=False)
    out_graph = model.predict(Tensor(input_data))

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = Net(3, 10)
    white_list = amp.get_white_list()
    white_list.clear()
    white_list.append(nn.ReLU)
    white_list.append(nn.Conv2d)
    net_pynative = amp.custom_mixed_precision(net_pynative, white_list=white_list)
    opt_pynative = nn.Momentum(params=net_pynative.trainable_params(), learning_rate=0.001, momentum=0.0009)
    loss_pynative = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    model_pynative = Model(net_pynative, loss_pynative, opt_pynative, amp_level="O0")
    model_pynative.train(1, dataset2, dataset_sink_mode=False)
    out_pynative = model_pynative.predict(Tensor(input_data))
    allclose_nparray(out_graph.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)
