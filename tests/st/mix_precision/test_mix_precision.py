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
import mindspore as ms
from mindspore.amp import auto_mixed_precision
from mindspore.common import dtype
from mindspore._c_expression import security
from mindspore import nn
from mindspore import ops
from mindspore import amp
from mindspore import Tensor
from mindspore import context
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train import Model
from mindspore._extends.parse import compile_config
from utils import FakeData
from utils import allclose_nparray
from utils import FakeDataInitMode
from utils import find_newest_validateir_file
from utils import clean_all_ir_files
from functools import wraps
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

def security_off_wrap(func):
    """Wrapper for tests which do not need to run security on."""

    @wraps(func)
    def pass_test_when_security_on(*args, **kwargs):
        if security.enable_security():
            return None
        return func(*args, **kwargs)

    return pass_test_when_security_on

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


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
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


class TestOp(ms.nn.Cell):
    def __init__(self):
        super(TestOp, self).__init__()
        self.weight = ms.Parameter(ms.Tensor(np.ones([32, 32, 3, 3]), ms.float32))

    def construct(self, x):
        ndim = x.ndim
        if ndim == 3:
            x = x.expand_dims(0)
            output = ms.ops.conv2d(x, self.weight)
            output = output.squeeze(0)
        else:
            output = ms.ops.conv2d(x, self.weight)
        return output


class TestNet(ms.nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.op = TestOp()

    def construct(self, x):
        out = self.op(x)
        return out


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@security_off_wrap
def test_all_subgraph_mix_precision():
    """
    Feature: Test all subgraph mixed precision
    Description: Test all subgraph mixed precision
    Expectation: success.
    """
    test_net = TestNet()
    mix_net = auto_mixed_precision(test_net, 'O2')
    x = ms.Tensor(np.ones([10, 32, 32, 32]), ms.float32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.AMP_ENABLE_ALL_FG = 1
    out_graph = mix_net(x)
    compile_config.AMP_ENABLE_ALL_FG = ''

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    out_pynative = mix_net(x)
    allclose_nparray(out_graph.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)


class AddNet(ms.nn.Cell):
    def __init__(self):
        super(AddNet, self).__init__()
        self.add = ops.Add()

    def construct(self, x):
        out = self.add(x, x)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dst_type', [ms.float16, ms.bfloat16])
@test_utils.run_test_with_On
def test_to_float(mode, dst_type):
    """
    Feature: to_float
    Description: Verify the result of to_float
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-1.0, 1.0, 0.0]), ms.float32)
    net = AddNet().to_float(dst_type)
    output = net(x)
    assert output.dtype == dst_type


class TestAmpNet(ms.nn.Cell):
    def __init__(self):
        super(TestAmpNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=1, weight_init='ones', bias_init='zeros')
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('amp_level', ["O1", "O2", "O3"])
def test_amp_bfloat16(amp_level):
    """
    Feature: to_float
    Description: Verify the result of to_float
    Expectation: success
    """
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O2"})
    context.set_context(save_graphs=True, save_graphs_path=f'./test_amp_{amp_level}')
    x = Tensor(np.random.rand(1, 3, 16, 16), ms.float32)
    net_graph = TestAmpNet()
    net_graph = amp.auto_mixed_precision(net_graph, amp_level=amp_level, dtype=ms.bfloat16)
    out_graph = net_graph(x)
    content = read_validateir_file(f'./test_amp_{amp_level}/')
    assert re.search(r"Conv2D(.*?)\n(.*) -> \(\<Tensor\[BFloat16\]", content), content
    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(save_graphs=False)
    net_pynative = TestAmpNet()
    net_pynative = amp.auto_mixed_precision(net_pynative, amp_level=amp_level, dtype=ms.bfloat16)
    out_pynative = net_pynative(x)
    allclose_nparray(out_graph.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_mixed_precision_bfloat16():
    """
    Feature: to_float
    Description: Verify the result of to_float
    Expectation: success
    """
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O2"})
    context.set_context(save_graphs=True, save_graphs_path=f'./test_custom_amp')
    x = Tensor(np.random.rand(1, 3, 16, 16), ms.float32)
    net_graph = TestAmpNet()
    white_list = [nn.ReLU, nn.Conv2d]
    net_graph = amp.custom_mixed_precision(net_graph, white_list=white_list, dtype=ms.bfloat16)
    out_graph = net_graph(x)
    content = read_validateir_file(f'./test_custom_amp/')
    assert re.search(r"Conv2D(.*?)\n(.*) -> \(\<Tensor\[BFloat16\]", content), content
    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(save_graphs=False)
    net_pynative = TestAmpNet()
    white_list = [nn.ReLU, nn.Conv2d]
    net_pynative = amp.custom_mixed_precision(net_pynative, white_list=white_list, dtype=ms.bfloat16)
    out_pynative = net_pynative(x)
    allclose_nparray(out_graph.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)
