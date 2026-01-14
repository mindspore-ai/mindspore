# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Tests the kernel-by-kernel dump functionalities with LeNet5 model.
"""

import os
import sys
import tempfile
import glob
import shutil
import re
import json
import numpy as np
from mindspore import nn
from mindspore import context, _c_expression, Callback, dataset, Model
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from dump_test_utils import generate_dump_json, check_dump_structure
from dump_check import SyncDumpCheck


def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class WithLossCell(nn.Cell):
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, label):
        predict = self.network(x)
        return self.loss(predict, label)


class TrainOneStepCell(nn.Cell):
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = nn.Momentum(self.weights, 0.1, 0.9)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True)

    def construct(self, x, label):
        weights = self.weights
        grads = self.grad(self.network, weights)(x, label)
        return self.optimizer(grads)


def run_trans_flag(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = LeNet5()
        predict = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
        expect = net(predict)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        assert os.path.exists(dump_data_path)
        if test_name == "test_e2e_dump_trans_true":
            output_name = "BiasAdd.Default_fc3-Dense_BiasAdd-op5.0.0.*.output.0.DefaultFormat.*.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 10)
            assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


def check_fullname(op_name, number, content):
    for i in range(number):
        if re.search(op_name + str(i), content):
            continue
        return False
    return True


def run_trans_flag_execution_order(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = LeNet5()
        predict = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
        _ = net(predict)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_execution_order_path = os.path.join(dump_path, 'rank_0', 'execution_order',
                                                 'ms_execution_order_graph_0.csv')
        assert os.path.exists(dump_execution_order_path)
        with open(dump_execution_order_path, 'r', encoding='utf-8') as f:
            execution_order_content = f.read()
        check_fullname("ReLU-op", 4, execution_order_content)
        check_fullname("MaxPool-op", 2, execution_order_content)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@security_off_wrap
def test_ascend_kernel_by_kernel_lenet():
    """
    Feature: Ascend kernel by kernel dump with lenet5.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_lenet")

class Net(nn.Cell):
    """The test net"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x_):
        return self.fc(x_)

class StopAtStep(Callback):
    """
    Start profiling base on step.

    Args:
        start_step (int): The start step number.
        stop_step (int): The stop step number.
    """
    def __init__(self, start_step, stop_step):
        super().__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        # pylint: disable=W0212
        _c_expression._dump_set_dynamic()

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            # pylint: disable=W0212
            _c_expression._dump_start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            # pylint: disable=W0212
            _c_expression._dump_stop()


def generator():
    for _ in range(1):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))


def run_kbk_data_dump_dynamic(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level='O0')
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        network = Net()
        dynamic_data_dump = StopAtStep(2, 3)
        optimizer = nn.Momentum(network.trainable_params(), 1, 0.9)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        data = dataset.GeneratorDataset(generator, ["data", "label"])
        model = Model(network, loss, optimizer)
        model.train(3, data, callbacks=[dynamic_data_dump], dataset_sink_mode=False)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '2')
        assert os.path.exists(dump_data_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']


def run_lenet_sink_false_kbk_dump_sync(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        network = Net()
        optimizer = nn.Momentum(network.trainable_params(), 1, 0.9)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        data = dataset.GeneratorDataset(generator, ["data", "label"])
        model = Model(network, loss, optimizer)
        model.train(3, data, dataset_sink_mode=False)
        with open(dump_config_path, 'r', encoding="utf-8") as f:
            dump_json = json.load(f)
        dump_check = SyncDumpCheck(dump_json, iteration_id_list=3)
        dump_check.dump_result_check()
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_lenet_O0_sink_false_kbk_dump_sync():
    """
    Feature: Ascend kernel by kernel dump with lenet5 in non-data sinking mode.
    Description: Test kernel by kernel dump in tensor dump mode.
    Expectation: Dump files has tensor data.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O0")
    run_lenet_sink_false_kbk_dump_sync("test_lenet_sink_false_kbk_dump_sync")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@security_off_wrap
def test_lenet_O1_sink_false_kbk_dump_sync():
    """
    Feature: Ascend kernel by kernel dump with lenet5 in non-data sinking mode.
    Description: Test kernel by kernel dump in tensor dump mode.
    Expectation: Dump files has tensor data.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O1")
    run_lenet_sink_false_kbk_dump_sync("test_lenet_sink_false_kbk_dump_sync")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@security_off_wrap
def test_kbk_dynamic_data_dump():
    """
    Feature: Ascend kernel by kernel dump with lenet5.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    run_kbk_data_dump_dynamic("test_e2e_dump_lenet")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@security_off_wrap
def test_ascend_kbk_lenet_op_fullname():
    """
    Feature: IR op fullname determinacy.
    Description: Run kernel by kernel dump with info log and save_graphs.
    Expectation: Op fullname in dump execution order file should be changeless.
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True, save_graphs_path="./irs")
    os.environ['GLOG_v'] = '1'
    run_trans_flag_execution_order("test_e2e_dump_lenet")
    del os.environ['GLOG_v']
    shutil.rmtree("./irs")
