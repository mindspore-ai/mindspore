# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import sys
import tempfile
import shutil
import glob
import numpy as np
import pytest
import time
import mindspore.context as context

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.nn import Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell
from dump_test_utils import generate_dump_json, generate_dump_json_with_overflow, \
    generate_statistic_dump_json, check_statistic_dump, check_data_dump
from tests.security_utils import security_off_wrap


os.environ['MS_ENABLE_GE'] = '1'
os.environ['MS_DISABLE_REF_MODE'] = '1'


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)


def check_saved_data(iteration_path, saved_data):
    if not saved_data:
        return
    if saved_data in ('statistic', 'full'):
        check_statistic_dump(iteration_path)
    if saved_data in ('tensor', 'full'):
        check_data_dump(iteration_path, True)
    if saved_data == 'statistic':
        # assert only file is statistic.csv, tensor data is not saved
        assert len(os.listdir(iteration_path)) == 1
    elif saved_data == 'tensor':
        # assert only tensor data is saved, not statistics
        stat_path = os.path.join(iteration_path, 'statistic.csv')
        assert not os.path.isfile(stat_path)


def check_overflow_file(iteration_path, overflow_num, need_check):
    if not need_check:
        return overflow_num
    overflow_files = glob.glob(os.path.join(iteration_path, "Opdebug.Node_OpDebug.*.*.*"))
    overflow_num += len(overflow_files)
    return overflow_num


def check_iteration(iteration_id, num_iteration):
    if iteration_id.isdigit():
        assert int(iteration_id) < num_iteration


def check_ge_dump_structure(dump_path, num_iteration, device_num=1, check_overflow=False, saved_data=None):
    overflow_num = 0
    for _ in range(3):
        if not os.path.exists(dump_path):
            time.sleep(2)
    sub_paths = os.listdir(dump_path)
    for sub_path in sub_paths:
        # on GE, the whole dump directory of one training is saved within a time path, like '20230822120819'
        if not (sub_path.isdigit() and len(sub_path) == 14):
            continue
        time_path = os.path.join(dump_path, sub_path)
        assert os.path.isdir(time_path)
        device_paths = os.listdir(time_path)
        assert len(device_paths) == device_num
        for device_path in device_paths:
            assert device_path.isdigit()
            abs_device_path = os.path.join(time_path, device_path)
            assert os.path.isdir(abs_device_path)
            model_names = os.listdir(abs_device_path)
            for model_name in model_names:
                model_path = os.path.join(abs_device_path, model_name)
                assert os.path.isdir(model_path)
                model_ids = os.listdir(model_path)
                for model_id in model_ids:
                    model_id_path = os.path.join(model_path, model_id)
                    assert os.path.isdir(model_id_path)
                    iteration_ids = os.listdir(model_id_path)
                    for iteration_id in iteration_ids:
                        check_iteration(iteration_id, num_iteration)
                        iteration_path = os.path.join(model_id_path, iteration_id)
                        assert os.path.isdir(iteration_path)
                        check_saved_data(iteration_path, saved_data)
                        overflow_num = check_overflow_file(iteration_path, overflow_num, check_overflow)
    if check_overflow:
        assert overflow_num


def run_ge_dump(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'ge_dump')
        dump_config_path = os.path.join(tmp_dir, 'ge_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        output = add(Tensor(x), Tensor(y))
        check_ge_dump_structure(dump_path, 1, 1)
        if test_name == "test_ge_dump_npy":
            find_x_cmd = 'find {0} -name "Data.x*.output.*.npy"'.format(dump_path)
            x_file_path = os.popen(find_x_cmd).read()
            x_file_path = x_file_path.replace('\n', '')
            find_y_cmd = 'find {0} -name "Data.y*.output.*.npy"'.format(dump_path)
            y_file_path = os.popen(find_y_cmd).read()
            y_file_path = y_file_path.replace('\n', '')
            find_add_cmd = 'find {0} -name "Add.*.output.*.npy"'.format(dump_path)
            add_file_path = os.popen(find_add_cmd).read()
            add_file_path = add_file_path.replace('\n', '')
            x_output = np.load(x_file_path)
            y_output = np.load(y_file_path)
            add_output = np.load(add_file_path)
            assert (x_output == x).all()
            assert (y_output == y).all()
            assert (add_output == output.asnumpy()).all()
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_dump():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_ge_dump("test_ge_dump")


class ReluReduceMeanDenseRelu(Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.dense = Dense(in_channel, num_class, kernel, bias)

    def construct(self, x_):
        x_ = self.relu(x_)
        x_ = self.mean(x_, (2, 3))
        x_ = self.dense(x_)
        x_ = self.relu(x_)
        return x_


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_dump_net_multi_layer_mode1():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump on GE backend.
    Expectation: dump data are generated as GE dump structure.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'ge_dump_net_multi_layer_mode1')
        json_file_path = os.path.join(tmp_dir, "test_ge_dump_net_multi_layer_mode1.json")
        generate_dump_json(dump_path, json_file_path, 'test_ge_dump_net_multi_layer_mode1', 'test')
        os.environ['MINDSPORE_DUMP_CONFIG'] = json_file_path
        weight = Tensor(np.ones((1000, 2048)).astype(np.float32))
        bias = Tensor(np.ones((1000,)).astype(np.float32))
        net = ReluReduceMeanDenseRelu(weight, bias, 2048, 1000)
        criterion = SoftmaxCrossEntropyWithLogits(sparse=False)
        optimizer = Momentum(learning_rate=0.1, momentum=0.1,
                             params=filter(lambda x: x.requires_grad, net.get_parameters()))
        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()
        inputs = Tensor(np.random.randn(32, 2048, 7, 7).astype(np.float32))
        label = Tensor(np.zeros(shape=(32, 1000)).astype(np.float32))
        _ = train_network(inputs, label)
        check_ge_dump_structure(dump_path, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_dump_with_diagnostic_path():
    """
    Feature: Dump on GE backend when the MS_DIANOSTIC_DATA_PATH is set.
    Description: Test Ascend dump on GE when path is not set (set to empty) in dump json file and
     MS_DIAGNOSTIC_DATA_PATH is set.
    Expectation: Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_config_path = os.path.join(tmp_dir, 'ge_dump.json')
        generate_dump_json('', dump_config_path, 'test_ge_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        diagnose_path = os.path.join(tmp_dir, 'ge_dump')
        os.environ['MS_DIAGNOSTIC_DATA_PATH'] = diagnose_path
        add = Net()
        add(Tensor(x), Tensor(y))
        dump_path = os.path.join(diagnose_path, 'debug_dump')
        check_ge_dump_structure(dump_path, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['MS_DIAGNOSTIC_DATA_PATH']


def run_overflow_dump():
    """Run async dump and generate overflow"""
    if sys.platform != 'linux':
        return
    overflow_x = np.array([60000, 60000]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        generate_dump_json_with_overflow(dump_path, dump_config_path, 'test_ge_dump', 3)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(overflow_x), Tensor(overflow_x))
        check_ge_dump_structure(dump_path, 1, 1, True)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_overflow_dump():
    """
    Feature: Overflow Dump on GE backend
    Description: Test overflow dump
    Expectation: Overflow is occurred, and overflow dump file is in correct format
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    run_overflow_dump()


def run_train():
    context.set_context(mode=context.GRAPH_MODE)
    add = Net()
    add(Tensor(x), Tensor(y))


def run_saved_data_dump_test(scenario, saved_data):
    """Run dump on GE backend, testing statistic dump"""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_saved_data')
        dump_config_path = os.path.join(tmp_dir, 'test_saved_data.json')
        generate_statistic_dump_json(dump_path, dump_config_path, scenario, saved_data)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        exec_network_cmd = 'cd {0}; python -c "from test_ge_dump import run_train; run_train()"'.format(os.getcwd())
        _ = os.system(exec_network_cmd)
        check_ge_dump_structure(dump_path, 1, 1, saved_data=saved_data)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_statistic_dump():
    """
    Feature: Ascend Statistics Dump on GE backend
    Description: Test Ascend statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_ge_dump', 'statistic')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_tensor_dump():
    """
    Feature: Ascend Tensor Dump on GE backend
    Description: Test Ascend tensor dump
    Expectation: Tensors are stored in npy files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_ge_dump', 'tensor')


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_full_dump():
    """
    Feature: Ascend Full Dump on GE backend
    Description: Test Ascend full dump
    Expectation: Tensors are stored in npy files and their statistics stored in statistic.csv
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_ge_dump', 'full')

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ge_dump_npy():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with file_format set to npy
    Expectation: dump data are generated as npy files, and the value is correct
    """
    run_ge_dump("test_ge_dump_npy")
