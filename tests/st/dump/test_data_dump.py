# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import json
import tempfile
import time
import shutil
import glob
import csv
from importlib import import_module
from pathlib import Path
import numpy as np
import pytest
import mindspore.context as context

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P, constexpr
from mindspore.nn import Cell
from mindspore.nn import Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell
from dump_test_utils import generate_dump_json, generate_dump_json_with_overflow, \
    generate_statistic_dump_json, check_dump_structure, find_nth_pos
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)


def run_async_dump(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump')
        dump_config_path = os.path.join(tmp_dir, 'async_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        assert len(os.listdir(dump_file_path)) == 1
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump():
    """
    Feature: async dump on Ascend
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_async_dump("test_async_dump")


def run_e2e_dump():
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'e2e_dump')
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_e2e_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        if context.get_context("device_target") == "Ascend":
            assert len(os.listdir(dump_file_path)) == 3
            output_name = "Add.Add-op*.0.0.*.output.0.DefaultFormat.npy"
        elif context.get_context("device_target") == "CPU":
            assert len(os.listdir(dump_file_path)) == 5
            output_name = "Add.Add-op*.0.0.*.output.0.DefaultFormat.npy"
        else:
            assert len(os.listdir(dump_file_path)) == 3
            output_name = "Add.Add-op*.0.0.*.output.0.DefaultFormat.npy"
        output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
        real_path = os.path.realpath(output_path)
        output = np.load(real_path)
        expect = np.array([[8, 10, 12], [14, 16, 18]], np.float32)
        assert output.dtype == expect.dtype
        assert np.array_equal(output, expect)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_e2e_dump()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump_with_hccl_env():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()
    del os.environ['RANK_TABLE_FILE']
    del os.environ['RANK_ID']


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@security_off_wrap
def test_cpu_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_e2e_dump()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@security_off_wrap
def test_cpu_e2e_dump_with_hccl_set():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()
    del os.environ['RANK_TABLE_FILE']
    del os.environ['RANK_ID']


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_e2e_dump()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_dump_with_hccl_set():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()
    del os.environ['RANK_TABLE_FILE']
    del os.environ['RANK_ID']


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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump_net_multi_layer_mode1():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump_net_multi_layer_mode1')
        json_file_path = os.path.join(tmp_dir, "test_async_dump_net_multi_layer_mode1.json")
        generate_dump_json(dump_path, json_file_path, 'test_async_dump_net_multi_layer_mode1', 'test')
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
        net_dict = train_network(inputs, label)
        dump_file_path = os.path.join(dump_path, 'rank_0', 'test', '0', '0')
        dump_file_name = list(Path(dump_file_path).rglob("SoftmaxCrossEntropyWithLogits*"))[0]
        dump_file_full_path = os.path.join(dump_file_path, dump_file_name)
        npy_path = os.path.join(dump_path, "npy_files")
        if os.path.exists(npy_path):
            shutil.rmtree(npy_path)
        os.mkdir(npy_path)
        tool_path_search_list = list(Path('/usr/local/Ascend').rglob('msaccucmp.py*'))
        if tool_path_search_list:
            converter = import_module("mindspore.offline_debug.convert_async")
            converter.AsyncDumpConverter([dump_file_full_path], npy_path).convert_files()
            npy_result_file = list(Path(npy_path).rglob("*output.0.*.npy"))[0]
            dump_result = np.load(os.path.join(npy_path, npy_result_file))
            for index, value in enumerate(net_dict):
                assert value.asnumpy() == dump_result[index]
        else:
            print('Failed to find hisi convert tools: msaccucmp.py or msaccucmp.pyc.')
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_diagnostic_path():
    """
    Test e2e dump when path is not set (set to empty) in dump json file and MS_DIAGNOSTIC_DATA_PATH is set.
    Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json('', dump_config_path, 'test_e2e_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        diagnose_path = os.path.join(tmp_dir, 'e2e_dump')
        os.environ['MS_DIAGNOSTIC_DATA_PATH'] = diagnose_path
        dump_file_path = os.path.join(diagnose_path, 'debug_dump', 'rank_0', 'Net', '0', '0')
        if os.path.isdir(diagnose_path):
            shutil.rmtree(diagnose_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        assert len(os.listdir(dump_file_path)) == 3
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['MS_DIAGNOSTIC_DATA_PATH']


def run_e2e_dump_execution_graph():
    """Run e2e dump and check execution order."""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'e2e_dump_exe_graph')
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_e2e_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        exe_graph_path = os.path.join(dump_path, 'rank_0', 'execution_order')
        assert len(os.listdir(exe_graph_path)) == 2
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_execution_graph():
    """Test dump with execution graph."""
    context.set_context(mode=context.GRAPH_MODE)
    run_e2e_dump_execution_graph()


def run_overflow_dump():
    """Run async dump and generate overflow"""
    if sys.platform != 'linux':
        return
    overflow_x = np.array([60000, 60000]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        generate_dump_json_with_overflow(dump_path, dump_config_path, 'test_async_dump', 3)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(overflow_x), Tensor(overflow_x))
        exe_graph_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        for _ in range(5):
            if not os.path.exists(exe_graph_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        # check if overflow dump generate exact two files, and the naming format
        assert len(os.listdir(exe_graph_path)) == 2
        output_path = glob.glob(os.path.join(exe_graph_path, "Add.Default_Add-op0.*.*.*"))[0]
        overflow_path = glob.glob(os.path.join(exe_graph_path, "Opdebug.Node_OpDebug.*.*.*"))[0]
        assert output_path
        assert overflow_path
        # check if generated files have matching task and stream id
        output_file_name = os.path.split(output_path)
        overflow_file_name = os.path.split(overflow_path)
        output_second_dot_pos = find_nth_pos(output_file_name[1], ".", 2)
        output_third_dot_pos = find_nth_pos(output_file_name[1], ".", 3)
        output_fourth_dot_pos = find_nth_pos(output_file_name[1], ".", 4)
        output_task_id = output_file_name[1][output_second_dot_pos+1:output_third_dot_pos]
        output_stream_id = output_file_name[1][output_third_dot_pos+1:output_fourth_dot_pos]

        overflow_second_dot_pos = find_nth_pos(overflow_file_name[1], ".", 2)
        overflow_third_dot_pos = find_nth_pos(overflow_file_name[1], ".", 3)
        overflow_fourth_dot_pos = find_nth_pos(overflow_file_name[1], ".", 4)
        overflow_task_id = overflow_file_name[1][overflow_second_dot_pos+1:overflow_third_dot_pos]
        overflow_stream_id = overflow_file_name[1][overflow_third_dot_pos+1:overflow_fourth_dot_pos]
        assert output_task_id == overflow_task_id
        assert output_stream_id == overflow_stream_id
        # Convert the overflow file into json format
        convert_tool = '/usr/local/Ascend/latest/tools/operator_cmp/compare/msaccucmp.py'
        convert_to_json_cmd = 'python {0} convert -d {1} -out {2}'.format(convert_tool, overflow_path, exe_graph_path)
        _ = os.system(convert_to_json_cmd)
        overflow_json_path = overflow_path + '.output.0.json'
        for _ in range(3):
            if not os.path.exists(overflow_json_path):
                time.sleep(2)
        # check if overflow dump file contains same task and stream id as file name
        with open(overflow_json_path) as f:
            file_content = json.load(f)
            ai_core = file_content["AI Core"]
            task_id_infile = ai_core["task_id"]
            stream_id_infile = ai_core["stream_id"]
            assert output_task_id == str(task_id_infile)
            assert output_stream_id == str(stream_id_infile)
        del os.environ['MINDSPORE_DUMP_CONFIG']


def run_not_overflow_dump():
    """Run async dump and not generate overflow"""
    if sys.platform != 'linux':
        return
    overflow_x = np.array([60000, 60000]).astype(np.float16)
    overflow_y = np.array([2, 2]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        generate_dump_json_with_overflow(dump_path, dump_config_path, 'test_async_dump', 3)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(overflow_x), Tensor(overflow_y))
        exe_graph_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        # check no overflow is happening, and path should not be generated
        assert not os.path.exists(exe_graph_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_overflow_dump():
    """
    Feature: Overflow Dump
    Description: Test overflow dump
    Expectation: Overflow is occurred, and overflow dump file is in correct format
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    run_overflow_dump()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_not_overflow_dump():
    """
    Feature: Overflow Dump
    Description: Test overflow dump
    Expectation: Overflow is not occurred, and overflow dump file is not generated
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    run_not_overflow_dump()

def check_statistic_dump(dump_file_path):
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path) as f:
        reader = csv.DictReader(f)
        stats = list(reader)
        num_tensors = len(stats)
        assert num_tensors == 3
        for tensor in stats:
            if tensor['IO'] == 'input' and tensor['Slot'] == 0:
                assert tensor['Min Value'] == '1'
                assert tensor['Max Value'] == '6'
            elif tensor['IO'] == 'input' and tensor['Slot'] == 1:
                assert tensor['Min Value'] == '7'
                assert tensor['Max Value'] == '12'
            elif tensor['IO'] == 'output' and tensor['Slot'] == 0:
                assert tensor['Min Value'] == '8'
                assert tensor['Max Value'] == '18'

def check_data_dump(dump_file_path):
    output_name = "Add.Add-op*.output.0.*.npy"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    output = np.load(real_path)
    expect = np.array([[8, 10, 12], [14, 16, 18]], np.float32)
    assert np.array_equal(output, expect)


def run_train():
    context.set_context(mode=context.GRAPH_MODE)
    add = Net()
    add(Tensor(x), Tensor(y))


def run_saved_data_dump_test(scenario, saved_data):
    """Run e2e dump on scenario, testing statistic dump"""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_saved_data')
        dump_config_path = os.path.join(tmp_dir, 'test_saved_data.json')
        generate_statistic_dump_json(dump_path, dump_config_path, scenario, saved_data)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        exec_network_cmd = 'cd {0}; python -c "from test_data_dump import run_train; run_train()"'.format(os.getcwd())
        _ = os.system(exec_network_cmd)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        if saved_data in ('statistic', 'full'):
            check_statistic_dump(dump_file_path)
        if saved_data in ('tensor', 'full'):
            check_data_dump(dump_file_path)
        if saved_data == 'statistic':
            # assert only file is statistic.csv, tensor data is not saved
            assert len(os.listdir(dump_file_path)) == 1
        elif saved_data == 'tensor':
            # assert only tensor data is saved, not statistics
            stat_path = os.path.join(dump_file_path, 'statistic.csv')
            assert not os.path.isfile(stat_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_statistic_dump():
    """
    Feature: GPU Statistics Dump
    Description: Test GPU statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_saved_data_dump_test('test_gpu_e2e_dump', 'statistic')

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_tensor_dump():
    """
    Feature: GPU Tensor Dump
    Description: Test GPU tensor dump
    Expectation: Tensor data are stored in npy files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_saved_data_dump_test('test_gpu_e2e_dump', 'tensor')

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_full_dump():
    """
    Feature: GPU Full Dump
    Description: Test GPU full dump
    Expectation: Tensor are stored in npy files and their statistics stored in statistic.csv
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_saved_data_dump_test('test_gpu_e2e_dump', 'full')

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_stat_dump_nulls():
    """
    Feature: GPU Statistics Dump
    Description: Test GPU statistics dump when printing tensors full with NaNs and Infs
    Expectation: Min, Max, Avg Values stored in statistic.csv show null for such tensors
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if sys.platform != 'linux':
        return
    empty_x = np.array([]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_saved_data')
        dump_config_path = os.path.join(tmp_dir, 'test_saved_data.json')
        generate_statistic_dump_json(dump_path, dump_config_path, 'test_gpu_e2e_dump', 'statistic')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(empty_x), Tensor(empty_x))
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        # check dumped data
        output_path = glob.glob(os.path.join(dump_file_path, 'statistic.csv'))[0]
        real_path = os.path.realpath(output_path)
        with open(real_path) as f:
            reader = csv.DictReader(f)
            [output] = list(reader)
            assert output['IO'] == 'output'
            assert output['Min Value'] == 'null'
            assert output['Max Value'] == 'null'
            assert output['Avg Value'] == 'null'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_statistic_dump():
    """
    Feature: Ascend Statistics Dump
    Description: Test Ascend statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_async_dump', 'statistic')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_statistic_dump_kernel_by_kernel():
    """
    Feature: Ascend Statistics Dump in kernel by kernel (mindRT) mode
    Description: Test Ascend statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """
    # set env `GRAPH_OP_RUN`` to enable kernel-by-kernel mode.
    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_async_dump', 'statistic')
    del os.environ['GRAPH_OP_RUN']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_tensor_dump():
    """
    Feature: Ascend Tensor Dump
    Description: Test Ascend tensor dump
    Expectation: Tensors are stored in npy files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_async_dump', 'tensor')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_full_dump():
    """
    Feature: Ascend Full Dump
    Description: Test Ascend full dump
    Expectation: Tensors are stored in npy files and their statistics stored in statistic.csv
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_async_dump', 'full')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_full_dump_kernel_by_kernel():
    """
    Feature: Ascend Full Dump in kernel-by-kernel (MindRT) mode
    Description: Test Ascend full dump
    Expectation: Tensors are stored in npy files and their statistics stored in statistic.csv
    """
    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_async_dump', 'full')
    del os.environ['GRAPH_OP_RUN']


@constexpr
def construct_tensor(cst):
    return Tensor(np.array(cst))


class ConstantNet(nn.Cell):
    def __init__(self):
        super(ConstantNet, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x_):
        return self.relu(construct_tensor(ops.shape(x_)))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_constant_async_ascend_dump():
    """
    Feature: Constant async dump
    Description: Test async constant dump in Ascend
    Expectation: constant dump folder is created, dump file has expected tensor info
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'constant_dump')
        dump_config_path = os.path.join(tmp_dir, 'constant_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_async_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = ConstantNet()
        tensor = Tensor(np.random.random([1, 2, 3]))
        expect = net(tensor)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        constant_path = os.path.join(dump_path, 'rank_0', 'Net', '0', 'constants')
        assert os.path.exists(constant_path)
        assert len(os.listdir(constant_path)) == 1

        output_name = "Parameter.data-*.0.0.*.DefaultFormat.npy"
        output_path = glob.glob(os.path.join(constant_path, output_name))[0]
        real_path = os.path.realpath(output_path)
        output = np.load(real_path)
        assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


def run_constant_e2e_dump():
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'constant_dump')
        dump_config_path = os.path.join(tmp_dir, 'constant_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_e2e_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = ConstantNet()
        tensor = Tensor(np.random.random([1, 2, 3]))
        expect = net(tensor)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        constant_path = os.path.join(dump_path, 'rank_0', 'Net', '0', 'constants')
        assert os.path.exists(constant_path)
        assert len(os.listdir(constant_path)) == 1

        output_name = "Parameter.data-*.0.0.*.DefaultFormat.npy"
        output_path = glob.glob(os.path.join(constant_path, output_name))[0]
        real_path = os.path.realpath(output_path)
        output = np.load(real_path)
        assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_constant_gpu_e2e_dump():
    """
    Feature: Constant sync dump
    Description: Test constant sync dump in GPU
    Expectation: constant dump folder is created, dump file has expected tensor info
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_constant_e2e_dump()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_constant_ascend_e2e_dump():
    """
    Feature: Constant sync dump
    Description: Test constant sync dump in Ascend
    Expectation: constant dump folder is created, dump file has expected tensor info
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_constant_e2e_dump()
