# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import time
import shutil
import glob
import csv
import numpy as np
import pytest
import mindspore.context as context

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, set_dump
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops import operations as P, constexpr
from mindspore.nn import Cell
from mindspore.nn import Dense
from dump_test_utils import generate_dump_json, generate_statistic_dump_json, check_dump_structure, \
    check_statistic_dump, check_data_dump
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)


def run_e2e_dump(test_key="test_e2e_dump"):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'e2e_dump')
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_key)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        if test_key == "test_kbk_e2e_set_dump":
            set_dump(add)
        add(Tensor(x), Tensor(y))
        if context.get_context("device_target") == "Ascend":
            assert len(os.listdir(dump_file_path)) == 3
            output_name = "Add.Default_Add-op*.0.0.*.output.0.DefaultFormat.npy"
        elif context.get_context("device_target") == "CPU":
            assert len(os.listdir(dump_file_path)) == 5
            output_name = "Add.Default_Add-op*.0.0.*.output.0.DefaultFormat.npy"
        else:
            assert len(os.listdir(dump_file_path)) == 3
            output_name = "Add.Default_Add-op*.0.0.*.output.0.DefaultFormat.npy"
        output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
        real_path = os.path.realpath(output_path)
        output = np.load(real_path)
        expect = np.array([[8, 10, 12], [14, 16, 18]], np.float32)
        assert output.dtype == expect.dtype
        assert np.array_equal(output, expect)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        if test_key == "test_kbk_e2e_set_dump" or \
           test_key == "test_kbk_e2e_dump_reg":
            check_dump_structure(dump_path, dump_config_path, 1, 1, 1, execution_history=False)
        else:
            check_dump_structure(dump_path, dump_config_path, 1, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']


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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_kbk_e2e_set_dump():
    """
    Feature: set_dump API for kbk e2e dump
    Description: Test set_dump API for kbk e2e dump
    Expectation: Targets are dumped
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level='O0')
    run_e2e_dump(test_key="test_kbk_e2e_set_dump")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_kbk_e2e_dump_reg():
    """
    Feature: test_kbk_e2e_dump_reg
    Description: test_kbk_e2e_dump_reg
    Expectation: Targets are dumped
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level='O0')
    run_e2e_dump(test_key="test_kbk_e2e_dump_reg")


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
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_diagnostic_path():
    """
    Test e2e dump when path is not set (set to empty) in dump json file and MS_DIAGNOSTIC_DATA_PATH is set.
    Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_execution_graph():
    """Test dump with execution graph."""
    context.set_context(mode=context.GRAPH_MODE)
    run_e2e_dump_execution_graph()


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


@pytest.mark.level2
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


@pytest.mark.level1
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
    empty_x = Tensor_(np.array([]).astype(np.float16))
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


@constexpr
def construct_tensor(cst):
    return Tensor(np.array(cst))


class ConstantNet(nn.Cell):
    def __init__(self):
        super(ConstantNet, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x_):
        return self.relu(construct_tensor(ops.shape(x_)))


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


@pytest.mark.level1
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_save_cce_graph():
    """
    Feature: Save cce file for Ascend ops
    Description: Test save cce file in GRAPH_MODE
    Expectation: there are cce files saved in kernel_meta/kernel_meta_*/kernel_meta
    """
    os.environ["MS_COMPILER_OP_LEVEL"] = "1"
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    cce_path = os.path.join(cur_path, "kernel_meta")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    add = Net()
    add(Tensor(x), Tensor(y))
    cce_file = glob.glob(cce_path + "/kernel_meta_*/kernel_meta/*.cce")[0]
    assert cce_file
    del os.environ["MS_COMPILER_OP_LEVEL"]
