# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from importlib import import_module
from pathlib import Path
import numpy as np
import pytest
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
from tests.st.dump.dump_test_utils import generate_dump_json
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump')
        dump_config_path = os.path.join(tmp_dir, 'async_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_async_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        time.sleep(5)
        assert len(os.listdir(dump_file_path)) == 1


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
            assert len(os.listdir(dump_file_path)) == 5
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_e2e_dump()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump_with_hccl_env():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@security_off_wrap
def test_cpu_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_e2e_dump()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@security_off_wrap
def test_cpu_e2e_dump_with_hccl_set():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_e2e_dump()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_e2e_dump_with_hccl_set():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()


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
        generate_dump_json(dump_path, json_file_path, 'test_async_dump_net_multi_layer_mode1')
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
        dump_file_name = list(Path(dump_file_path).rglob("*SoftmaxCrossEntropyWithLogits*"))[0]
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


@pytest.mark.level0
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
        assert len(os.listdir(dump_file_path)) == 5


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
        assert len(os.listdir(exe_graph_path)) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_execution_graph():
    """Test dump with execution graph on GPU."""
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    run_e2e_dump_execution_graph()
