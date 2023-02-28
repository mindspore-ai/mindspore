# Copyright 2022 Huawei Technologies Co., Ltd
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
from pathlib import Path
import numpy as np
import pytest
import mindspore.context as context

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.nn import Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithLossCell
from mindspore import dataset as ds
from mindspore.train import Model
from dump_test_utils import generate_dump_json, check_dump_structure
from tests.security_utils import security_off_wrap


def dataset_generator():
    for i in range(1, 10):
        yield np.ones((32, 2 * i), dtype=np.float32), np.ones((32, 2 * i), dtype=np.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.shape = P.TensorShape()
        self.reshape = P.Reshape()

    def construct(self, x_, y_):
        val = self.add(x_, y_)
        size = self.shape(val)
        res = self.reshape(val, size)
        return res


def run_async_dump(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    network = Net()
    dataset = ds.GeneratorDataset(dataset_generator, ['data1', 'data2'])
    t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
    t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
    network.set_inputs(t0, t1)
    model = Model(network)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump')
        dump_config_path = os.path.join(tmp_dir, 'async_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '1', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        model.train(10, dataset, dataset_sink_mode=True)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 2, 1, [1])
        assert os.listdir(dump_file_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump():
    """
    Feature: async dump on Ascend
    Description: test async dump with default file_format value ("bin"), in fact, the tensor file is saved as npy.
    Expectation: dump data are generated as npy.
    """
    run_async_dump("test_async_dump")


def run_e2e_dump():
    if sys.platform != 'linux':
        return
    network = Net()
    dataset = ds.GeneratorDataset(dataset_generator, ['data1', 'data2'])
    t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
    t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
    network.set_inputs(t0, t1)
    model = Model(network)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'e2e_dump')
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_e2e_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '1', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        model.train(10, dataset, dataset_sink_mode=True)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, [1])
        assert os.listdir(dump_file_path)
        output_name = "Add.Add-op*.0.*.*.output.0.DefaultFormat.npy"
        output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
        real_path = os.path.realpath(output_path)
        output = np.load(real_path)
        expect = np.ones((32, 2), dtype=np.float32) * 2
        assert output.dtype == expect.dtype
        assert np.array_equal(output, expect)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump():
    """
    Feature: e2e dump on Ascend.
    Description: test e2e dump.
    Expectation: dump data are generated as npy.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_e2e_dump()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump_with_hccl_env():
    """
    Feature: e2e dump on Ascend with hccl env.
    Description: test e2e dump.
    Expectation: dump data are generated as npy.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump_net_multi_layer_mode1():
    """
    Feature: e2e dump on Ascend.
    Description: test e2e dump with a multi_layer_model.
    Expectation: dump data are generated as npy.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump_net_multi_layer_mode1')
        json_file_path = os.path.join(tmp_dir, "test_async_dump_net_multi_layer_mode1.json")
        generate_dump_json(dump_path, json_file_path, 'test_async_dump_net_multi_layer_mode1_npy')
        os.environ['MINDSPORE_DUMP_CONFIG'] = json_file_path
        weight = Tensor(np.ones((1000, 2048)).astype(np.float32))
        bias = Tensor(np.ones((1000,)).astype(np.float32))
        net = ReluReduceMeanDenseRelu(weight, bias, 2048, 1000)
        criterion = SoftmaxCrossEntropyWithLogits(sparse=False)
        net_with_criterion = WithLossCell(net, criterion)
        input_dynamic = Tensor(shape=[None, 2048, 7, 7], dtype=mindspore.float32)
        label_dynamic = Tensor(shape=[None, 1000], dtype=mindspore.float32)
        inputs = Tensor(np.random.randn(32, 2048, 7, 7).astype(np.float32))
        label = Tensor(np.zeros(shape=(32, 1000)).astype(np.float32))
        net_with_criterion.set_inputs(input_dynamic, label_dynamic)
        net_dict = net_with_criterion(inputs, label)
        dump_file_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
        dump_file_name = list(Path(dump_file_path).rglob("*SoftmaxCrossEntropyWithLogits*.output.0.*.npy"))[0]
        dump_file_full_path = os.path.join(dump_file_path, dump_file_name)
        dump_result = np.load(dump_file_full_path)
        for index, value in enumerate(net_dict):
            assert value.asnumpy() == dump_result[index]
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_diagnostic_path():
    """
    Feature: e2e dump on Ascend with path not set (set to empty).
    Description: test e2e dump when path is not set (set to empty) in dump json file and MS_DIAGNOSTIC_DATA_PATH is set.
    Expectation: Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
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
        inputx_dynamic = Tensor(shape=[None, 100], dtype=mindspore.float32)
        inputy_dynamic = Tensor(shape=[None, 100], dtype=mindspore.float32)
        add.set_inputs(inputx_dynamic, inputy_dynamic)
        x = Tensor(np.random.randn(10, 100).astype(np.float32))
        y = Tensor(np.random.randn(10, 100).astype(np.float32))
        add(Tensor(x), Tensor(y))
        assert len(os.listdir(dump_file_path)) == 8
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
        inputx_dynamic = Tensor(shape=[None, 100], dtype=mindspore.float32)
        inputy_dynamic = Tensor(shape=[None, 100], dtype=mindspore.float32)
        add.set_inputs(inputx_dynamic, inputy_dynamic)
        x = Tensor(np.random.randn(10, 100).astype(np.float32))
        y = Tensor(np.random.randn(10, 100).astype(np.float32))
        add(Tensor(x), Tensor(y))
        exe_graph_path = os.path.join(dump_path, 'rank_0', 'execution_order')
        assert len(os.listdir(exe_graph_path)) == 2
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_execution_graph():
    """
    Feature: e2e dump on Ascend saves execution graph.
    Description: test e2e dump  saves execution graph.
    Expectation: ms_execution_order_graph_0.csv and ms_global_execution_order_graph_0.csv are expected to be dumped into
    folder execution_order.
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_e2e_dump_execution_graph()


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
    add = Net()
    inputx_dynamic = Tensor(shape=[None, 3], dtype=mindspore.float32)
    inputy_dynamic = Tensor(shape=[None, 3], dtype=mindspore.float32)
    add.set_inputs(inputx_dynamic, inputy_dynamic)
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)
    add(Tensor(x), Tensor(y))
