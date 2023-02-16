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
import numpy as np
import pytest
from mindspore import context, Model, nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Accuracy
from mindspore.common import set_seed
from mindspore.common.initializer import Normal
import mindspore.dataset as ds
from dump_test_utils import generate_dump_json, generate_statistic_dump_json, check_dump_structure
from tests.security_utils import security_off_wrap

set_seed(1)


class LeNet5(nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def mock_mnistdataset(batch_size=32, repeat_size=1):
    """Mock the mnistdataset."""
    images = [np.random.randn(1, 32, 32).astype(np.float32) for i in range(10 * batch_size)]
    labels = [np.random.randint(9) for i in range(10 * batch_size)]
    data = ds.NumpySlicesDataset((images, labels), ['image', 'label'])
    data = data.batch(batch_size)
    data = data.repeat(repeat_size)
    return data


def train_net(epoch_size, repeat_size, sink_mode):
    """Define the training method."""
    ds_train = mock_mnistdataset(2, repeat_size)
    # create the network
    net = LeNet5()
    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    model.train(epoch_size, ds_train, dataset_sink_mode=sink_mode)


def run_async_dump(test_name):
    """Run lenet with async dump."""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'async_dump')
        dump_config_path = os.path.join(tmp_dir, 'async_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name, 'LeNet')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        train_net(1, 1, True)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, [1], [0])
        constant_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', 'constants')
        assert os.path.exists(constant_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_dump_dataset_sink():
    """
    Feature: async dump on Ascend
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_async_dump("test_async_dump_dataset_sink")


def run_e2e_dump():
    """Run lenet with sync dump."""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'e2e_dump')
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_e2e_dump', 'LeNet')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        train_net(1, 1, True)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, [1], [0])
        constant_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', 'constants')
        assert os.path.exists(constant_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_e2e_dump():
    """
    Feature: sync dump on Ascend.
    Description: test sync dump with dataset_sink_mode=True.
    Expectation: dump data are generated.
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
    Feature: sync dump on Ascend.
    Description: test sync dump with dataset_sink_mode=True, RANK_TABLE_FILE and RANK_ID envs are set.
    Expectation: dump data are generated.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    os.environ["RANK_TABLE_FILE"] = "invalid_file.json"
    os.environ["RANK_ID"] = "4"
    run_e2e_dump()
    del os.environ['RANK_TABLE_FILE']
    del os.environ['RANK_ID']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_dump_with_diagnostic_path():
    """
    Feature: Sync dump on Ascend.
    Description: Test sync dump with dataset_sink_mode=True when path is not set (set to empty) in dump json file and
                 MS_DIAGNOSTIC_DATA_PATH is set.
    Expectation: Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_config_path = os.path.join(tmp_dir, 'e2e_dump.json')
        generate_dump_json('', dump_config_path, 'test_e2e_dump', 'LeNet')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        diagnose_path = os.path.join(tmp_dir, 'e2e_dump')
        os.environ['MS_DIAGNOSTIC_DATA_PATH'] = diagnose_path
        if os.path.isdir(diagnose_path):
            shutil.rmtree(diagnose_path)
        train_net(1, 1, True)
        dump_path = os.path.join(diagnose_path, 'debug_dump')
        dump_file_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', '0')
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, [1], [0])
        constant_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', 'constants')
        assert os.path.exists(constant_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['MS_DIAGNOSTIC_DATA_PATH']


def check_statistic_dump(dump_file_path):
    """Check whether the statistic file exists in dump_file_path."""
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    assert os.path.getsize(real_path)


def check_data_dump(dump_file_path):
    """Check whether the tensor files exists in dump_file_path."""
    output_name = "*.npy"
    output_files = glob.glob(os.path.join(dump_file_path, output_name))
    assert len(output_files) > 11


def run_saved_data_dump_test(scenario, saved_data):
    """Run e2e dump on scenario, testing the saved_data field in dump config file."""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_saved_data')
        dump_config_path = os.path.join(tmp_dir, 'test_saved_data.json')
        generate_statistic_dump_json(dump_path, dump_config_path, scenario, saved_data, 'LeNet')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dump_file_path = os.path.join(dump_path, 'rank_0', 'LeNet', '1', '0')
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        train_net(1, 1, True)
        for _ in range(3):
            if not os.path.exists(dump_file_path):
                time.sleep(2)
        check_dump_structure(dump_path, dump_config_path, 1, 1, 1, [1], [0])
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
def test_ascend_tensor_dump():
    """
    Feature: Ascend Tensor Dump
    Description: Test Ascend tensor dump
    Expectation: Tensors are stored in npy files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_async_dump', 'tensor')


@pytest.mark.level1
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
