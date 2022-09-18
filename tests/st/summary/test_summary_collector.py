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
"""test SummaryCollector."""
import os
import re
import shutil
import tempfile
import json
from collections import Counter
import numpy as np

import pytest
from mindspore.common import set_seed
from mindspore import nn, Tensor, context
from mindspore.common.initializer import Normal
from mindspore.train.metrics import Loss
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.callback import SummaryCollector, SummaryLandscape
from mindspore.train.summary.summary_record import SummaryRecord
from tests.st.summary.dataset import create_mnist_dataset
from tests.summary_utils import SummaryReader
from tests.security_utils import security_off_wrap


def callback_fn():
    """A python function job"""
    network = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    metrics = {"Loss": Loss()}
    model = Model(network, loss, metrics=metrics)
    ds_train = create_mnist_dataset("train", num_samples=6)
    return model, network, ds_train, metrics


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """

    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

        self.scalar_summary = P.ScalarSummary()
        self.image_summary = P.ImageSummary()
        self.histogram_summary = P.HistogramSummary()
        self.tensor_summary = P.TensorSummary()
        self.channel = Tensor(num_channel)

    def construct(self, x):
        """construct."""
        self.image_summary('image', x)
        x = self.conv1(x)
        self.histogram_summary('histogram', x)
        x = self.relu(x)
        self.tensor_summary('tensor', x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        self.scalar_summary('scalar', self.channel)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestSummary:
    """Test summary collector the basic function."""
    base_summary_dir = ''

    @classmethod
    def setup_class(cls):
        """Run before test this class."""
        device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
        context.set_context(mode=context.GRAPH_MODE, device_id=device_id)
        cls.base_summary_dir = tempfile.mkdtemp(suffix='summary')

    @classmethod
    def teardown_class(cls):
        """Run after test this class."""
        if os.path.exists(cls.base_summary_dir):
            shutil.rmtree(cls.base_summary_dir)

    def _run_network(self, dataset_sink_mode=False, num_samples=2, **kwargs):
        """run network."""
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'loss': Loss()})
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=2, **kwargs)

        ds_train = create_mnist_dataset("train", num_samples=num_samples)
        model.train(1, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)

        ds_eval = create_mnist_dataset("test")
        model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode, callbacks=[summary_collector])
        return summary_dir

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_with_sink_mode_false(self):
        """Test summary with sink mode false, and num samples is 64."""
        summary_dir = self._run_network(num_samples=10)

        tag_list = self._list_summary_tags(summary_dir)

        expected_tag_set = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                            'fc2.weight/auto', 'input_data/auto', 'loss/auto',
                            'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tag_set) == set(tag_list)

        # num samples is 10, batch size is 2, so step is 5, collect freq is 2,
        # SummaryCollector will collect the first step and 2th, 4th step
        tag_count = 3
        for value in Counter(tag_list).values():
            assert value == tag_count

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_with_sink_mode_true(self):
        """Test summary with sink mode true, and num samples is 64."""
        summary_dir = self._run_network(dataset_sink_mode=True, num_samples=10)

        tag_list = self._list_summary_tags(summary_dir)

        # There will not record input data when dataset sink mode is True
        expected_tags = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                         'fc2.weight/auto', 'loss/auto', 'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tags) == set(tag_list)

        tag_count = 1
        for value in Counter(tag_list).values():
            assert value == tag_count

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summarycollector_user_defind(self):
        """Test SummaryCollector with user-defined."""
        summary_dir = self._run_network(dataset_sink_mode=True, num_samples=2,
                                        custom_lineage_data={'test': 'self test'},
                                        export_options={'tensor_format': 'npy'})

        tag_list = self._list_summary_tags(summary_dir)
        file_list = self._list_tensor_files(summary_dir)
        # There will not record input data when dataset sink mode is True
        expected_tags = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                         'fc2.weight/auto', 'loss/auto', 'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tags) == set(tag_list)
        expected_files = {'tensor_1.npy'}
        assert set(expected_files) == set(file_list)

    @staticmethod
    def _list_summary_tags(summary_dir):
        """list summary tags."""
        summary_file_path = ''
        for file in os.listdir(summary_dir):
            if re.search("_MS", file):
                summary_file_path = os.path.join(summary_dir, file)
                break
        assert summary_file_path

        tags = list()
        with SummaryReader(summary_file_path) as summary_reader:

            while True:
                summary_event = summary_reader.read_event()
                if not summary_event:
                    break
                for value in summary_event.summary.value:
                    tags.append(value.tag)
        return tags

    @staticmethod
    def _list_tensor_files(summary_dir):
        """list tensor tags."""
        export_file_path = ''
        for file in os.listdir(summary_dir):
            if re.search("export_", file):
                export_file_path = os.path.join(summary_dir, file)
                break
        assert export_file_path
        tensor_file_path = os.path.join(export_file_path, "tensor")
        assert tensor_file_path

        tensors = list()
        for file in os.listdir(tensor_file_path):
            tensors.append(file)

        return tensors

    def _train_network(self, epoch=3, dataset_sink_mode=False, num_samples=2, **kwargs):
        """run network."""
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.01, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim)
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=2, **kwargs)

        ds_train = create_mnist_dataset("train", num_samples=num_samples)
        model.train(epoch, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)
        return summary_dir

    @staticmethod
    def _list_summary_collect_landscape_tags(summary_dir):
        """list summary landscape tags."""
        summary_dir_path = ''
        for file in os.listdir(summary_dir):
            if re.search("ckpt_dir", file):
                summary_dir_path = os.path.join(summary_dir, file)
                break
        assert summary_dir_path

        summary_file_path = ''
        for file in os.listdir(summary_dir_path):
            if re.search(".json", file):
                summary_file_path = os.path.join(summary_dir_path, file)
                break
        assert summary_file_path

        tags = list()
        with open(summary_file_path, 'r') as file:
            data = json.load(file)
        for key, value in data.items():
            tags.append(key)

            assert value
        return tags

    @staticmethod
    def _list_landscape_tags(summary_dir):
        """list landscape tags."""
        expected_tags = {'landscape_[1, 3]', 'landscape_[3]'}
        summary_list = []
        for file in os.listdir(summary_dir):
            if re.search("_MS", file):
                summary_file_path = os.path.join(summary_dir, file)
                summary_list = summary_list + [summary_file_path]
            else:
                continue

        assert summary_list

        tags = []
        for summary_path in summary_list:
            with SummaryReader(summary_path) as summary_reader:

                while True:
                    summary_event = summary_reader.read_event()
                    if not summary_event:
                        break
                    for value in summary_event.summary.value:
                        if value.tag in expected_tags:
                            tags.append(value.loss_landscape.landscape.z.float_data)
                            break
        return tags

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_collector_landscape(self):
        """
        Feature: Summary collector with landscape.
        Description: Test summary collector with landscape.
        Expectation: Landscape data collected with expected value.
        """
        set_seed(1)
        interval_1 = [1, 2, 3]
        num_samples = 6
        summary_dir = self._train_network(epoch=3, num_samples=num_samples,
                                          collect_specified_data={'collect_landscape':
                                                                  {'landscape_size': 4,
                                                                   'unit': 'epoch',
                                                                   'create_landscape': {'train': True,
                                                                                        'result': True},
                                                                   'num_samples': num_samples,
                                                                   'intervals': [interval_1]}})

        tag_list = self._list_summary_collect_landscape_tags(summary_dir)
        expected_tags = {'epoch_group', 'model_params_file_map', 'step_per_epoch', 'unit', 'num_samples',
                         'landscape_size', 'create_landscape'}
        assert set(expected_tags) == set(tag_list)
        device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
        summary_landscape = SummaryLandscape(summary_dir)
        summary_landscape.gen_landscapes_with_multi_process(callback_fn, device_ids=[device_id])
        tag_list_landscape = self._list_landscape_tags(summary_dir)
        assert np.allclose(tag_list_landscape[0], 2.28, atol=0.03)
        assert np.allclose(tag_list_landscape[1], 2.28, atol=0.03)

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_of_more_than_one_instance(self):
        """
        Feature: Test the multi instances of SummaryRecord in a script.
        Description: Multi instances of SummaryRecord in a script.
        Expectation: Throw RuntimeError.
        """
        with pytest.raises(RuntimeError) as errinfo:
            summary_dir1 = tempfile.mkdtemp(dir=self.base_summary_dir)
            summary_record1 = SummaryRecord(log_dir=summary_dir1)
            summary_dir2 = tempfile.mkdtemp(dir=self.base_summary_dir)
            _ = SummaryRecord(log_dir=summary_dir2)
        assert "only one instance is supported in a training process" in str(errinfo.value)
        summary_record1.close()
