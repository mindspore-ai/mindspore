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
# ============================================================================
""" test model train """
import os
import re
import tempfile
import shutil

from collections import Counter

import pytest

from mindspore import dataset as ds
from mindspore import nn, Tensor, context
from mindspore.nn.metrics import Loss
from mindspore.nn.optim import Momentum
from mindspore.dataset.transforms import c_transforms as C
from mindspore.dataset.vision import c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal
from mindspore.train import Model
from mindspore.train.callback import SummaryCollector

from tests.summary_utils import SummaryReader


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


def create_dataset(data_path, num_samples=2):
    """create dataset for train or test"""
    num_parallel_workers = 1

    # define dataset
    mnist_ds = ds.MnistDataset(data_path, num_samples=num_samples)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift=0.0)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.shuffle(buffer_size=10000)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size=2, drop_remainder=True)

    return mnist_ds


class TestSummary:
    """Test summary collector the basic function."""
    base_summary_dir = ''
    mnist_path = '/home/workspace/mindspore_dataset/mnist'

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
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'loss': Loss()})
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=2, **kwargs)

        ds_train = create_dataset(os.path.join(self.mnist_path, "train"), num_samples=num_samples)
        model.train(1, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)

        ds_eval = create_dataset(os.path.join(self.mnist_path, "test"))
        model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode, callbacks=[summary_collector])
        return summary_dir

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
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
    @pytest.mark.env_onecard
    def test_summarycollector_user_defind(self):
        """Test SummaryCollector with user defind."""
        summary_dir = self._run_network(dataset_sink_mode=True, num_samples=2, user_defind={'test': 'self test'})

        tag_list = self._list_summary_tags(summary_dir)
        # There will not record input data when dataset sink mode is True
        expected_tags = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                         'fc2.weight/auto', 'loss/auto', 'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tags) == set(tag_list)


    @staticmethod
    def _list_summary_tags(summary_dir):
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
