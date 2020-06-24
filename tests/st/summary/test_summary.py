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

import pytest

from mindspore import dataset as ds
from mindspore import nn, Tensor, context
from mindspore.nn.metrics import Accuracy
from mindspore.nn.optim import Momentum
from mindspore.dataset.transforms import c_transforms as C
from mindspore.dataset.transforms.vision import c_transforms as CV
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.callback import SummaryCollector

from tests.summary_utils import SummaryReader


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """Define LeNet5 network."""
    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.scalar_summary = P.ScalarSummary()
        self.image_summary = P.ImageSummary()
        self.histogram_summary = P.HistogramSummary()
        self.tensor_summary = P.TensorSummary()
        self.channel = Tensor(channel)

    def construct(self, data):
        """define construct."""
        self.image_summary('image', data)
        output = self.conv1(data)
        self.histogram_summary('histogram', output)
        output = self.relu(output)
        self.tensor_summary('tensor', output)
        output = self.max_pool2d(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.max_pool2d(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        self.scalar_summary('scalar', self.channel)
        return output


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """create dataset for train or test"""
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

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
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.shuffle(buffer_size=10000)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class TestSummary:
    """Test summary collector the basic function."""
    base_summary_dir = ''
    mnist_path = '/home/workspace/mindspore_dataset/mnist'

    @classmethod
    def setup_class(cls):
        """Run before test this class."""
        cls.base_summary_dir = tempfile.mkdtemp(suffix='summary')

    @classmethod
    def teardown_class(cls):
        """Run after test this class."""
        if os.path.exists(cls.base_summary_dir):
            shutil.rmtree(cls.base_summary_dir)

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_summary_ascend(self):
        """Test summary ascend."""
        context.set_context(mode=context.GRAPH_MODE)
        self._run_network()

    def _run_network(self, dataset_sink_mode=True):
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'acc': Accuracy()})
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)

        ds_train = create_dataset(os.path.join(self.mnist_path, "train"))
        model.train(1, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)

        ds_eval = create_dataset(os.path.join(self.mnist_path, "test"))
        model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode, callbacks=[summary_collector])

        self._check_summary_result(summary_dir)

    @staticmethod
    def _check_summary_result(summary_dir):
        summary_file_path = ''
        for file in os.listdir(summary_dir):
            if re.search("_MS", file):
                summary_file_path = os.path.join(summary_dir, file)
                break

        assert not summary_file_path

        with SummaryReader(summary_file_path) as summary_reader:
            tags = set()

            # Read the event that record by SummaryCollector.begin
            summary_reader.read_event()

            summary_event = summary_reader.read_event()
            for value in summary_event.summary.value:
                tags.add(value.tag)

            # There will not record input data when dataset sink mode is True
            expected_tags = ['conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                             'fc2.weight/auto', 'histogram', 'image', 'scalar', 'tensor']
            assert set(expected_tags) == tags
