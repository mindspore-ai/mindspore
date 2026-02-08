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
"""SummaryCollector scripts without main function."""
import os
import re
from collections import Counter
import tempfile
import shutil
import sys
sys.path.append('../../../../../')
from mindspore import nn, Tensor, context
from mindspore.common.initializer import Normal
from mindspore.train import Loss
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore import SummaryCollector
from mindspore.communication import init, get_rank
import mindspore as ms
from tests.st.tools.summary.dataset import create_mnist_dataset
from tests.summary_utils import SummaryReader


context.set_context(mode=ms.GRAPH_MODE)
init()
rank_id = get_rank()
base_summary_dir = tempfile.mkdtemp(suffix='summary')

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
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid', weight_init="normal", bias_init="zeros")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid', weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02), bias_init="zeros")
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02), bias_init="zeros")
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02), bias_init="zeros")

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


def run_network(dataset_sink_mode=False, num_samples=2, **kwargs):
    """run network."""
    lenet = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'loss': Loss()})
    summary_dir = base_summary_dir + '/summary_' + str(rank_id)
    summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=2, **kwargs)

    ds_train = create_mnist_dataset("train", num_samples=num_samples)
    model.train(1, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)

    ds_eval = create_mnist_dataset("test")
    model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode, callbacks=[summary_collector])
    return summary_dir


def list_summary_tags(summary_dir):
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


def test_summary_net():
    """
    Feature: Test SummaryCollector in distribute training.
    Description: Run Summary script, init() is not in main function.
    Expectation: No error occur.
    """
    summary_path = run_network(num_samples=10)

    tag_list = list_summary_tags(summary_path)

    expected_tag_set = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                        'fc2.weight/auto', 'input_data/auto', 'loss/auto',
                        'histogram', 'image', 'scalar', 'tensor'}
    assert set(expected_tag_set) == set(tag_list)

    # num samples is 10, batch size is 2, so step is 5, collect freq is 2,
    # SummaryCollector will collect the first step and 2th, 4th step
    tag_count = 3
    # pylint: disable=E1121
    for count in Counter(tag_list).values():
        assert count == tag_count

    if os.path.exists(base_summary_dir):
        shutil.rmtree(base_summary_dir)
