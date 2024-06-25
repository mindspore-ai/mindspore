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
import tempfile
import time
import shutil
import numpy as np
from mindspore import context, Model, nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Accuracy
from mindspore.common import set_seed
from mindspore.common.initializer import Normal
import mindspore.dataset as ds
from dump_test_utils import generate_dump_json, check_ge_dump_structure
from tests.mark_utils import arg_mark
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
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        train_net(1, 1, True)
        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        check_ge_dump_structure(dump_path, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_async_dump_dataset_sink():
    """
    Feature: async dump on Ascend
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_async_dump("test_async_dump_dataset_sink")
