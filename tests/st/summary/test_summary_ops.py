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
""" test summary ops."""
import os
import shutil
import tempfile

import numpy as np
import pytest

from mindspore import nn, Tensor, context
from mindspore.common.initializer import Normal
from mindspore.train.metrics import Loss
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.summary.summary_record import _get_summary_tensor_data
from tests.st.summary.dataset import create_mnist_dataset
from tests.security_utils import security_off_wrap


class LeNet5(nn.Cell):
    """LeNet network"""

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
        self.tensor_summary = P.TensorSummary()
        self.channel = Tensor(num_channel)

    def construct(self, x):
        """construct"""
        self.image_summary('x', x)
        self.tensor_summary('x', x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        self.scalar_summary('x_fc3', x[0][0])
        return x


class TestSummaryOps:
    """Test summary ops."""
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

    @pytest.mark.level1
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_ops(self):
        """Test summary operators."""
        ds_train = create_mnist_dataset('train', num_samples=1, batch_size=1)
        ds_train_iter = ds_train.create_dict_iterator()
        expected_data = next(ds_train_iter)['image'].asnumpy()

        net = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(net, loss_fn=loss, optimizer=optim, metrics={'loss': Loss()})
        model.train(1, ds_train, dataset_sink_mode=False)

        summary_data = _get_summary_tensor_data()
        image_data = summary_data['x[:Image]'].asnumpy()
        tensor_data = summary_data['x[:Tensor]'].asnumpy()
        x_fc3 = summary_data['x_fc3[:Scalar]'].asnumpy()

        assert np.allclose(expected_data, image_data)
        assert np.allclose(expected_data, tensor_data)
        assert not np.allclose(0, x_fc3)
