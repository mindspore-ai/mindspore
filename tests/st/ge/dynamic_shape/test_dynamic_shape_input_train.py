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
import itertools
import math

import pytest
import numpy as np
from PIL import Image
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.common.initializer import HeNormal, HeUniform, Zero, initializer
from mindspore.train.callback import LossMonitor
from mindspore import Model, Tensor, context, nn, ops, set_seed

from tests.st.ge import ge_train_env  # pylint: disable=unused-import

set_seed(1)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = initializer(
        HeNormal(mode="fan_out", nonlinearity='relu'), weight_shape, mstype.float32)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = initializer(
        HeUniform(negative_slope=math.sqrt(5)), weight_shape, mstype.float32)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell([
            _conv3x3(3, 16, 2),
            _bn(16),
            nn.ReLU(),
        ])
        self.layer2 = nn.SequentialCell([
            _conv3x3(16, 32, 2),
            _bn(32),
            nn.ReLU(),
        ])
        self.mean = ops.ReduceMean(keep_dims=False)
        self.fc = _fc(32, 10)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.mean(x, (-2, -1))
        x = self.fc(x)
        return x


class NetWithLoss(nn.Cell):

    def __init__(self):
        super().__init__()
        self.net = Net()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, x, label):
        x = self.net(x)
        return self.criterion(x, label)


class DummyImageDataset:

    def __init__(self) -> None:
        self.class_names = [f'c{i}' for i in range(10)]
        self.class_idx = dict(
            zip(self.class_names, range(len(self.class_names))))
        rs = np.random.RandomState(123)
        sample_num = 1000
        cls_indexs = rs.randint(10, size=sample_num)
        values = (rs.rand(sample_num) * 64 + 63).astype(np.int32)
        if rs.rand() < 0.5:
            values = -values
        self.instance_seed = list(zip(cls_indexs, values))

    def __getitem__(self, index):
        label, value = self.instance_seed[index]
        size = 32
        img = np.ones((size, size, 3), dtype=np.int32) * 128
        if label == 0:
            return img.astype(np.uint8), label
        pos = label - 1
        y = pos // 3
        x = pos % 3
        delta = size / 3
        img[int(y*delta):int((y+1)*delta), int(x*delta):int((x+1)*delta), :] += value
        return img.astype(np.uint8), label

    def __len__(self):
        return len(self.instance_seed)


def create_dataset(batch_size=32):
    shapes = itertools.cycle([(24, 24), (32, 32), (48, 48)])

    def normal_image_preprocess(imgs, labels):
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        imgs = np.array(imgs, dtype=np.float32) / 255.
        imgs = (imgs - mean) / std
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        return imgs, np.array(labels, dtype=np.int32)

    def multi_shape_trans(imgs, labels, _):
        batch_img = []
        batch_label = []
        size = next(shapes)
        for img, label in zip(imgs, labels):
            img = Image.fromarray(img)
            img = img.resize(size)
            img = np.array(img, dtype=np.float32)
            batch_img.append(img)
            batch_label.append(label)
        return normal_image_preprocess(batch_img, batch_label)

    data_set = ds.GeneratorDataset(
        source=DummyImageDataset(), column_names=["image", "label"])
    return data_set.batch(batch_size, drop_remainder=True, per_batch_map=multi_shape_trans)


class LossCallBack(LossMonitor):
    def __init__(self):
        super(LossCallBack, self).__init__()
        self.last_10_losses = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        loss = np.mean(loss.asnumpy())
        self.last_10_losses = self.last_10_losses[-9:] + [loss]


def train(batch_size, lr, momentum, epochs, dataset_sink_mode):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = NetWithLoss()
    dyn_img = Tensor(shape=[batch_size, 3, None, None], dtype=mstype.float32)
    dyn_label = Tensor(shape=[batch_size], dtype=mstype.int32, init=Zero())
    net.set_inputs(dyn_img, dyn_label)

    optimizer = nn.Momentum(net.trainable_params(), lr, momentum)
    model = Model(net, loss_fn=None, optimizer=optimizer)
    dummy_dataset = create_dataset(batch_size)

    loss_callback = LossCallBack()

    model.train(epochs, dummy_dataset, callbacks=[loss_callback],
                sink_size=dummy_dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)

    avg_loss = np.min(loss_callback.last_10_losses)
    return avg_loss


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dummy_train_no_sink():
    """
    Feature: Training when the inputs is dynamic shape on GE backend
    Description: Train a small conv network on a dummy dataset and check the loss
    Expectation: the loss is convergent
    """
    avg_loss = train(32, 0.2, 0.9, 5, False)
    assert avg_loss < 0.1
