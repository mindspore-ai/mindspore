# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from tests.mark_utils import arg_mark


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def proc_dataset(data_path, batch_size=32):
    mnist_ds = ds.MnistDataset(data_path, shuffle=True)

    # define map operations
    image_transforms = [
        ds.vision.Resize(32),
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transforms = ds.transforms.transforms.TypeCast(ms.int32)

    mnist_ds = mnist_ds.map(operations=label_transforms, input_columns="label")
    mnist_ds = mnist_ds.map(operations=image_transforms, input_columns="image")
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


def create_model():
    model = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(model.trainable_params(), learning_rate=0.01, momentum=0.9)
    trainer = ms.Model(model, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": nn.Accuracy()})
    return trainer


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_net_build_then_train_sink_size_1():
    """
    Feature: Test model.build and model.train in graph mode under Ascend platform
    Description: Test sink_size is equal to 1 and epoch is equal to 130, execute model.build first and then model.train
    Expectation: Training completes successfully
    """
    ms.set_context(mode=ms.GRAPH_MODE, op_timeout=60)
    trainer = create_model()
    train_dataset = proc_dataset(os.path.join("/home/workspace/mindspore_dataset/mnist", "train"))
    trainer.build(train_dataset, epoch=130, sink_size=1)
    trainer.train(130, train_dataset, dataset_sink_mode=True, sink_size=1)


if __name__ == '__main__':
    test_net_build_then_train_sink_size_1()
