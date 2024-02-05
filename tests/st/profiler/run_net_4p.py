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
import numpy as np
import mindspore.nn as nn
from mindspore.communication.management import init
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.common import dtype
from mindspore.profiler import Profiler

set_seed(100)
ds.config.set_seed(100)


def get_lr_cifar10(current_step, lr_max, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       current_step(int): current steps of the training
       lr_max(float): max learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        else:
            lr = lr_max * 0.1
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def create_dataset_cifar10(data_path, device_num, rank_id, batch_size=32, num_parallel_workers=8):
    """
    create dataset for train or test
    """

    ds.config.set_prefetch_size(64)

    cifar_ds = ds.Cifar10Dataset(data_path, num_parallel_workers=num_parallel_workers,
                                 shuffle=True, num_shards=device_num, shard_id=rank_id)

    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = CV.Resize((227, 227))
    rescale_op = CV.Rescale(rescale, shift)
    random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
    random_horizontal_op = CV.RandomHorizontalFlip()

    typecast_op = C.TypeCast(dtype.int32)
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op,
                            num_parallel_workers=1)
    compose_op = [random_crop_op, random_horizontal_op, resize_op, rescale_op]
    cifar_ds = cifar_ds.map(input_columns="image", operations=compose_op, num_parallel_workers=num_parallel_workers)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    return cifar_ds


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class DataNormTranspose(nn.Cell):

    def __init__(self):
        super(DataNormTranspose, self).__init__()
        # Computed from random subset of ImageNet training images
        self.mean = Tensor(np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 1, 3)), dtype.float32)
        self.std = Tensor(np.array([0.2023, 0.1994, 0.2010]).reshape((1, 1, 1, 3)), dtype.float32)

    def construct(self, x):
        x = (x - self.mean) / self.std
        x = F.transpose(x, (0, 3, 1, 2))
        return x


class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
        super(AlexNet, self).__init__()
        self.data_trans = DataNormTranspose()
        self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = P.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = nn.Dropout(p=1 - dropout_ratio)

    def construct(self, x):
        """define network"""
        x = self.data_trans(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(q, profiler_path, device_num=8, device_id=0):
    # config
    data_path = '/home/workspace/mindspore_dataset/cifar-10-batches-bin/'
    device_target = 'Ascend'
    num_classes = 10
    sink_size = -1
    lr = 0.016
    epoch_size = 1
    momentum = 0.9
    dataset_sink_mode = True

    os.environ["RANK_ID"] = str(device_id)
    os.environ["DEVICE_ID"] = str(device_id)

    try:
        context.set_context(mode=context.GRAPH_MODE, device_id=device_id, device_target=device_target)

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()

        profiler = Profiler(output_path=profiler_path, profile_communication=True)

        ds_train = create_dataset_cifar10(data_path, device_num, device_id)

        if ds_train.get_dataset_size() == 0:
            raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

        network = AlexNet(num_classes, phase='train')
        loss_scale_manager = None
        step_per_epoch = ds_train.get_dataset_size()

        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_cifar10(0, lr, epoch_size, step_per_epoch))
        opt = nn.Momentum(network.trainable_params(), lr, momentum)

        model = Model(network, loss_fn=loss, optimizer=opt, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)

        print("============== Starting Training ==============")
        model.train(epoch_size, ds_train,
                    dataset_sink_mode=dataset_sink_mode, sink_size=sink_size)

        profiler.analyse()

        q.put('success')
    except ValueError as err:
        q.put(err)
