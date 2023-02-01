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
import pytest

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.ops import operations as P
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager

ms.set_seed(1)


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 64

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(256, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


class Config:
    def __init__(self):
        self.data_path = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/train"
        self.device_num = 2
        self.device_target = "GPU"
        self.all_reduce_fusion_config = [85, 160]
        self.batch_size = 64
        self.train_image_size = 28
        self.run_distribute = True
        self.class_num = 1001
        self.lr_init = 0.04
        self.lr_end_kf = 0.0
        self.lr_max_kf = 0.4
        self.lr_end_ft = 0.0
        self.lr_max_ft = 0.8
        self.epoch_kf = 2
        self.epoch_ft = 1
        self.momentum = 0.9
        self.loss_scale = 1024
        self.keep_checkpoint_max = 10


config = Config()


def _get_rank_info(distribute):
    """
    get rank size and rank id
    """
    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id


def create_dataset(dataset_path, do_train, batch_size=32, train_image_size=28, target="Ascend", distribute=False):
    device_num, rank_id = _get_rank_info(distribute)

    ds.config.set_prefetch_size(64)
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id, num_samples=256)

    # Computed from random subset of ImageNet training images
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            ds.vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ds.vision.RandomHorizontalFlip(prob=0.5)
        ]
    trans_norm = [ds.vision.Normalize(mean=mean, std=std), ds.vision.HWC2CHW()]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    data_set = data_set.map(operations=trans_norm, input_columns="image", num_parallel_workers=12)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def set_parameter():
    """set_parameter"""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=config.device_target, save_graphs=False)
    init()
    ms.set_auto_parallel_context(device_num=config.device_num,
                                 parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                 gradients_mean=True,
                                 all_reduce_fusion_config=config.all_reduce_fusion_config)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_net_fade_then_sink():
    """
    Feature: The PYNATIVE mode under GPU has been trained for two consecutive times
    Description: Two consecutive training, non-sinking and then sinking mode
    Expectation: Training completes successfully
    """
    set_parameter()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             target=config.device_target, distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = LeNet()

    # apply golden-stick algo
    lr = 0.001

    optimizer = nn.Momentum(filter(lambda p: p.requires_grad, net.get_parameters()),
                            learning_rate=lr,
                            momentum=config.momentum,
                            loss_scale=config.loss_scale
                            )

    kf_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [loss_cb, time_cb]
    model = ms.Model(net, loss_fn=kf_loss_fn, optimizer=optimizer)
    model.train(config.epoch_kf, dataset, callbacks=cb, dataset_sink_mode=False)
    train_ft_fade(net)


def train_ft_fade(net):
    """train finetune."""
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             target=config.device_target, distribute=config.run_distribute)
    lr_ft_new = 0.001

    optimizer_ft = nn.Momentum(filter(lambda p: p.requires_grad, net.get_parameters()),
                               learning_rate=lr_ft_new,
                               momentum=config.momentum,
                               loss_scale=config.loss_scale
                               )
    net.set_train()
    metrics = {"acc"}
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    ft_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model_ft = ms.Model(net, loss_fn=ft_loss_fn, optimizer=optimizer_ft, loss_scale_manager=loss_scale,
                        metrics=metrics,
                        amp_level="O2", boost_level="O0", keep_batchnorm_fp32=False)

    step_size = dataset.get_dataset_size()

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=5 * step_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="resnet", config=config_ck)
    ft_cb = [time_cb, loss_cb, ckpt_cb]

    model_ft.train(config.epoch_ft, dataset, callbacks=ft_cb,
                   sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)


if __name__ == '__main__':
    test_train_net_fade_then_sink()
