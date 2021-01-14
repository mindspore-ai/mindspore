# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train FCN8s."""

import os
import argparse
from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src.data import dataset as data_generator
from src.loss import loss
from src.utils.lr_scheduler import CosineAnnealingLR
from src.nets.FCN8s import FCN8s
from src.config import FCN8s_VOC2012_cfg

set_seed(1)


def parse_args():
    parser = argparse.ArgumentParser('mindspore FCN training')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: None)')
    args, _ = parser.parse_known_args()
    return args


def train():
    args = parse_args()
    cfg = FCN8s_VOC2012_cfg
    device_num = int(os.environ.get("DEVICE_NUM", 1))
    # init multicards training
    if device_num > 1:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=device_num)
        init()
    args.rank = get_rank()
    args.group_size = get_group_size()

    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                        device_target="Ascend", device_id=args.device_id)

    # dataset
    dataset = data_generator.SegDataset(image_mean=cfg.image_mean,
                                        image_std=cfg.image_std,
                                        data_file=cfg.data_file,
                                        batch_size=cfg.batch_size,
                                        crop_size=cfg.crop_size,
                                        max_scale=cfg.max_scale,
                                        min_scale=cfg.min_scale,
                                        ignore_label=cfg.ignore_label,
                                        num_classes=cfg.num_classes,
                                        num_readers=2,
                                        num_parallel_calls=4,
                                        shard_id=args.rank,
                                        shard_num=args.group_size)
    dataset = dataset.get_dataset(repeat=1)

    net = FCN8s(n_class=cfg.num_classes)
    loss_ = loss.SoftmaxCrossEntropyLoss(cfg.num_classes, cfg.ignore_label)

    # load pretrained vgg16 parameters to init FCN8s
    if cfg.ckpt_vgg16:
        param_vgg = load_checkpoint(cfg.ckpt_vgg16)
        param_dict = {}
        for layer_id in range(1, 6):
            sub_layer_num = 2 if layer_id < 3 else 3
            for sub_layer_id in range(sub_layer_num):
                # conv param
                y_weight = 'conv{}.{}.weight'.format(layer_id, 3 * sub_layer_id)
                x_weight = 'vgg16_feature_extractor.conv{}_{}.0.weight'.format(layer_id, sub_layer_id + 1)
                param_dict[y_weight] = param_vgg[x_weight]
                # BatchNorm param
                y_gamma = 'conv{}.{}.gamma'.format(layer_id, 3 * sub_layer_id + 1)
                y_beta = 'conv{}.{}.beta'.format(layer_id, 3 * sub_layer_id + 1)
                x_gamma = 'vgg16_feature_extractor.conv{}_{}.1.gamma'.format(layer_id, sub_layer_id + 1)
                x_beta = 'vgg16_feature_extractor.conv{}_{}.1.beta'.format(layer_id, sub_layer_id + 1)
                param_dict[y_gamma] = param_vgg[x_gamma]
                param_dict[y_beta] = param_vgg[x_beta]
        load_param_into_net(net, param_dict)
    # load pretrained FCN8s
    elif cfg.ckpt_pre_trained:
        param_dict = load_checkpoint(cfg.ckpt_pre_trained)
        load_param_into_net(net, param_dict)

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()

    lr_scheduler = CosineAnnealingLR(cfg.base_lr,
                                     cfg.train_epochs,
                                     iters_per_epoch,
                                     cfg.train_epochs,
                                     warmup_epochs=0,
                                     eta_min=0)
    lr = Tensor(lr_scheduler.get_lr())

    # loss scale
    manager_loss_scale = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001,
                            loss_scale=cfg.loss_scale)

    model = Model(net, loss_fn=loss_, loss_scale_manager=manager_loss_scale, optimizer=optimizer, amp_level="O3")

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_steps,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=cfg.model, directory=cfg.train_dir, config=config_ck)
        cbs.append(ckpoint_cb)

    model.train(cfg.train_epochs, dataset, callbacks=cbs)


if __name__ == '__main__':
    train()
