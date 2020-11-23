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
"""
######################## train alexnet example ########################
train alexnet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import ast
import argparse
import os
from src.config import alexnet_cifar10_cfg, alexnet_imagenet_cfg
from src.dataset import create_dataset_cifar10, create_dataset_imagenet
from src.generator_lr import get_lr_cifar10, get_lr_imagenet
from src.alexnet import AlexNet
from src.get_param_groups import get_param_groups
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import dataset as de
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed

set_seed(1)
de.config.set_seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['imagenet', 'cifar10'],
                        help='dataset name.')
    parser.add_argument('--sink_size', type=int, default=-1, help='control the amount of data in each sink')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./", help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                            path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=ast.literal_eval,
                        default=True, help='dataset_sink_mode is False or True')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: 0)')
    args = parser.parse_args()

    device_num = int(os.environ.get("DEVICE_NUM", 1))
    if args.dataset_name == "cifar10":
        cfg = alexnet_cifar10_cfg
        if device_num > 1:
            cfg.learning_rate = cfg.learning_rate * device_num
            cfg.epoch_size = cfg.epoch_size * 2
    elif args.dataset_name == "imagenet":
        cfg = alexnet_imagenet_cfg
    else:
        raise ValueError("Unsupport dataset.")

    device_target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(save_graphs=False)

    if device_target == "Ascend":
        context.set_context(device_id=args.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    elif device_target == "GPU":
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        raise ValueError("Unsupported platform.")

    if args.dataset_name == "cifar10":
        ds_train = create_dataset_cifar10(args.data_path, cfg.batch_size, target=args.device_target)
    elif args.dataset_name == "imagenet":
        ds_train = create_dataset_imagenet(args.data_path, cfg.batch_size)
    else:
        raise ValueError("Unsupport dataset.")

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = AlexNet(cfg.num_classes, phase='train')

    loss_scale_manager = None
    metrics = None
    step_per_epoch = ds_train.get_dataset_size() if args.sink_size == -1 else args.sink_size
    if args.dataset_name == 'cifar10':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_cifar10(0, cfg.learning_rate, cfg.epoch_size, step_per_epoch))
        opt = nn.Momentum(network.trainable_params(), lr, cfg.momentum)
        metrics = {"Accuracy": Accuracy()}

    elif args.dataset_name == 'imagenet':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_imagenet(cfg.learning_rate, cfg.epoch_size, step_per_epoch))
        opt = nn.Momentum(params=get_param_groups(network),
                          learning_rate=lr,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay,
                          loss_scale=cfg.loss_scale)

        from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    else:
        raise ValueError("Unsupport dataset.")

    if device_target == "Ascend":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)
    elif device_target == "GPU":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Unsupported platform.")

    if device_num > 1:
        ckpt_save_dir = os.path.join(args.ckpt_path + "_" + str(get_rank()))
    else:
        ckpt_save_dir = args.ckpt_path

    time_cb = TimeMonitor(data_size=step_per_epoch)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=ckpt_save_dir, config=config_ck)

    print("============== Starting Training ==============")
    model.train(cfg.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode, sink_size=args.sink_size)
