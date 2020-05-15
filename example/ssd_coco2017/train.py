# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train SSD and get checkpoint files."""

import os
import math
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer

from mindspore.model_zoo.ssd import SSD300, SSDWithLossCell, TrainingWrapper, ssd_mobilenet_v2
from config import ConfigSSD
from dataset import create_ssd_dataset, data_to_mindrecord_byte_image


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def init_net_param(network, initialize_mode='XavierUniform'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_parameter_data(initializer(initialize_mode, p.data.shape(), p.data.dtype()))

def main():
    parser = argparse.ArgumentParser(description="SSD training")
    parser.add_argument("--only_create_dataset", type=bool, default=False, help="If set it true, only create "
                                                                                "Mindrecord, default is false.")
    parser.add_argument("--distribute", type=bool, default=False, help="Run distribute, default is false.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.25, help="Learning rate, default is 0.25.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, defalut is coco.")
    parser.add_argument("--epoch_size", type=int, default=70, help="Epoch size, default is 70.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint file path.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5, help="Save checkpoint epochs, default is 5.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)

    if args_opt.distribute:
        device_num = args_opt.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                          device_num=device_num)
        init()
        rank = args_opt.device_id % device_num
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is ssd.mindrecord0, 1, ... file_num.

    config = ConfigSSD()
    prefix = "ssd.mindrecord"
    mindrecord_dir = config.MINDRECORD_DIR
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.COCO_ROOT):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("COCO_ROOT not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    if not args_opt.only_create_dataset:
        loss_scale = float(args_opt.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
        dataset = create_ssd_dataset(mindrecord_file, repeat_num=args_opt.epoch_size,
                                     batch_size=args_opt.batch_size, device_num=device_num, rank=rank)

        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        ssd = SSD300(backbone=ssd_mobilenet_v2(), config=config)
        net = SSDWithLossCell(ssd, config)
        init_net_param(net)

        # checkpoint
        ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs)
        ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=None, config=ckpt_config)

        lr = Tensor(get_lr(global_step=0, lr_init=0, lr_end=0, lr_max=args_opt.lr,
                           warmup_epochs=max(args_opt.epoch_size // 20, 1),
                           total_epochs=args_opt.epoch_size,
                           steps_per_epoch=dataset_size))
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9, 0.0001, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

        if args_opt.checkpoint_path != "":
            param_dict = load_checkpoint(args_opt.checkpoint_path)
            load_param_into_net(net, param_dict)

        callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]

        model = Model(net)
        dataset_sink_mode = False
        if args_opt.mode == "sink":
            print("In sink mode, one epoch return a loss.")
            dataset_sink_mode = True
        print("Start train SSD, the first epoch will be slower because of the graph compilation.")
        model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)

if __name__ == '__main__':
    main()
