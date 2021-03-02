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

"""Train SSD and get checkpoint files."""

import os
import argparse
import ast
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.ssd_ghostnet import SSD300, SSDWithLossCell, TrainingWrapper, ssd_ghostnet
from src.config_ghostnet_13x import config
from src.dataset import create_ssd_dataset, data_to_mindrecord_byte_image, voc_data_to_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter


def get_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="SSD training")
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=4,
                        help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1,
                        help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink",
                        help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco",
                        help="Dataset, default is coco.")
    parser.add_argument("--epoch_size", type=int, default=500,
                        help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None,
                        help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int,
                        default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int,
                        default=10, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024,
                        help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter weight parameters, default is False.")
    args_opt = parser.parse_args()
    return args_opt

def main():
    args_opt = get_args()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", device_id=args_opt.device_id)

    if args_opt.distribute:
        device_num = args_opt.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = args_opt.device_id % device_num
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is ssd.mindrecord0, 1, ... file_num.

    prefix = "ssd.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif args_opt.dataset == "voc":
            if os.path.isdir(config.voc_dir):
                print("Create Mindrecord.")
                voc_data_to_mindrecord(mindrecord_dir, True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("voc_dir not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")

    if not args_opt.only_create_dataset:
        loss_scale = float(args_opt.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
        dataset = create_ssd_dataset(mindrecord_file, repeat_num=1,
                                     batch_size=args_opt.batch_size, device_num=device_num, rank=rank)

        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        backbone = ssd_ghostnet()
        ssd = SSD300(backbone=backbone, config=config)
        net = SSDWithLossCell(ssd, config)
        init_net_param(net)

        # checkpoint
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs, keep_checkpoint_max=60)
        ckpoint_cb = ModelCheckpoint(
            prefix="ssd", directory=None, config=ckpt_config)

        if args_opt.pre_trained:
            if args_opt.pre_trained_epoch_size <= 0:
                raise KeyError(
                    "pre_trained_epoch_size must be greater than 0.")
            param_dict = load_checkpoint(args_opt.pre_trained)
            if args_opt.filter_weight:
                filter_checkpoint_parameter(param_dict)
            load_param_into_net(net, param_dict)

        lr = Tensor(get_lr(global_step=config.global_step,
                           lr_init=config.lr_init, lr_end=config.lr_end_rate * args_opt.lr, lr_max=args_opt.lr,
                           warmup_epochs=config.warmup_epochs,
                           total_epochs=args_opt.epoch_size,
                           steps_per_epoch=dataset_size))
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

        callback = [TimeMonitor(data_size=dataset_size),
                    LossMonitor(), ckpoint_cb]

        model = Model(net)
        dataset_sink_mode = False
        if args_opt.mode == "sink":
            print("In sink mode, one epoch return a loss.")
            dataset_sink_mode = True
        print("Start train SSD, the first epoch will be slower because of the graph compilation.")
        model.train(args_opt.epoch_size, dataset,
                    callbacks=callback, dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    main()
