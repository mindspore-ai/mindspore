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
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.ssd_ghostnet import SSD300, SSDWithLossCell, TrainingWrapper, ssd_ghostnet
from src.dataset import create_ssd_dataset, data_to_mindrecord_byte_image, voc_data_to_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


@moxing_wrapper()
def train_net():
    """train net"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, device_id=config.device_id)

    if config.run_distribute:
        device_num = config.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = get_rank()
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is ssd.mindrecord0, 1, ... file_num.

    prefix = "ssd.mindrecord"
    mindrecord_dir = os.path.join(config.data_path, "MindRecord_COCO")
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            coco_root = os.path.join(config.data_path, "coco_ori")
            if os.path.isdir(coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif config.dataset == "voc":
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

    if not config.only_create_dataset:
        loss_scale = float(config.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
        dataset = create_ssd_dataset(mindrecord_file, repeat_num=1,
                                     batch_size=config.batch_size, device_num=device_num, rank=rank)

        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        backbone = ssd_ghostnet()
        ssd = SSD300(backbone=backbone)
        net = SSDWithLossCell(ssd)
        init_net_param(net)

        # checkpoint
        ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path + str(rank))
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs, keep_checkpoint_max=60)
        ckpoint_cb = ModelCheckpoint(
            prefix="ssd", directory=ckpt_save_dir, config=ckpt_config)

        if config.pre_trained:
            if config.pre_trained_epoch_size <= 0:
                raise KeyError(
                    "pre_trained_epoch_size must be greater than 0.")
            param_dict = load_checkpoint(config.pre_trained)
            if config.filter_weight:
                filter_checkpoint_parameter(param_dict)
            load_param_into_net(net, param_dict)

        lr = Tensor(get_lr(global_step=config.global_step,
                           lr_init=config.lr_init, lr_end=config.lr_end_rate * config.lr, lr_max=config.lr,
                           warmup_epochs=config.warmup_epochs,
                           total_epochs=config.epoch_size,
                           steps_per_epoch=dataset_size))
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

        callback = [TimeMonitor(data_size=dataset_size),
                    LossMonitor(), ckpoint_cb]

        model = Model(net)
        dataset_sink_mode = False
        if config.sink_mode == "sink":
            print("In sink mode, one epoch return a loss.")
            dataset_sink_mode = True
        print("Start train SSD, the first epoch will be slower because of the graph compilation.")
        model.train(config.epoch_size, dataset,
                    callbacks=callback, dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    train_net()
