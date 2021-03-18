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

"""train Deeptext and get checkpoint files."""

import argparse
import ast
import os
import time

import numpy as np
from src.Deeptext.deeptext_vgg16 import Deeptext_VGG16
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_deeptext_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet

import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import Momentum
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

np.set_printoptions(threshold=np.inf)

set_seed(1)

parser = argparse.ArgumentParser(description="Deeptext training")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default: False.")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset name, default: coco.")
parser.add_argument("--pre_trained", type=str, default="", help="Pretrained file path.")
parser.add_argument("--device_id", type=int, default=5, help="Device id, default: 5.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
parser.add_argument("--imgs_path", type=str, required=True,
                    help="Train images files paths, multiple paths can be separated by ','.")
parser.add_argument("--annos_path", type=str, required=True,
                    help="Annotations files paths of train images, multiple paths can be separated by ','.")
parser.add_argument("--mindrecord_prefix", type=str, default='Deeptext-TRAIN', help="Prefix of mindrecord.")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)

if __name__ == '__main__':
    if args_opt.run_distribute:
        rank = args_opt.rank_id
        device_num = args_opt.device_num
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is DeepText.mindrecord0, 1, ... file_num.
    prefix = args_opt.mindrecord_prefix
    config.train_images = args_opt.imgs_path
    config.train_txts = args_opt.annos_path
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if os.path.isdir(config.coco_root):
            if not os.path.exists(config.coco_root):
                print("Please make sure config:coco_root is valid.")
                raise ValueError(config.coco_root)
            print("Create Mindrecord. It may take some time.")
            data_to_mindrecord_byte_image(True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("coco_root not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    loss_scale = float(config.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_deeptext_dataset(mindrecord_file, repeat_num=1,
                                      batch_size=config.batch_size, device_num=device_num, rank_id=rank)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done! dataset_size = ", dataset_size)
    net = Deeptext_VGG16(config=config)
    net = net.set_train()

    load_path = args_opt.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        load_param_into_net(net, param_dict)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)

    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if args_opt.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='deeptext', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
