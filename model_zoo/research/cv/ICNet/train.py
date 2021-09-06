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
"""train ICNet and get checkpoint files."""
import os
import sys
import logging
import argparse
import yaml
import mindspore.nn as nn
from mindspore import Model
from mindspore import context
from mindspore import set_seed
from mindspore.context import ParallelMode
from mindspore.communication import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import TimeMonitor

device_id = int(os.getenv('RANK_ID'))
device_num = int(os.getenv('RANK_SIZE'))

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

parser = argparse.ArgumentParser(description="ICNet Evaluation")
parser.add_argument("--project_path", type=str, help="project_path")

args_opt = parser.parse_args()

def train_net():
    """train"""
    set_seed(1234)
    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True,
                                          gradients_mean=True)
        init()
    prefix = 'cityscapes-2975.mindrecord'
    mindrecord_dir = cfg['train']["mindrecord_dir"]
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    dataset = create_icnet_dataset(mindrecord_file, batch_size=cfg['train']["train_batch_size_percard"],
                                   device_num=device_num, rank_id=device_id)

    train_data_size = dataset.get_dataset_size()
    print("data_size", train_data_size)
    epoch = cfg["train"]["epochs"]
    network = ICNetdc(pretrained_path=cfg["train"]["pretrained_model_path"])  # __init__

    iters_per_epoch = train_data_size
    total_train_steps = iters_per_epoch * epoch
    base_lr = cfg["optimizer"]["init_lr"]
    iter_lr = poly_lr(base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    optim = nn.SGD(params=network.trainable_params(), learning_rate=iter_lr, momentum=cfg["optimizer"]["momentum"],
                   weight_decay=cfg["optimizer"]["weight_decay"])

    model = Model(network, optimizer=optim, metrics=None)

    config_ck_train = CheckpointConfig(save_checkpoint_steps=iters_per_epoch * cfg["train"]["save_checkpoint_epochs"],
                                       keep_checkpoint_max=cfg["train"]["keep_checkpoint_max"])
    ckpoint_cb_train = ModelCheckpoint(prefix='ICNet', directory=args_opt.project_path + 'ckpt' + str(device_id),
                                       config=config_ck_train)
    time_cb_train = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb_train = LossMonitor()
    print("train begins------------------------------")
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpoint_cb_train, loss_cb_train, time_cb_train],
                dataset_sink_mode=True)


if __name__ == '__main__':
    # Set config file
    sys.path.append(args_opt.project_path)
    from src.cityscapes_mindrecord import create_icnet_dataset
    from src.models.icnet_dc import ICNetdc
    from src.lr_scheduler import poly_lr
    config_file = "src/model_utils/icnet.yaml"
    config_path = os.path.join(args_opt.project_path, config_file)
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
    logging.basicConfig(level=logging.INFO)
    train_net()
