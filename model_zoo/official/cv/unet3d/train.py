# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import argparse
import ast
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from src.dataset import create_dataset
from src.unet3d_model import UNet3d
from src.config import config as cfg
from src.lr_schedule import dynamic_lr
from src.loss import SoftmaxCrossEntropyWithLogits

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, \
                    device_id=device_id)
mindspore.set_seed(1)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet3D on images and target masks')
    parser.add_argument('--data_url', dest='data_url', type=str, default='', help='image data directory')
    parser.add_argument('--seg_url', dest='seg_url', type=str, default='', help='seg data directory')
    parser.add_argument('--run_distribute', dest='run_distribute', type=ast.literal_eval, default=False, \
                        help='Run distribute, default: false')
    return parser.parse_args()

def train_net(data_dir,
              seg_dir,
              run_distribute,
              config=None):
    if run_distribute:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=True)
    else:
        rank_id = 0
        rank_size = 1
    train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, config=config, \
                                    rank_size=rank_size, rank_id=rank_id, is_training=True)
    train_data_size = train_dataset.get_dataset_size()
    print("train dataset length is:", train_data_size)

    network = UNet3d(config=config)

    loss = SoftmaxCrossEntropyWithLogits()
    lr = Tensor(dynamic_lr(config, train_data_size), mstype.float32)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
    scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    network.set_train()

    model = Model(network, loss_fn=loss, optimizer=optimizer, loss_scale_manager=scale_manager)

    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(config.model),
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks_list)
    print("============== End Training ==============")

if __name__ == '__main__':
    args = get_args()
    print("Training setting:", args)
    train_net(data_dir=args.data_url,
              seg_dir=args.seg_url,
              run_distribute=args.run_distribute,
              config=cfg)
