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

import os
import argparse
import logging
import ast

import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.communication.management import init, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.data_loader import create_dataset
from src.loss import CrossEntropyWithLogits
from src.utils import StepLossTimeMonitor
from src.config import cfg_unet

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

mindspore.set_seed(1)

def train_net(data_dir,
              cross_valid_ind=1,
              epochs=400,
              batch_size=16,
              lr=0.0001,
              run_distribute=False,
              cfg=None):


    if run_distribute:
        init()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)

    if cfg['model'] == 'unet_medical':
        net = UNetMedical(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    elif cfg['model'] == 'unet_nested':
        net = NestedUNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])
    elif cfg['model'] == 'unet_simple':
        net = UNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])
    else:
        raise ValueError("Unsupported model: {}".format(cfg['model']))

    if cfg['resume']:
        param_dict = load_checkpoint(cfg['resume_ckpt'])
        load_param_into_net(net, param_dict)

    criterion = CrossEntropyWithLogits()
    train_dataset, _ = create_dataset(data_dir, epochs, batch_size, True, cross_valid_ind, run_distribute, cfg["crop"],
                                      cfg['img_size'])
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_unet_medical_adam',
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=cfg['weight_decay'],
                        loss_scale=cfg['loss_scale'])

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg['FixedLossScaleManager'], False)

    model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")

    print("============== Starting Training ==============")
    model.train(1, train_dataset, callbacks=[StepLossTimeMonitor(batch_size=batch_size), ckpoint_cb],
                dataset_sink_mode=False)
    print("============== End Training ==============")




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-t', '--run_distribute', type=ast.literal_eval,
                        default=False, help='Run distribute, default: false.')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print("Training setting:", args)

    epoch_size = cfg_unet['epochs'] if not args.run_distribute else cfg_unet['distribute_epochs']
    train_net(data_dir=args.data_url,
              cross_valid_ind=cfg_unet['cross_valid_ind'],
              epochs=epoch_size,
              batch_size=cfg_unet['batchsize'],
              lr=cfg_unet['lr'],
              run_distribute=args.run_distribute,
              cfg=cfg_unet)
