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
"""Train FaceBoxes."""
from __future__ import print_function
import os
import math
import argparse
import mindspore
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import faceboxes_config
from src.network import FaceBoxes, FaceBoxesWithLossCell, TrainingWrapper
from src.loss import MultiBoxLoss
from src.dataset import create_dataset
from src.lr_schedule import adjust_learning_rate
from src.utils import prior_box

parser = argparse.ArgumentParser(description='FaceBoxes: Face Detection')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--resume', type=str, default=None, help='resume training')
parser.add_argument('--device_target', type=str, default="Ascend", help='run device_target')
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = faceboxes_config
    mindspore.common.seed.set_seed(config['seed'])
    print('train config:\n', config)

    # set context and device init
    if args_opt.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=config['device_id'],
                            save_graphs=False)
        if int(os.getenv('RANK_SIZE', '1')) > 1:
            context.set_auto_parallel_context(device_num=config['rank_size'], parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        raise ValueError("Unsupported device_target.")

    # set parameters
    batch_size = config['batch_size']
    max_epoch = config['epoch']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    initial_lr = config['initial_lr']
    gamma = config['gamma']
    num_classes = 2
    negative_ratio = 7
    stepvalues = (config['decay1'], config['decay2'])

    # define dataset
    ds_train = create_dataset(args_opt.dataset_path, config, batch_size, multiprocessing=True,
                              num_worker=config["num_worker"])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    # define loss
    anchors_num = prior_box(config['image_size'], config['min_sizes'], config['steps'], config['clip']).shape[0]
    multibox_loss = MultiBoxLoss(num_classes, anchors_num, negative_ratio, config['batch_size'])

    # define net
    net = FaceBoxes(phase='train')
    net.set_train(True)
    # resume
    if args_opt.resume:
        param_dict = load_checkpoint(args_opt.resume)
        load_param_into_net(net, param_dict)
    net = FaceBoxesWithLossCell(net, multibox_loss, config)

    # define optimizer
    lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                              warmup_epoch=config['warmup_epoch'])
    opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                           weight_decay=weight_decay, loss_scale=1)

    # define model
    net = TrainingWrapper(net, opt)
    model = Model(net)

    # save model
    rank = 0
    if int(os.getenv('RANK_SIZE', '1')) > 1:
        rank = get_rank()
    ckpt_save_dir = config['save_checkpoint_path'] + "ckpt_" + str(rank) + "/"
    config_ck = CheckpointConfig(save_checkpoint_steps=config['save_checkpoint_epochs'],
                                 keep_checkpoint_max=config['keep_checkpoint_max'])
    ckpt_cb = ModelCheckpoint(prefix="FaceBoxes", directory=ckpt_save_dir, config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpt_cb]

    # training
    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list, dataset_sink_mode=True)
    print("============== End Training ==============")
