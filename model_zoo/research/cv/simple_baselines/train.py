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
'''
train
'''
from __future__ import division

import os
import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.optim import Adam
from mindspore.common import set_seed

from src.config import config
from src.pose_resnet import GetPoseResNet
from src.network_with_loss import JointsMSELoss, PoseResNetWithLoss
from src.dataset import CreateDatasetCoco

if config.MODELARTS.IS_MODEL_ARTS:
    import moxing as mox

set_seed(config.GENERAL.TRAIN_SEED)
device_id = int(os.getenv('DEVICE_ID'))

def get_lr(begin_epoch,
           total_epochs,
           steps_per_epoch,
           lr_init=0.1,
           factor=0.1,
           epoch_number_to_drop=(90, 120)
           ):
    '''
    get_lr
    '''
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    step_number_to_drop = [steps_per_epoch * x for x in epoch_number_to_drop]
    for i in range(int(total_steps)):
        if i in step_number_to_drop:
            lr_init = lr_init * factor
        lr_each_step.append(lr_init)
    current_step = steps_per_epoch * begin_epoch
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate

def parse_args():
    parser = argparse.ArgumentParser(description="Simpleposenet training")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args = parser.parse_args()
    return args

def main():
    print("loading parse...")
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False,
                        device_id=device_id)

    if config.GENERAL.RUN_DISTRIBUTE:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    if config.MODELARTS.IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=args.data_url, dst_url=config.MODELARTS.CACHE_INPUT)

    dataset = CreateDatasetCoco(rank=rank,
                                group_size=device_num,
                                train_mode=True,
                                num_parallel_workers=config.TRAIN.NUM_PARALLEL_WORKERS,
                                )
    net = GetPoseResNet(config)
    loss = JointsMSELoss(config.LOSS.USE_TARGET_WEIGHT)
    net_with_loss = PoseResNetWithLoss(net, loss)
    dataset_size = dataset.get_dataset_size()
    lr = Tensor(get_lr(config.TRAIN.BEGIN_EPOCH,
                       config.TRAIN.END_EPOCH,
                       dataset_size,
                       lr_init=config.TRAIN.LR,
                       factor=config.TRAIN.LR_FACTOR,
                       epoch_number_to_drop=config.TRAIN.LR_STEP))
    opt = Adam(net.trainable_params(), learning_rate=lr)
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.TRAIN.SAVE_CKPT:
        config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size, keep_checkpoint_max=20)
        prefix = ''
        if config.GENERAL.RUN_DISTRIBUTE:
            prefix = 'multi_'
        else:
            prefix = 'single_'
        prefix = prefix + 'train_poseresnet_' + config.GENERAL.VERSION + '_' + os.getenv('DEVICE_ID')

        directory = ''
        if config.MODELARTS.IS_MODEL_ARTS:
            directory = config.MODELARTS.CACHE_OUTPUT
        else:
            directory = config.TRAIN.CKPT_PATH
        directory = directory + 'device_'+ os.getenv('DEVICE_ID')

        ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=directory, config=config_ck)
        cb.append(ckpoint_cb)
    model = Model(net_with_loss, loss_fn=None, optimizer=opt, amp_level="O2")
    epoch_size = config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH
    print("************ Start training now ************")
    print('start training, epoch size = %d' % epoch_size)
    model.train(epoch_size, dataset, callbacks=cb)

    if config.MODELARTS.IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=config.MODELARTS.CACHE_OUTPUT, dst_url=args.train_url)

if __name__ == '__main__':
    main()
