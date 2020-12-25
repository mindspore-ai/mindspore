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
import numpy as np

from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.optim import Adam
from mindspore.common import set_seed

from src.config import config
from src.model import get_pose_net
from src.network_define import JointsMSELoss, WithLossCell
from src.dataset import keypoint_dataset

set_seed(1)
device_id = int(os.getenv('DEVICE_ID'))


def get_lr(begin_epoch,
           total_epochs,
           steps_per_epoch,
           lr_init=0.1,
           factor=0.1,
           epoch_number_to_drop=(90, 120)
           ):
    """
    Generate learning rate array.

    Args:
        begin_epoch (int): Initial epoch of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.316.
        factor:Factor of lr to drop.
        epoch_number_to_drop:Learing rate will drop after these epochs.
    Returns:
        np.array, learning rate array.
    """
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
    parser.add_argument("--run-distribute",
                        help="Run distribute, default is false.",
                        action='store_true')
    parser.add_argument('--ckpt-path', type=str, help='ckpt path to save')
    parser.add_argument('--batch-size', type=int, help='training batch size')
    args = parser.parse_args()
    return args


def main():
    # load parse and config
    print("loading parse...")
    args = parse_args()
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    print('batch size :{}'.format(config.TRAIN.BATCH_SIZE))

    # distribution and context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False,
                        device_id=device_id)

    if args.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    # only rank = 0 can write
    rank_save_flag = False
    if rank == 0 or device_num == 1:
        rank_save_flag = True

        # create dataset
    dataset, _ = keypoint_dataset(config,
                                  rank=rank,
                                  group_size=device_num,
                                  train_mode=True,
                                  num_parallel_workers=8)

    # network
    net = get_pose_net(config, True, ckpt_path=config.MODEL.PRETRAINED)
    loss = JointsMSELoss(use_target_weight=True)
    net_with_loss = WithLossCell(net, loss)

    # lr schedule and optim
    dataset_size = dataset.get_dataset_size()
    lr = Tensor(get_lr(config.TRAIN.BEGIN_EPOCH,
                       config.TRAIN.END_EPOCH,
                       dataset_size,
                       lr_init=config.TRAIN.LR,
                       factor=config.TRAIN.LR_FACTOR,
                       epoch_number_to_drop=config.TRAIN.LR_STEP))
    opt = Adam(net.trainable_params(), learning_rate=lr)

    # callback
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if args.ckpt_path and rank_save_flag:
        config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size, keep_checkpoint_max=20)
        ckpoint_cb = ModelCheckpoint(prefix="simplepose", directory=args.ckpt_path, config=config_ck)
        cb.append(ckpoint_cb)
        # train model
    model = Model(net_with_loss, loss_fn=None, optimizer=opt, amp_level="O2")
    epoch_size = config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH
    print('start training, epoch size = %d' % epoch_size)
    model.train(epoch_size, dataset, callbacks=cb)


if __name__ == '__main__':
    main()
