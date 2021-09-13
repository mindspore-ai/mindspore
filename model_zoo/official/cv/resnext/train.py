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
"""train ImageNet."""
import os
import time
import datetime

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, Callback
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.common import set_seed

from src.dataset import classification_dataset
from src.crossentropy import CrossEntropy
from src.lr_generator import get_lr
from src.utils.logging import get_logger
from src.utils.optimizers__init__ import get_param_groups
from src.utils.var_init import load_pretrain_model
from src.image_classification import get_network
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

set_seed(1)

class BuildTrainNetwork(nn.Cell):
    """build training network"""
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss

class ProgressMonitor(Callback):
    """monitor loss and time"""
    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args
        self.ckpt_history = []

    def begin(self, run_context):
        self.args.logger.info('start network train...')

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = self.args.per_batch_size * (me_step-self.me_epoch_start_step_num) * self.args.group_size / time_used
        self.args.logger.info('epoch[{}], iter[{}], loss:{}, mean_fps:{:.2f}'
                              'imgs/sec'.format(real_epoch, me_step, cb_params.net_outputs, fps_mean))

        if self.args.rank_save_ckpt_flag:
            import glob
            ckpts = glob.glob(os.path.join(self.args.outputs_dir, '*.ckpt'))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith('{}-'.format(self.args.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.args.logger.info('epoch[{}], iter[{}], loss:{}, ckpt:{},'
                                      'ckpt_fn:{}'.format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn))


        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info('end network train...')


def set_parameters():
    """parameters"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    # init distributed
    if config.run_distribute:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1

    if config.is_dynamic_loss_scale == 1:
        config.loss_scale = 1  # for dynamic loss scale can not set loss scale in momentum opt

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.output_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    return config

def set_graph_kernel_context(device_target):
    if device_target == "GPU":
        context.set_context(enable_graph_kernel=True)

@moxing_wrapper()
def train():
    """training process"""
    set_parameters()
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    set_graph_kernel_context(config.device_target)

    # init distributed
    if config.run_distribute:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=config.group_size,
                                          gradients_mean=True)
    # dataloader
    de_dataset = classification_dataset(config.data_path, config.image_size,
                                        config.per_batch_size, 1,
                                        config.rank, config.group_size, num_parallel_workers=8)
    config.steps_per_epoch = de_dataset.get_dataset_size()

    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')
    # get network and init
    network = get_network(network=config.network, num_classes=config.num_classes, platform=config.device_target)

    load_pretrain_model(config.checkpoint_file_path, network, config)

    # lr scheduler
    lr = get_lr(config)

    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=config.momentum,
                   weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale)


    # loss
    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager,
                  metrics={'acc'}, amp_level="O3")

    # checkpoint save
    progress_cb = ProgressMonitor(config)
    callbacks = [progress_cb,]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval * config.steps_per_epoch,
                                       keep_checkpoint_max=config.ckpt_save_max)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank))
        callbacks.append(ckpt_cb)

    model.train(config.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == "__main__":
    train()
