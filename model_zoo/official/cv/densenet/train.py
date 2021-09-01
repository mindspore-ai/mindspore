# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""train launch."""
import os
import time
import datetime
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_group_size
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from src.optimizers import get_param_groups
from src.losses.crossentropy import CrossEntropy
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.utils.logging import get_logger
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_rank_id

set_seed(1)


def modelarts_pre_process():
    pass


class BuildTrainNetwork(nn.Cell):
    """build training network"""

    def __init__(self, net, crit):
        super(BuildTrainNetwork, self).__init__()
        self.network = net
        self.criterion = crit

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


class ProgressMonitor(Callback):
    """monitor loss and time"""

    def __init__(self, configs):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.configs = configs
        self.ckpt_history = []

    def begin(self, run_context):
        self.configs.logger.info('start network train...')

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        """process epoch end"""
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.configs.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = self.configs.per_batch_size * (me_step - self.me_epoch_start_step_num) * \
                   self.configs.group_size / time_used
        self.configs.logger.info('epoch[{}], iter[{}], loss:{}, mean_fps:{:.2f} imgs/sec'.format(real_epoch
                                                                                                 , me_step,
                                                                                                 cb_params.net_outputs,
                                                                                                 fps_mean))
        if self.configs.rank_save_ckpt_flag:
            import glob
            ckpts = glob.glob(os.path.join(self.configs.outputs_dir, '*.ckpt'))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith('{}-'.format(self.configs.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.configs.logger.info('epoch[{}], iter[{}], loss:{}, ckpt:{},'
                                         'ckpt_fn:{}'.format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn))

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.configs.logger.info('end network train...')


def get_lr_scheduler(configs):
    if configs.lr_scheduler == 'exponential':
        lr_scheduler = MultiStepLR(configs.lr, configs.lr_epochs, configs.lr_gamma
                                   , configs.steps_per_epoch, configs.max_epoch, warmup_epochs=configs.warmup_epochs)
    elif config.lr_scheduler == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(configs.lr, configs.T_max, configs.steps_per_epoch, configs.max_epoch,
                                         warmup_epochs=configs.warmup_epochs, eta_min=configs.eta_min)
    else:
        raise NotImplementedError(configs.lr_scheduler)
    return lr_scheduler


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    """training process"""
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.image_size = list(map(int, config.image_size.split(',')))

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False)

    if config.device_target == 'Ascend':
        devid = get_device_id()
        context.set_context(device_id=devid)

    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank_id()
        config.group_size = get_group_size()

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
    config.outputs_dir = os.path.join(config.save_ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)

    if config.net == "densenet100":
        from src.network.densenet import DenseNet100 as DenseNet
    else:
        from src.network.densenet import DenseNet121 as DenseNet

    if config.dataset == "cifar10":
        from src.datasets import classification_dataset_cifar10 as classification_dataset
    else:
        from src.datasets import classification_dataset_imagenet as classification_dataset

    # dataloader
    de_dataset = classification_dataset(config.train_data_dir, config.image_size
                                        , config.per_batch_size, config.max_epoch, config.rank, config.group_size)
    config.steps_per_epoch = de_dataset.get_dataset_size()

    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')
    # get network and init
    network = DenseNet(config.num_classes)
    # loss
    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    criterion = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    # load pretrain model
    if os.path.isfile(config.train_pretrained):
        param_dict = load_checkpoint(config.train_pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        config.logger.info('load model %s success', str(config.train_predtrained))

    # lr scheduler
    lr_scheduler = get_lr_scheduler(config)
    lr_schedule = lr_scheduler.get_lr()

    # optimizer
    opt = Momentum(params=get_param_groups(network), learning_rate=Tensor(lr_schedule),
                   momentum=config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    # mixed precision training
    criterion.add_flags_recursive(fp32=True)

    # package training process, adjust lr + forward + backward + optimizer
    train_net = BuildTrainNetwork(network, criterion)
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE
    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=config.group_size,
                                      gradients_mean=True)

    if config.device_target == 'Ascend':
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O3")
    elif config.device_target == 'GPU':
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O0")
    elif config.device_target == 'CPU':
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O0")
    else:
        raise ValueError("Unsupported device target.")

    # checkpoint save
    progress_cb = ProgressMonitor(config)
    callbacks = [progress_cb,]
    if config.rank_save_ckpt_flag:
        ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=config.save_ckpt_path,
                                  prefix='%s' % config.rank)
        callbacks.append(ckpt_cb)

    model.train(config.max_epoch, de_dataset, callbacks=callbacks)


if __name__ == "__main__":
    train()
