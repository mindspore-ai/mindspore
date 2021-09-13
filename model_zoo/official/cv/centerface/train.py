# Copyright 2020-21 Huawei Technologies Co., Ltd
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
"""
Train centerface and get network model files(.ckpt)
"""

import os
import time
import datetime
import numpy as np

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.optim.adam import Adam
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim.sgd import SGD
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.profiler.profiling import Profiler
from mindspore.common import set_seed

from src.utils import AverageMeter
from src.lr_scheduler import warmup_step_lr
from src.lr_scheduler import warmup_cosine_annealing_lr, \
    warmup_cosine_annealing_lr_v2, warmup_cosine_annealing_lr_sample
from src.lr_scheduler import MultiStepLR
from src.var_init import default_recurisive_init
from src.centerface import CenterfaceMobilev2
from src.utils import load_backbone, get_param_groups

from src.centerface import CenterFaceWithLossCell, TrainingWrapper
from src.dataset import GetDataLoader
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


set_seed(1)
dev_id = get_device_id()
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                    save_graphs=False, device_id=dev_id, reserve_class_name_in_scope=False)


if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.t_max:
    config.t_max = config.max_epoch

config.lr_epochs = list(map(int, config.lr_epochs.split(',')))


def convert_training_shape(args_):
    """
    Convert training shape
    """
    training_shape = [int(args_.training_shape), int(args_.training_shape)]
    return training_shape


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, para_name):
        return self[para_name]

    def __setattr__(self, para_name, para_value):
        self[para_name] = para_value


def modelarts_pre_process():
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_centerface():
    pass


if __name__ == "__main__":
    train_centerface()
    print('\ntrain.py config:\n', config)
    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(
        config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    if config.need_profiler:
        profiler = Profiler(output_path=config.outputs_dir)

    loss_meter = AverageMeter('loss')

    context.reset_auto_parallel_context()
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    else:
        parallel_mode = ParallelMode.STAND_ALONE
        degree = 1

    # Notice: parameter_broadcast should be supported, but current version has bugs, thus been disabled.
    # To make sure the init weight on all npu is the same, we need to set a static seed in default_recurisive_init when weight initialization
    context.set_auto_parallel_context(
        parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
    network = CenterfaceMobilev2()
    # init, to avoid overflow, some std of weight should be enough small
    default_recurisive_init(network)

    if config.pretrained_backbone:
        network = load_backbone(network, config.pretrained_backbone, config)
        print(
            'load pre-trained backbone {} into network'.format(config.pretrained_backbone))
    else:
        print('Not load pre-trained backbone, please be careful')

    if os.path.isfile(config.resume):
        param_dict = load_checkpoint(config.resume)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.') or key.startswith('moment1.') or key.startswith('moment2.'):
                continue
            elif key.startswith('centerface_network.'):
                param_dict_new[key[19:]] = values
            else:
                param_dict_new[key] = values

        load_param_into_net(network, param_dict_new)
        print('load_model {} success'.format(config.resume))
    else:
        print('{} not set/exists or not a pre-trained file'.format(config.resume))

    network = CenterFaceWithLossCell(network)
    print('finish get network')

    # -------------reset config-----------------
    if config.training_shape:
        config.multi_scale = [convert_training_shape(config)]

    # data loader
    data_loader, config.steps_per_epoch = GetDataLoader(per_batch_size=config.per_batch_size,
                                                        max_epoch=config.max_epoch, rank=config.rank,
                                                        group_size=config.group_size,
                                                        config=config, split='train')
    config.steps_per_epoch = config.steps_per_epoch // config.max_epoch
    print('Finish loading dataset')

    if not config.ckpt_interval:
        config.ckpt_interval = config.steps_per_epoch

    # lr scheduler
    if config.lr_scheduler == 'multistep':
        lr_fun = MultiStepLR(config.lr, config.lr_epochs, config.lr_gamma, config.steps_per_epoch, config.max_epoch,
                             config.warmup_epochs)
        lr = lr_fun.get_lr()
    elif config.lr_scheduler == 'exponential':
        lr = warmup_step_lr(config.lr,
                            config.lr_epochs,
                            config.steps_per_epoch,
                            config.warmup_epochs,
                            config.max_epoch,
                            gamma=config.lr_gamma
                            )
    elif config.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(config.lr,
                                        config.steps_per_epoch,
                                        config.warmup_epochs,
                                        config.max_epoch,
                                        config.t_max,
                                        config.eta_min)
    elif config.lr_scheduler == 'cosine_annealing_V2':
        lr = warmup_cosine_annealing_lr_v2(config.lr,
                                           config.steps_per_epoch,
                                           config.warmup_epochs,
                                           config.max_epoch,
                                           config.t_max,
                                           config.eta_min)
    elif config.lr_scheduler == 'cosine_annealing_sample':
        lr = warmup_cosine_annealing_lr_sample(config.lr,
                                               config.steps_per_epoch,
                                               config.warmup_epochs,
                                               config.max_epoch,
                                               config.t_max,
                                               config.eta_min)
    else:
        raise NotImplementedError(config.lr_scheduler)

    if config.optimizer == "adam":
        opt = Adam(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale)
        print("use adam optimizer")
    elif config.optimizer == "sgd":
        opt = SGD(params=get_param_groups(network),
                  learning_rate=Tensor(lr),
                  momentum=config.momentum,
                  weight_decay=config.weight_decay,
                  loss_scale=config.loss_scale)
    else:
        opt = Momentum(params=get_param_groups(network),
                       learning_rate=Tensor(lr),
                       momentum=config.momentum,
                       weight_decay=config.weight_decay,
                       loss_scale=config.loss_scale)

    network = TrainingWrapper(network, opt, sens=config.loss_scale)
    network.set_train()

    if config.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=config.outputs_dir,
                                  prefix='{}'.format(config.rank))
        cb_params = InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

        print('config.steps_per_epoch = {} config.ckpt_interval ={}'.format(config.steps_per_epoch,
                                                                            config.ckpt_interval))

    t_end = time.time()

    for i_all, batch_load in enumerate(data_loader):
        i = i_all % config.steps_per_epoch
        epoch = i_all // config.steps_per_epoch + 1
        images, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks = batch_load

        images = Tensor(images)
        hm = Tensor(hm)
        reg_mask = Tensor(reg_mask)
        ind = Tensor(ind)
        wh = Tensor(wh)
        wight_mask = Tensor(wight_mask)
        hm_offset = Tensor(hm_offset)
        hps_mask = Tensor(hps_mask)
        landmarks = Tensor(landmarks)

        loss, overflow, scaling = network(
            images, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks)
        # Tensor to numpy
        overflow = np.all(overflow.asnumpy())
        loss = loss.asnumpy()
        loss_meter.update(loss)
        print('epoch:{}, iter:{}, avg_loss:{}, loss:{}, overflow:{}, loss_scale:{}'.format(
            epoch, i, loss_meter, loss, overflow, scaling.asnumpy()))

        if config.rank_save_ckpt_flag:
            # ckpt progress
            cb_params.cur_epoch_num = epoch
            cb_params.cur_step_num = i + 1 + (epoch-1)*config.steps_per_epoch
            cb_params.batch_num = i + 2 + (epoch-1)*config.steps_per_epoch
            ckpt_cb.step_end(run_context)

        if (i_all+1) % config.steps_per_epoch == 0:
            time_used = time.time() - t_end
            fps = config.per_batch_size * config.steps_per_epoch * config.group_size / time_used
            if config.rank == 0:
                print(
                    'epoch[{}], {}, {:.2f} imgs/sec, lr:{}'
                    .format(epoch, loss_meter, fps, lr[i + (epoch-1)*config.steps_per_epoch])
                )
            t_end = time.time()
            loss_meter.reset()

    if config.need_profiler:
        profiler.analyse()

    print('==========end training===============')
