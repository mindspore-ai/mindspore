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
"""
#################train vgg19 example on cifar10########################
"""
import datetime
import os
import time

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset

from src.crossentropy import CrossEntropy
from src.warmup_step_lr import warmup_step_lr
from src.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
from src.warmup_step_lr import lr_steps
from src.utils.logging import get_logger
from src.utils.util import get_param_groups
from src.vgg import vgg19

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

set_seed(1)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''run train'''
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.image_size = list(map(int, config.image_size.split(',')))
    config.per_batch_size = config.batch_size

    _enable_graph_kernel = config.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=_enable_graph_kernel, device_target=config.device_target)
    config.rank = get_rank_id()
    config.device_id = get_device_id()
    config.group_size = get_device_num()

    if config.is_distributed:
        if config.device_target == "Ascend":
            init()
            context.set_context(device_id=config.device_id)
        elif config.device_target == "GPU":
            if not config.enable_modelarts:
                init()
            else:
                if not config.need_modelarts_dataset_unzip:
                    init()

        device_num = config.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=[2, 18])
    else:
        if config.device_target == "Ascend":
            context.set_context(device_id=config.device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.ckpt_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)

    if config.dataset == "cifar10":
        dataset = vgg_create_dataset(config.data_dir, config.image_size, config.per_batch_size,
                                     config.rank, config.group_size)
    else:
        dataset = classification_dataset(config.data_dir, config.image_size, config.per_batch_size,
                                         config.rank, config.group_size)

    batch_num = dataset.get_dataset_size()
    config.steps_per_epoch = dataset.get_dataset_size()
    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')

    # get network and init
    network = vgg19(config.num_classes, config)

    # pre_trained
    if config.pre_trained:
        load_param_into_net(network, load_checkpoint(config.pre_trained))

    # lr scheduler
    if config.lr_scheduler == 'exponential':
        lr = warmup_step_lr(config.lr,
                            config.lr_epochs,
                            config.steps_per_epoch,
                            config.warmup_epochs,
                            config.max_epoch,
                            gamma=config.lr_gamma,
                            )
    elif config.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(config.lr,
                                        config.steps_per_epoch,
                                        config.warmup_epochs,
                                        config.max_epoch,
                                        config.T_max,
                                        config.eta_min)
    elif config.lr_scheduler == 'step':
        lr = lr_steps(0, lr_init=config.lr_init, lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                      total_epochs=config.max_epoch, steps_per_epoch=batch_num)
    else:
        raise NotImplementedError(config.lr_scheduler)

    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=config.momentum,
                   weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale)

    if config.dataset == "cifar10":
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(network, loss_fn=loss, optimizer=opt, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)
    else:
        if not config.label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O2")

    # define callbacks
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor(per_print_times=batch_num)
    callbacks = [time_cb, loss_cb]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval * config.steps_per_epoch,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank))
        callbacks.append(ckpt_cb)

    model.train(config.max_epoch, dataset, callbacks=callbacks)

if __name__ == '__main__':
    run_train()
