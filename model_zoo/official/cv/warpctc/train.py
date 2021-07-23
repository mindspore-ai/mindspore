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
"""Warpctc training"""
import os
import time
import math as m
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_group_size, get_rank

from src.loss import CTCLoss
from src.dataset import create_dataset
from src.warpctc import StackedRNN, StackedRNNForGPU, StackedRNNForCPU
from src.warpctc_for_train import TrainOneStepCellWithGradClip
from src.lr_schedule import get_lr

from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

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
                print("Unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("Unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("Cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.modelarts_dataset_unzip_name:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))
    config.save_checkpoint_path = os.path.join(config.output_path, config.save_checkpoint_path)
    config.train_data_dir = config.data_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    """Train function."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == 'Ascend':
        context.set_context(device_id=get_device_id())
    lr_scale = 1
    if config.run_distribute:
        if config.device_target == 'Ascend':
            device_num = get_device_num()
            rank = get_rank_id()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        else:
            init()
            device_num = get_group_size()
            rank = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    else:
        device_num = 1
        rank = 0
    enable_graph_kernel = config.device_target == 'GPU'
    context.set_context(enable_graph_kernel=enable_graph_kernel)

    max_captcha_digits = config.max_captcha_digits
    input_size = m.ceil(config.captcha_height / 64) * 64 * 3
    # create dataset
    dataset_dir = config.train_data_dir
    dataset = create_dataset(dataset_path=dataset_dir, batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank, device_target=config.device_target)
    step_size = dataset.get_dataset_size()
    # define lr
    lr_init = config.learning_rate if not config.run_distribute else config.learning_rate * device_num * lr_scale
    lr = get_lr(config.epoch_size, step_size, lr_init)
    loss = CTCLoss(max_sequence_length=config.captcha_width,
                   max_label_length=max_captcha_digits,
                   batch_size=config.batch_size)
    if config.device_target == 'Ascend':
        net = StackedRNN(input_size=input_size, batch_size=config.batch_size, hidden_size=config.hidden_size)
    elif config.device_target == 'GPU':
        net = StackedRNNForGPU(input_size=input_size, batch_size=config.batch_size, hidden_size=config.hidden_size)
    else:
        net = StackedRNNForCPU(input_size=input_size, batch_size=config.batch_size, hidden_size=config.hidden_size)
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum)

    net = WithLossCell(net, loss)
    net = TrainOneStepCellWithGradClip(net, opt, device_num).set_train()
    # define model
    model = Model(net)
    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.save_checkpoint_path, str(rank))
        ckpt_cb = ModelCheckpoint(prefix="warpctc", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpt_cb)
    model.train(config.epoch_size, dataset, callbacks=callbacks)


if __name__ == '__main__':
    train()
