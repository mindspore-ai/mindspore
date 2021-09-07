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
"""train imagenet"""
import time
import math
import os
import numpy as np

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inceptionv4 import Inceptionv4

from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.communication import init, get_rank, get_group_size
from mindspore.nn import RMSProp
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
set_seed(1)

DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}

config.device_id = get_device_id()
config.device_num = get_device_num()
device_num = config.device_num
create_dataset = DS_DICT[config.ds_type]


def generate_cosine_lr(steps_per_epoch, total_epochs,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       steps_per_epoch(int): steps number per epoch
       total_epochs(int): all epoch in training.
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
            lr = (lr_max - lr_end) * cosine_decay + lr_end
        lr_each_step.append(lr)
    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = steps_per_epoch * (config.start_epoch - 1)
    learning_rate = learning_rate[current_step:]
    return learning_rate


def modelarts_pre_process():
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print('Extract Start...')
                print('unzip file num: {}'.format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print('unzip percent: {}%'.format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print('cost time: {}min:{}s.'.format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print('Extract Done')
            else:
                print('This is not zip.')
        else:
            print('Zip has been extracted.')

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + '.zip')
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = '/tmp/unzip_sync.lock'

        # Each server contains 8 devices as most
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print('Zip file path: ', zip_file_1)
            print('Unzip file save dir: ', save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print('===Finish extract data synchronization===')
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print('Device: {}, Finish sync unzip data from {} to {}.'.format(get_device_id(), zip_file_1, save_dir_1))
        print('#' * 200, os.listdir(save_dir_1))
        print('#' * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

    config.dataset_path = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def inception_v4_train():
    """
    Train Inceptionv4 in data parallelism
    """
    print('epoch_size: {} batch_size: {} class_num {}'.format(config.epoch_size, config.batch_size, config.num_classes))

    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform)
    if config.platform == "Ascend":
        context.set_context(device_id=get_device_id())
        context.set_context(enable_graph_kernel=False)

    if device_num > 1:
        if config.platform == "Ascend":
            init(backend_name='hccl')
        elif config.platform == "GPU":
            init()
        else:
            raise ValueError("Unsupported device target.")

        config.rank = get_rank()
        config.group_size = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          all_reduce_fusion_config=[200, 400])
    else:
        config.rank = 0
        config.group_size = 1

    # create dataset
    train_dataset = create_dataset(dataset_path=config.dataset_path, do_train=True, cfg=config)
    train_step_size = train_dataset.get_dataset_size()

    # create model
    net = Inceptionv4(classes=config.num_classes)
    # loss
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # learning rate
    lr = Tensor(generate_cosine_lr(steps_per_epoch=train_step_size, total_epochs=config.epoch_size))

    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            param.set_data(initializer(XavierUniform(), param.data.shape, param.data.dtype))
    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    opt = RMSProp(group_params, lr, decay=config.decay, epsilon=config.epsilon, weight_decay=config.weight_decay,
                  momentum=config.momentum, loss_scale=config.loss_scale)

    if get_device_id() == 0:
        print(lr)
        print(train_step_size)
    if config.resume:
        ckpt = load_checkpoint(config.resume)
        load_param_into_net(net, ckpt)

    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc', 'top_1_accuracy', 'top_5_accuracy'},
                  loss_scale_manager=loss_scale_manager, amp_level=config.amp_level)

    # define callbacks
    performance_cb = TimeMonitor(data_size=train_step_size)
    loss_cb = LossMonitor(per_print_times=train_step_size)
    ckp_save_step = config.save_checkpoint_epochs * train_step_size
    config_ck = CheckpointConfig(save_checkpoint_steps=ckp_save_step, keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"inceptionV4-train-rank{config.rank}",
                                 directory='ckpts_rank_' + str(config.rank), config=config_ck)
    callbacks = [performance_cb, loss_cb]
    if device_num > 1 and config.is_save_on_master:
        if get_device_id() == 0:
            callbacks.append(ckpoint_cb)
    else:
        callbacks.append(ckpoint_cb)

    # train model
    model.train(config.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=config.ds_sink_mode)


if __name__ == '__main__':
    if config.ds_type == 'imagenet':
        config.dataset_path = os.path.join(config.dataset_path, 'train')
    inception_v4_train()
    print('Inceptionv4 training success!')
