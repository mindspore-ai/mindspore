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
"""
######################## train alexnet example ########################
train alexnet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.dataset import create_dataset_cifar10, create_dataset_imagenet
from src.generator_lr import get_lr_cifar10, get_lr_imagenet
from src.alexnet import AlexNet
from src.get_param_groups import get_param_groups

import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import dataset as de
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed

set_seed(1)
de.config.set_seed(1)

def modelarts_pre_process():
    pass
    # config.ckpt_path = os.path.join(config.output_path, str(get_rank_id()), config.checkpoint_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def train_alexnet():
    print(config)
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())

    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(save_graphs=False)
    if device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    device_num = get_device_num()
    if config.dataset_name == "cifar10":
        if device_num > 1:
            config.learning_rate = config.learning_rate * device_num
            config.epoch_size = config.epoch_size * 2
    elif config.dataset_name == "imagenet":
        pass
    else:
        raise ValueError("Unsupported dataset.")

    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, \
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif device_target == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())

    if config.dataset_name == "cifar10":
        ds_train = create_dataset_cifar10(config, config.data_path, config.batch_size, target=config.device_target)
    elif config.dataset_name == "imagenet":
        ds_train = create_dataset_imagenet(config, config.data_path, config.batch_size)
    else:
        raise ValueError("Unsupported dataset.")

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = AlexNet(config.num_classes, phase='train')

    loss_scale_manager = None
    metrics = None
    step_per_epoch = ds_train.get_dataset_size() if config.sink_size == -1 else config.sink_size
    if config.dataset_name == 'cifar10':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_cifar10(0, config.learning_rate, config.epoch_size, step_per_epoch))
        opt = nn.Momentum(network.trainable_params(), lr, config.momentum)
        metrics = {"Accuracy": Accuracy()}

    elif config.dataset_name == 'imagenet':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_imagenet(config.learning_rate, config.epoch_size, step_per_epoch))
        opt = nn.Momentum(params=get_param_groups(network),
                          learning_rate=lr,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay,
                          loss_scale=config.loss_scale)

        from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
        if config.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    else:
        raise ValueError("Unsupported dataset.")

    if device_target == "Ascend":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)
    elif device_target == "GPU":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2",
                      loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Unsupported platform.")

    if device_num > 1:
        ckpt_save_dir = os.path.join(config.ckpt_path + "_" + str(get_rank()))
    else:
        ckpt_save_dir = config.ckpt_path

    time_cb = TimeMonitor(data_size=step_per_epoch)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=ckpt_save_dir, config=config_ck)

    print("============== Starting Training ==============")
    model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=config.dataset_sink_mode, sink_size=config.sink_size)

if __name__ == "__main__":
    train_alexnet()
