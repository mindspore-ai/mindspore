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
Model training entrypoint.
"""

import os

from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, \
    LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.config import Config
from src.dataset import create_cifar10_dataset
from src.network import WRN
from src.optim import get_lr
from src.utils import init_utils


if __name__ == '__main__':
    conf = Config(training=True)
    init_utils(conf)
    set_seed(conf.seed)

    # Initialize context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=conf.device_target,
        save_graphs=False,
    )
    if conf.run_distribute:
        if conf.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(
                device_id=device_id
            )
            context.set_auto_parallel_context(
                device_num=conf.device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
            init()
        elif conf.device_target == 'GPU':
            init()
            context.set_auto_parallel_context(
                device_num=get_group_size(),
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
    else:
        try:
            device_id = int(os.getenv('DEVICE_ID'))
        except TypeError:
            device_id = 0
        context.set_context(device_id=device_id)

    # Create dataset
    if conf.dataset == 'cifar10':
        dataset = create_cifar10_dataset(
            dataset_path=conf.dataset_path,
            do_train=True,
            repeat_num=1,
            batch_size=conf.batch_size,
            target=conf.device_target,
            distribute=conf.run_distribute,
            augment=conf.augment,
        )
    step_size = dataset.get_dataset_size()

    # Define net
    net = WRN(160, 3, conf.class_num)

    # Load weight if pre_trained is configured
    if conf.pre_trained:
        param_dict = load_checkpoint(conf.pre_trained)
        load_param_into_net(net, param_dict)

    # Initialize learning rate
    lr = Tensor(get_lr(
        lr_init=conf.lr_init, lr_max=conf.lr_max,
        warmup_epochs=conf.warmup_epochs, total_epochs=conf.epoch_size,
        steps_per_epoch=step_size, lr_decay_mode=conf.lr_decay_mode,
    ))

    # Define loss, opt, and model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(
        conf.loss_scale,
        drop_overflow_update=False,
    )
    opt = Momentum(
        filter(lambda x: x.requires_grad, net.get_parameters()),
        lr, conf.momentum, conf.weight_decay, conf.loss_scale,
    )
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, metrics={'acc'})

    # Define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(
        save_checkpoint_steps=conf.save_checkpoint_epochs * step_size,
        keep_checkpoint_max=conf.keep_checkpoint_max,
    )
    ck_cb = ModelCheckpoint(
        prefix='train_%s_%s' % (conf.net, conf.dataset),
        directory=conf.save_checkpoint_path,
        config=config_ck,
    )

    # Train
    if conf.run_distribute:
        callbacks = [time_cb, loss_cb]
        if conf.device_target == 'GPU' and str(get_rank()) == '0':
            callbacks = [time_cb, loss_cb, ck_cb]
        elif conf.device_target == 'Ascend' and device_id == 0:
            callbacks = [time_cb, loss_cb, ck_cb]
    else:
        callbacks = [time_cb, loss_cb, ck_cb]

    model.train(conf.epoch_size, dataset, callbacks=callbacks)
