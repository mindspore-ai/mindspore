# Copyright 2019 Huawei Technologies Co., Ltd
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
#################train vgg16 example on cifar10########################
python train.py --data_path=$DATA_HOME --device_id=$DEVICE_ID
"""
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore import context
import numpy as np
import argparse
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import dataset
from mindspore.model_zoo.vgg import vgg16
from config import cifar_cfg as cfg
import random
random.seed(1)
np.random.seed(1)

def lr_steps(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        elif i < decay_epoch_index[1]:
            lr = lr_max * 0.1
        elif i < decay_epoch_index[2]:
            lr = lr_max * 0.01
        else:
            lr = lr_max * 0.001
        lr_each_step.append(lr)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cifar10 classification')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--data_path', type=str, default='./cifar', help='path where the dataset is saved')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target != 'CPU' and args_opt.device_id:
        context.set_context(device_id=args_opt.device_id)
    context.set_context(enable_task_sink = True, enable_loop_sink = True)
    context.set_context(enable_mem_reuse=True, enable_hccl=False)

    net = vgg16(batch_size=cfg.batch_size, num_classes=cfg.num_classes)
    lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size, steps_per_epoch=50000 // cfg.batch_size)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)

    model = Model(net, loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False), optimizer=opt, metrics={'acc'})

    dataset = dataset.create_dataset(args_opt.data_path, cfg.epoch_size)
    batch_num = dataset.get_dataset_size()
    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="train_vgg_cifar10", directory="./", config=config_ck)
    loss_cb = LossMonitor()
    model.train(cfg.epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])
