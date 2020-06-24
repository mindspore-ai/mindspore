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
"""train_imagenet."""
import os
import argparse
import numpy as np
from dataset import create_dataset
from lr_generator import get_lr
from config import config
from mindspore import context
from mindspore import Tensor
from mindspore.model_zoo.resnet import resnet50
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindspore.train.model import Model, ParallelMode

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
args_opt = parser.parse_args()


if __name__ == '__main__':
    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    np.random.seed(1)
    if not args_opt.do_eval and args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id,
                                enable_auto_mixed_precision=True)
            init()
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            auto_parallel_context().set_all_reduce_fusion_split_indices([107, 160])
            ckpt_save_dir = config.save_checkpoint_path
        elif target == "GPU":
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"
    epoch_size = config.epoch_size
    net = resnet50(class_num=config.class_num)

    if args_opt.do_train:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True,
                                 repeat_num=epoch_size, batch_size=config.batch_size, target=target)
        step_size = dataset.get_dataset_size()

        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        lr = Tensor(get_lr(global_step=0, lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                           warmup_epochs=config.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                           lr_decay_mode='poly'))
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                       config.weight_decay, config.loss_scale)
        if target == 'GPU':
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, is_grad=False, reduction='mean')
            opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum)
            model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
        else:
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                          amp_level="O2", keep_batchnorm_fp32=False)

        time_cb = TimeMonitor(data_size=step_size)
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs*step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
            cb += [ckpt_cb]
        model.train(epoch_size, dataset, callbacks=cb)
