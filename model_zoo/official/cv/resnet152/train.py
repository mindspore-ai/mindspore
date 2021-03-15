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
"""train resnet."""
import os
import argparse
import ast

from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank
from mindspore.common import set_seed
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.resnet import resnet152 as resnet
from src.config import config5 as config
from src.dataset import create_dataset2 as create_dataset          # imagenet2012

parser = argparse.ArgumentParser(description='Image classification--resnet152')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--is_save_on_master', type=ast.literal_eval, default=True, help='save ckpt on master or all rank')
args_opt = parser.parse_args()

set_seed(1)

if __name__ == '__main__':
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    print(args_opt.run_distribute)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

    if args_opt.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        rank_size = int(os.environ.get("RANK_SIZE", 1))
        print(rank_size)
        device_num = rank_size
        context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=[180, 313])
        init()
        args_opt.rank = get_rank()
    print(args_opt.rank)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args_opt.rank_save_ckpt_flag = 0
    if args_opt.is_save_on_master:
        if args_opt.rank == 0:
            args_opt.rank_save_ckpt_flag = 1
    else:
        args_opt.rank_save_ckpt_flag = 1
    local_data_path = args_opt.data_url

    local_data_path = args_opt.data_url
    print('Download data:')

    # create dataset
    dataset = create_dataset(dataset_path=local_data_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target="Ascend", distribute=args_opt.run_distribute)

    step_size = dataset.get_dataset_size()
    print("step"+str(step_size))

    # define net
    net = resnet(class_num=config.class_num)

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    # init lr
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)

    # define loss, model
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                  metrics={'top_1_accuracy', 'top_5_accuracy'},
                  amp_level="O3", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        if args_opt.rank_save_ckpt_flag:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="resnet152", directory=ckpt_save_dir, config=config_ck)
            cb += [ckpt_cb]

    # train model
    dataset_sink_mode = True
    print(dataset.get_dataset_size())
    model.train(config.epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
