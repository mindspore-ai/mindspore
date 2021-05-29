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
train hardnet
"""
import os
import math
import argparse
import ast
from mindspore import context
from mindspore import Tensor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init

from src.dataset import create_dataset_ImageNet
from src.lr_scheduler import get_lr
from src.HarDNet import HarDNet85
from src.EntropyLoss import CrossEntropySmooth
from src.config import config

parser = argparse.ArgumentParser(description='Image classification with HarDNet on Imagenet')

parser.add_argument('--dataset_path', type=str, default='/home/hardnet/imagenet_original/train/',
                    help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_num', type=int, default=8, help='Device num')
parser.add_argument('--pre_trained', type=str, default=True)
parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)
parser.add_argument('--pre_ckpt_path', type=str, default='/home/work/user-job-dir/hardnet/src/HarDNet85.ckpt')
parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='label_smooth_factor')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=True)
parser.add_argument('--distribute', type=ast.literal_eval, default=True)
parser.add_argument('--device_id', type=int, default=0, help='device_id')

args = parser.parse_args()

if args.isModelArts:
    import moxing as mox

if __name__ == '__main__':
    target = args.device_target

    if args.distribute:
        # init context
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
        context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()

    else:
        device_id = args.device_id
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target,
                            save_graphs=False,
                            device_id=args.device_id)

    if args.isModelArts:
        import moxing as mox
        # download dataset from obs to cache
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID')
        # create dataset
        train_dataset = create_dataset_ImageNet(dataset_path=train_dataset_path,
                                                do_train=True,
                                                repeat_num=1,
                                                batch_size=config.batch_size,
                                                target=target)
    else:
        train_dataset = create_dataset_ImageNet(dataset_path=args.dataset_path,
                                                do_train=True,
                                                repeat_num=1,
                                                batch_size=config.batch_size,
                                                target=target)

    step_size = train_dataset.get_dataset_size()

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define net
    network = HarDNet85(num_classes=config.class_num)
    print("----network----")

    # init weight
    if args.pre_trained:
        param_dict = load_checkpoint(args.pre_ckpt_path)
        load_param_into_net(network, param_dict)
    else:
        for _, cell in network.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(gain=1 / math.sqrt(3)),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)
            if isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(weight_init.initializer('zeros', cell.beta.shape))
            if isinstance(cell, nn.Dense):
                cell.bias.default_input = weight_init.initializer('zeros', cell.bias.shape, cell.bias.dtype)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in network.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': network.trainable_params()}]

    net_opt = nn.Momentum(group_params, lr, config.momentum,
                          weight_decay=config.weight_decay,
                          loss_scale=config.loss_scale)
    # define loss
    loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                              num_classes=config.class_num)

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(network, loss_fn=loss, optimizer=net_opt,
                  loss_scale_manager=loss_scale, metrics={'acc'}, amp_level="O3")

    # define callbacks
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        if args.isModelArts:
            save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
        else:
            save_checkpoint_path = config.save_checkpoint_path

        ckpt_cb = ModelCheckpoint(prefix="HarDNet85",
                                  directory=save_checkpoint_path,
                                  config=config_ck)
        cb += [ckpt_cb]

    print("\n\n========================")
    print("Dataset path: {}".format(args.dataset_path))
    print("Total epoch: {}".format(config.epoch_size))
    print("Batch size: {}".format(config.batch_size))
    print("Class num: {}".format(config.class_num))
    print("======= Multiple Training begin========")
    model.train(config.epoch_size, train_dataset,
                callbacks=cb, dataset_sink_mode=True)
    if args.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args.train_url)
