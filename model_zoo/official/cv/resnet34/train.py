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
#################train resnet34 example on imagenet2012########################
python train.py
"""
import os
import ast
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.optim import Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init

from src.resnet import resnet34 as resnet
from src.config import config
from src.dataset import create_dataset
from src.lr_generator import get_linear_lr as get_lr
from src.cross_entropy_smooth import CrossEntropySmooth

set_seed(1)

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mindspore resnet34 example')
    parser.add_argument('--modelart', required=True, type=ast.literal_eval, default=False,
                        help='training on modelart or not, default is False')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--ckpt_url', required=True, default=None, help='Location of ckpt.')
    args = parser.parse_args()

    target = "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target=target,
                        device_id=int(os.environ["DEVICE_ID"]))

    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)

    dataset_sink_mode = True

    if args.modelart:
        import moxing as mox
        data_path = '/cache/data_path'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=data_path)
        tar_command = "tar -xvf /cache/data_path/imagenet_original.tar.gz -C /cache/data_path/"
        os.system(tar_command)
        data_path = '/cache/data_path/imagenet_original/'
    else:
        data_path = args.data_url
    data_path_train = os.path.join(data_path, 'train')

    # create dataset
    dataset_train = create_dataset(dataset_path=data_path_train, do_train=True,
                                   batch_size=config.batch_size)

    step_size = dataset_train.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)

    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

    # define loss function
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)
    loss_scale = FixedLossScaleManager(loss_scale=config.loss_scale, drop_overflow_update=False)

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size)
    lr = Tensor(lr)

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

    # define model
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                  metrics={'top_1_accuracy', 'top_5_accuracy'},
                  amp_level="O2", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()

    cb = [time_cb, loss_cb]

    ckpt_save_dir = config.save_checkpoint_path

    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    model.train(epoch=config.epoch_size, train_dataset=dataset_train, callbacks=cb,
                dataset_sink_mode=dataset_sink_mode)

    if args.modelart:
        import moxing as mox
        mox.file.copy_parallel(src_url=ckpt_save_dir, dst_url=args.ckpt_url)
