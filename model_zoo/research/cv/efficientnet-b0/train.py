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
"""train efficientnet."""
import os
import ast
import argparse

from mindspore import context
from mindspore import Tensor
from mindspore.nn import SGD, RMSProp
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import dtype as mstype
from mindspore.common import set_seed

from src.lr_generator import get_lr
from src.models.effnet import EfficientNet
from src.config import config
from src.monitor import Monitor
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training')
    # modelarts parameter
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')

    # Ascend parameter
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')

    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run mode')
    parser.add_argument('--resume', type=str, default='', help='resume training with existed checkpoint')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

    # init distributed
    if args_opt.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel', gradients_mean=True)
            local_data_url = os.path.join(local_data_url, str(device_id))
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
    else:
        if args_opt.run_distribute:
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id)
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            context.set_context(device_id=args_opt.device_id)
            device_num = 1
            device_id = 0

    # define network
    net = EfficientNet(1, 1)
    net.to_float(mstype.float16)

    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define dataset
    if args_opt.run_modelarts:
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    step_size = dataset.get_dataset_size()

    # resume
    if args_opt.resume:
        ckpt = load_checkpoint(args_opt.resume)
        load_param_into_net(net, ckpt)

    # get learning rate
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size,
                       lr_decay_mode=config.lr_decay_mode))

    # define optimization
    if config.opt == 'sgd':
        optimizer = SGD(net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                        weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    elif config.opt == 'rmsprop':
        optimizer = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                            momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)

    # define model
    model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                  metrics={'acc'}, amp_level='O3')

    # define callbacks
    cb = [Monitor(lr_init=lr.asnumpy())]
    if config.save_checkpoint and (device_num == 1 or device_id == 0):
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        if args_opt.run_modelarts:
            ckpt_cb = ModelCheckpoint(f"Efficientnet_b0-rank{device_id}", directory=local_train_url, config=config_ck)
        else:
            save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model_' + str(device_id) + '/')
            ckpt_cb = ModelCheckpoint(f"Efficientnet_b0-rank{device_id}", directory=save_ckpt_path, config=config_ck)
        cb += [ckpt_cb]

    # begine train
    model.train(config.epoch_size, dataset, callbacks=cb)
    if args_opt.run_modelarts and config.save_checkpoint and (device_num == 1 or device_id == 0):
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
