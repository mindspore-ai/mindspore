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
"""AVA training on cifar."""

import os
import argparse
import random
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint
import mindspore.dataset.engine as de

from src.optimizer import SGD_ as SGD
from src.config import get_config, save_config, get_logger
from src.datasets import get_train_dataset
from src.cifar_resnet import resnet18, resnet50, resnet101
from src.network_define import WithLossCell, TrainOneStepCell
from src.callbacks import LossCallBack
from src.loss import LossNet
from src.lr_schedule import step_cosine_lr, cosine_lr

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA training")
parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default is false.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--device_target", type=str, default="Ascend", help="Device target")
parser.add_argument("--train_data_dir", type=str, default="", help="training dataset directory")
parser.add_argument("--test_data_dir", type=str, default="", help="testing dataset directory")
parser.add_argument("--save_checkpoint_path", type=str, default="", help="path to save checkpoint")
parser.add_argument("--log_path", type=str, default="", help="path to save log file")
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    temp_path = ''

    device_id = args_opt.device_id
    device_num = args_opt.device_num
    print("device num:{}".format(device_num))
    print("device id:{}".format(device_id))

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=device_id)

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=False, parameter_broadcast=True)
        init()
        temp_path = os.path.join(temp_path, str(device_id))
        print("temp path with multi-device:{}".format(temp_path))

    save_checkpoint_path = os.path.join(args_opt.save_checkpoint_path,
                                        config.prefix + "/checkpoint" + config.time_prefix)
    save_checkpoint_path = os.path.join(temp_path, save_checkpoint_path)
    log_path = os.path.join(args_opt.log_path, config.prefix)
    log_path = os.path.join(temp_path, log_path)
    print(log_path)
    train_data_dir = os.path.join(temp_path, args_opt.train_data_dir)
    test_data_dir = os.path.join(temp_path, args_opt.test_data_dir)

    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = get_logger(os.path.join(log_path, 'log' + config.time_prefix + '.log'))

    print("start create dataset...")

    train_dataset = get_train_dataset(train_data_dir=train_data_dir, batchsize=config.batch_size,
                                      epoch=1, device_id=device_id, device_num=device_num)
    train_dataset.__loop_size__ = 1

    train_dataset_batch_num = int(train_dataset.get_dataset_size())

    print("the chosen network is {}".format(config.net_work))
    logger.info("the chosen network is %s", config.net_work)

    if config.net_work == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    elif config.net_work == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    elif config.net_work == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    else:
        raise "net work config error!!!"

    if config.breakpoint_training_path != "":
        print("breakpoint training from :{}".format(config.breakpoint_training_path))
        load_checkpoint(config.breakpoint_training_path)

    loss = LossNet(temp=config.sigma)

    net_with_loss = WithLossCell(resnet, loss)

    if config.lr_schedule == "cosine_lr":
        lr = Tensor(cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode,
            warmup_epoch=config.warmup_epoch
        ), mstype.float32)
    else:
        lr = Tensor(step_cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            epoch_stage=config.epoch_stage,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode,
            warmup_epoch=config.warmup_epoch
        ), mstype.float32)

    opt = SGD(params=net_with_loss.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if device_num > 1:
        net = TrainOneStepCell(net_with_loss, opt, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt)

    loss_cb = LossCallBack(data_size=train_dataset_batch_num, logger=logger)

    cb = [loss_cb]

    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=train_dataset_batch_num,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='AVA',
                                     directory=save_checkpoint_path,
                                     config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)

    print("save configs...")
    # save current config file
    config_name = 'config.json'
    save_config([os.path.join(save_checkpoint_path, config_name)], config, vars(args_opt))

    print("training begins...")

    print("model description:{}".format(config.description))
    dataset_sink_mode = bool(args_opt.device_target == "Ascend")
    model.train(config.epochs, train_dataset, callbacks=cb, dataset_sink_mode=dataset_sink_mode)
