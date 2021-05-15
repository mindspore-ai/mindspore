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
"""AVA train on hpa dataset"""
import os
import argparse
import random
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint
from mindspore.nn import SGD, Adam, BCELoss
import mindspore.dataset.engine as de

from src.config import get_train_config, save_config, get_logger
from src.datasets import makeup_dataset

from src.resnet import resnet18, resnet50, resnet101
from src.network_define_train import WithLossCell, TrainOneStepCell
from src.network_define_eval import EvalCell, EvalMetric, EvalCallBack
from src.callbacks import LossCallBack
from src.lr_schedule import step_cosine_lr, cosine_lr

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument('--device_target', type=str, default="Ascend", help='Device target')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument("--load_ckpt_path", type=str, default="", help="path to load pretrain model")
parser.add_argument("--data_dir", type=str, default="", help="dataset directory")
parser.add_argument("--save_checkpoint_path", type=str, default="", help="path to save checkpoint")
parser.add_argument("--log_path", type=str, default="", help="path to save log file")
parser.add_argument("--save_eval_path", type=str, default=".", help='path to save eval result')
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = get_train_config()

    temp_path = ""
    save_checkpoint_path = os.path.join(args_opt.save_checkpoint_path,
                                        config.prefix + "/checkpoint" + config.time_prefix)
    save_checkpoint_path = os.path.join(temp_path, save_checkpoint_path)
    log_path = os.path.join(args_opt.log_path, config.prefix)
    log_path = os.path.join(temp_path, log_path)

    data_dir = args_opt.data_dir

    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = get_logger(os.path.join(log_path, 'log' + config.time_prefix + '.log'))

    device_id = args_opt.device_id
    device_num = args_opt.device_num

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=device_id)

    print("device num:{}".format(device_num))
    print("device id:{}".format(device_id))

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=False, parameter_broadcast=True, full_batch=False)
        init()
        temp_path = os.path.join(temp_path, str(device_id))
        print("temp path with multi-device:{}".format(temp_path))

    print("start create dataset...")

    epoch_for_dataset = config.epochs

    train_dataset = makeup_dataset(data_dir=data_dir, mode='train', batch_size=config.batch_size_for_train,
                                   bag_size=config.bag_size_for_train, shuffle=True, classes=config.classes,
                                   num_parallel_workers=config.num_parallel_workers)
    eval_dataset = makeup_dataset(data_dir=data_dir, mode='val', batch_size=config.batch_size_for_eval,
                                  bag_size=config.bag_size_for_eval, classes=config.classes,
                                  num_parallel_workers=config.num_parallel_workers)
    test_dataset = makeup_dataset(data_dir=data_dir, mode='test', batch_size=1, bag_size=20, classes=config.classes,
                                  num_parallel_workers=8)

    train_dataset.__loop_size__ = 1
    eval_dataset.__loop_size__ = 1
    test_dataset.__loop_size__ = 1

    train_dataset_batch_num = int(train_dataset.get_dataset_size())
    eval_dataset_batch_num = int(eval_dataset.get_dataset_size())
    test_dataset_batch_num = int(test_dataset.get_dataset_size())
    print("train dataset.get_dataset_size:{}".format(train_dataset.get_dataset_size()))
    print("eval dataset.get_dataset_size:{}".format(eval_dataset.get_dataset_size()))
    print("test dataset.get_dataset_size:{}".format(test_dataset.get_dataset_size()))

    print("the chosen network is {}".format(config.network))
    logger.info("the chosen network is %s", config.network)

    if config.network == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims, pretrain=False, classes=config.classes)
    elif config.network == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims, pretrain=False, classes=config.classes)
    elif config.network == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims, pretrain=False, classes=config.classes)
    else:
        raise "Unsupported net work!"
    if args_opt.load_ckpt_path != "":
        print("load checkpoint from {}".format(args_opt.load_ckpt_path))
        load_checkpoint(args_opt.load_ckpt_path, net=resnet)
    else:
        print("dont load checkpoint")

    if config.breakpoint_training_path != "":
        print("breakpoint training from :{}".format(config.breakpoint_training_path))
        load_checkpoint(config.breakpoint_training_path)

    loss = BCELoss(reduction='mean')

    net_with_loss = WithLossCell(resnet, loss)

    if config.lr_schedule == "cosine_lr":
        lr = Tensor(cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)
    else:
        lr = Tensor(step_cosine_lr(
            init_lr=config.base_lr,
            total_epochs=config.epochs,
            epoch_stage=config.epoch_stage,
            steps_per_epoch=train_dataset_batch_num,
            mode=config.lr_mode
        ), mstype.float32)

    if config.type == 'SGD':
        opt = SGD(params=net_with_loss.trainable_params(), learning_rate=lr, momentum=config.momentum,
                  weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    elif config.type == 'Adam':
        opt = Adam(params=net_with_loss.trainable_params(), learning_rate=lr, beta1=config.beta1,
                   beta2=config.beta2, weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if device_num > 1:
        net = TrainOneStepCell(net_with_loss, opt, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt)

    eval_network = EvalCell(resnet, loss)

    loss_cb = LossCallBack(data_size=train_dataset_batch_num, logger=logger)

    cb = [loss_cb]

    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * train_dataset_batch_num,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='AVA', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net, metrics={'results_return': EvalMetric(path=args_opt.save_eval_path)},
                  eval_network=eval_network)

    epoch_per_eval = {"epoch": [], "f1_macro": [], "f1_micro": [], "auc": [], "val_loss": []}

    eval_cb = EvalCallBack(model=model, eval_dataset=eval_dataset, eval_per_epoch=config.eval_per_epoch,
                           epoch_per_eval=epoch_per_eval, logger=logger)
    cb += [eval_cb]

    print("save configs...")
    # save current config
    config_name = 'config.json'
    save_config([os.path.join(save_checkpoint_path, config_name)], config, vars(args_opt))

    print("training begins...")
    model.train(config.epochs, train_dataset, callbacks=cb, dataset_sink_mode=False)
    logger.info("Eval on test dataset ...")
    res = model.eval(test_dataset)
    print("Eval result:{}".format(res))
    logger.info("Eval result:%s", res)
