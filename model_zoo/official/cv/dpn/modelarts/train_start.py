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
"""Train DPN"""

import os
import argparse
import glob
import ast
import moxing as mox
import numpy as np
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
# from ast import literal_eval
from mindspore import context
from mindspore import Tensor
from mindspore import export
from mindspore.nn import SGD
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_group_size
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.imagenet_dataset import classification_dataset
from src.dpn import dpns
from src.lr_scheduler import get_lr_drop, get_lr_warmup
from src.crossentropy import CrossEntropy
from src.callbacks import SaveCallback

set_seed(1)

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)

def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)

def _parse_args():
    parser = argparse.ArgumentParser('dpn training args')

    # url for modelarts
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')

    # local path & device configs
    parser.add_argument('--data_path', type=str, default='/cache/data',
                        help='The location of input data')
    parser.add_argument('--output_path', type=str, default='/cache/train',
                        help='The location of the output file')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='device id of GPU or Ascend. (Default: Ascend)')
    parser.add_argument('--is_distributed', type=int, default=0,
                        help='distributed training')
    parser.add_argument('--group_size', type=int, default=1,
                        help='world size of distributed')

    # dataset & model configs
    parser.add_argument('--image_size_height', type=int, default=224,
                        help='height of model input image')
    parser.add_argument('--image_size_width', type=int, default=224,
                        help='width of model input image')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_parallel_workers', type=int, default=1,
                        help='number of parallel workers')
    parser.add_argument('--dataset', type=str, default='imagenet-1K',
                        help='train dataset')
    parser.add_argument('--keep_checkpoint_max', type=int, default=3,
                        help='keep checkpoint max')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--label_smooth', action='store_true',
                        help='label smooth')
    parser.add_argument('--label_smooth_factor', type=float, default=0.0,
                        help='label smooth factor')
    parser.add_argument('--backbone', type=str, default='dpn92',
                        help='backbone net chose from dpn92, dpn98, dpn131, dpn107.')
    parser.add_argument('--is_save_on_master', action='store_false',
                        help='is_save_on_master')

    # training configs
    parser.add_argument('--load_path', type=str, default='/cache/data/pretrained_dir',
                        help='The location of input data')
    parser.add_argument('--pretrained_ckpt', type=str, default='',
                        help='pretrained checkpoint file name')
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--eval_each_epoch', type=int, default=0,
                        help='evaluate on each epoch')
    parser.add_argument('--global_step', type=int, default=0,
                        help='global step')
    parser.add_argument('--epoch_size', type=int, default=1,
                        help='epoch size')
    parser.add_argument('--loss_scale_num', type=int, default=1024,
                        help='loss scale num')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='momentum')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_dir',
                        help='dircetory to save checkpoint')

    # learning rate configs
    parser.add_argument('--lr_schedule', type=str, default='warmup',
                        help='lr_schedule stratage')
    parser.add_argument('--lr_init', type=float, default=0.001,
                        help='init learning rate')
    parser.add_argument('--lr_max', type=float, default=0.1,
                        help='max learning rate')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='decay factor')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='warmup epochs')

    # export configs
    parser.add_argument('--export_dir', type=str, default='',
                        help='dircetory to save exported model, frozen model if not None')
    parser.add_argument('--width', type=int, default=224,
                        help='export width')
    parser.add_argument('--height', type=int, default=224,
                        help='export height')
    parser.add_argument('--file_name', type=str, default='dpn92',
                        help='export file name')
    parser.add_argument('--file_format', type=str, default='AIR',
                        help='export file format')
    # args, _ = parser.parse_known_args()
    _args = parser.parse_args()
    return _args

def filter_weight_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def dpn_train(config_args, ma_config):
    ma_config["training_data"] = config_args.data_path + "/train"
    ma_config["image_size"] = [config_args.image_size_height, config_args.image_size_width]
    train_dataset = classification_dataset(ma_config["training_data"],
                                           image_size=ma_config["image_size"],
                                           per_batch_size=config_args.batch_size,
                                           max_epoch=1,
                                           num_parallel_workers=config_args.num_parallel_workers,
                                           shuffle=True,
                                           rank=ma_config["rank"],
                                           group_size=config_args.group_size)
    if config_args.eval_each_epoch:
        print("create eval_dataset")
        ma_config["eval_data_dir"] = config_args.data_path + "/val"
        eval_dataset = classification_dataset(ma_config["eval_data_dir"],
                                              image_size=ma_config["image_size"],
                                              per_batch_size=config_args.batch_size,
                                              max_epoch=1,
                                              num_parallel_workers=config_args.num_parallel_workers,
                                              shuffle=False,
                                              rank=ma_config["rank"],
                                              group_size=config_args.group_siz,
                                              mode='eval')

    train_step_size = train_dataset.get_dataset_size()

    # choose net
    net = dpns[config_args.backbone](num_classes=config_args.num_classes)

    # load checkpoint
    ckpt_path = os.path.join(config_args.load_path, config_args.pretrained_ckpt)
    if os.path.isfile(ckpt_path):
        print("load ckpt:", ckpt_path)
        param_dict = load_checkpoint(ckpt_path)
        if config_args.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_weight_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    # learing rate schedule
    if config_args.lr_schedule == 'drop':
        print("lr_schedule:drop")
        lr = Tensor(get_lr_drop(global_step=config_args.global_step,
                                total_epochs=config_args.epoch_size,
                                steps_per_epoch=train_step_size,
                                lr_init=config_args.lr_init,
                                factor=config_args.factor))
    elif config_args.lr_schedule == 'warmup':
        print("lr_schedule:warmup")
        lr = Tensor(get_lr_warmup(global_step=config_args.global_step,
                                  total_epochs=config_args.epoch_size,
                                  steps_per_epoch=train_step_size,
                                  lr_init=config_args.lr_init,
                                  lr_max=config_args.lr_max,
                                  warmup_epochs=config_args.warmup_epochs))

    # optimizer
    opt = SGD(net.trainable_params(),
              lr,
              momentum=config_args.momentum,
              weight_decay=config_args.weight_decay,
              loss_scale=config_args.loss_scale_num)
    # loss scale
    loss_scale = FixedLossScaleManager(config_args.loss_scale_num, False)
    # loss function
    if config_args.dataset == "imagenet-1K":
        print("Use SoftmaxCrossEntropyWithLogits")
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        if not config_args.label_smooth:
            ma_config["label_smooth_factor"] = 0.0
        print("Use Label_smooth CrossEntropy")
        loss = CrossEntropy(smooth_factor=ma_config["label_smooth_factor"], num_classes=config_args.num_classes)
    # create model
    model = Model(net, amp_level="O2",
                  keep_batchnorm_fp32=False,
                  loss_fn=loss,
                  optimizer=opt,
                  loss_scale_manager=loss_scale,
                  metrics={'top_1_accuracy', 'top_5_accuracy'})

    # loss/time monitor & ckpt save callback
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=train_step_size)
    cb = [loss_cb, time_cb]
    if ma_config["rank_save_ckpt_flag"]:
        if config_args.eval_each_epoch:
            save_cb = SaveCallback(model, eval_dataset, ma_config["checkpoint_path"])
            cb += [save_cb]
        else:
            config_ck = CheckpointConfig(save_checkpoint_steps=train_step_size,
                                         keep_checkpoint_max=config_args.keep_checkpoint_max)
            ckpoint_cb = ModelCheckpoint(prefix="dpn", directory=ma_config["checkpoint_path"], config=config_ck)
            cb.append(ckpoint_cb)
    # train model
    model.train(config_args.epoch_size, train_dataset, callbacks=cb)
    return 0

def dpn_export(config_args, ma_config):
    backbone = config_args.backbone
    num_classes = config_args.num_classes
    net = dpns[backbone](num_classes=num_classes)

    # load checkpoint
    prob_ckpt_list = os.path.join(ma_config["checkpoint_path"], "dpn*.ckpt")
    ckpt_list = glob.glob(prob_ckpt_list)
    if not ckpt_list:
        print('Freezing model failed!')
        print("can not find ckpt files. ")
    else:
        ckpt_list.sort(key=os.path.getmtime)
        ckpt_name = ckpt_list[-1]
        print("checkpoint file name", ckpt_name)
        param_dict = load_checkpoint(ckpt_name)
        load_param_into_net(net, param_dict)
        net.set_train(False)

        image = Tensor(np.zeros([config_args.batch_size, 3, config_args.height, config_args.width], np.float32))
        export_path = os.path.join(config_args.output_path, config_args.export_dir)
        if not os.path.exists(export_path):
            os.makedirs(export_path, exist_ok=True)
        file_name = os.path.join(export_path, config_args.file_name)
        export(net, image, file_name=file_name, file_format=config_args.file_format)
        print('Freezing model success!')
    return 0


def main():
    config_args = _parse_args()
    # create local path
    if not os.path.exists(config_args.data_path):
        os.makedirs(config_args.data_path, exist_ok=True)
    if not os.path.exists(config_args.output_path):
        os.makedirs(config_args.output_path, exist_ok=True)
    ma_config = {}
    # init context
    ma_config["checkpoint_path"] = os.path.join(config_args.output_path, config_args.checkpoint_dir)
    if not os.path.exists(ma_config["checkpoint_path"]):
        os.makedirs(ma_config["checkpoint_path"], exist_ok=True)
    ma_config["device_id"] = get_device_id()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config_args.device_target, save_graphs=False, device_id=ma_config["device_id"])
    # init distributed
    ma_config["rank"] = 0
    if config_args.is_distributed:
        init()
        ma_config["rank"] = get_rank_id()
        ma_config["group_size"] = get_group_size()
        ma_config["device_num"] = get_device_num()
        context.set_auto_parallel_context(device_num=ma_config["device_num"], parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    # select for master rank save ckpt or all rank save, compatible for model parallel
    ma_config["rank_save_ckpt_flag"] = 0
    if config_args.is_save_on_master:
        if ma_config["rank"] == 0:
            ma_config["rank_save_ckpt_flag"] = 1
    else:
        ma_config["rank_save_ckpt_flag"] = 1
    # data sync
    mox.file.copy_parallel(config_args.data_url, config_args.data_path)
    # train
    dpn_train(config_args, ma_config)
    print('DPN training success!')
    # export
    if config_args.export_dir:
        dpn_export(config_args, ma_config)

    # data sync
    mox.file.copy_parallel(config_args.output_path, config_args.train_url)
    return 0

if __name__ == '__main__':
    main()
    print('Modelarts testing success!')
