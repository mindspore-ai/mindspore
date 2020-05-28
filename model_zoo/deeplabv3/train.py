#!/bin/bash
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
"""train."""
import os, time
import argparse
from mindspore import context
from mindspore import log as logger
from mindspore.communication.management import init
import mindspore.nn as nn
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import Model, ParallelMode
import argparse
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback,CheckpointConfig, ModelCheckpoint, TimeMonitor
from src.md_dataset import create_dataset
from src.losses import OhemLoss
from src.deeplabv3 import deeplabv3_resnet50
from src.config import config

parser = argparse.ArgumentParser(description="Deeplabv3 training")
parser.add_argument("--distribute", type=str, default="false", help="Run distribute, default is false.")
parser.add_argument('--epoch_size', type=int, default=2, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
parser.add_argument('--data_url', required=True, default=None, help='Train data url')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument('--checkpoint_url', default=None, help='Checkpoint path')
parser.add_argument("--enable_save_ckpt", type=str, default="true", help="Enable save checkpoint, default is true.")
parser.add_argument('--max_checkpoint_num', type=int, default=5, help='Max checkpoint number.')
parser.add_argument("--save_checkpoint_steps", type=int, default=1000, help="Save checkpoint steps, "
                                                                                "default is 1000.")
parser.add_argument("--save_checkpoint_num", type=int, default=1, help="Save checkpoint numbers, default is 1.")
args_opt = parser.parse_args()
print(args_opt)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
class LossCallBack(Callback):
    """
    Monitor the loss in training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self._per_print_times = per_print_times
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)))
def model_fine_tune(flags, net, fix_weight_layer):
    checkpoint_path = flags.checkpoint_url
    if checkpoint_path is None:
        return
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(net, param_dict)
    for para in net.trainable_params():
        if fix_weight_layer in para.name:
            para.requires_grad=False
if __name__ == "__main__":
    if args_opt.distribute == "true":
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True)
        init()
    args_opt.base_size = config.crop_size
    args_opt.crop_size = config.crop_size
    train_dataset = create_dataset(args_opt, args_opt.data_url, args_opt.epoch_size, args_opt.batch_size, usage="train")   
    dataset_size = train_dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    callback = [time_cb, LossCallBack()]
    if args_opt.enable_save_ckpt == "true":
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                     keep_checkpoint_max=args_opt.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_deeplabv3', config=config_ck)
        callback.append(ckpoint_cb)
    net =  deeplabv3_resnet50(crop_size.seg_num_classes, [args_opt.batch_size,3,args_opt.crop_size,args_opt.crop_size],
                                     infer_scale_sizes=crop_size.eval_scales, atrous_rates=crop_size.atrous_rates,
                                     decoder_output_stride=crop_size.decoder_output_stride, output_stride = crop_size.output_stride,
                                     fine_tune_batch_norm=crop_size.fine_tune_batch_norm, image_pyramid = crop_size.image_pyramid)
    net.set_train()
    model_fine_tune(args_opt, net, 'layer')
    loss = OhemLoss(crop_size.seg_num_classes, crop_size.ignore_label)
    opt = Momentum(filter(lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'depth' not in x.name and 'bias' not in x.name, net.trainable_params()), learning_rate=args_opt.learning_rate, momentum=args_opt.momentum, weight_decay=args_opt.weight_decay)
    model = Model(net, loss, opt)
    model.train(args_opt.epoch_size, train_dataset, callback)