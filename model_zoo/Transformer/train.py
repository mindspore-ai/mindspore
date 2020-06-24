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
"""Transformer training script."""

import time
import argparse
import random
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine as de
import mindspore.communication.management as D
from mindspore.train.parallel_utils import ParallelMode
from mindspore import context

from src.transformer_for_train import TransformerTrainOneStepCell, TransformerNetworkWithLoss, \
                                      TransformerTrainOneStepWithLossScaleCell
from src.config import cfg, transformer_net_cfg
from src.dataset import create_transformer_dataset
from src.lr_schedule import create_dynamic_lr

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
de.config.set_seed(random_seed)

def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))
time_stamp_init = False
time_stamp_first = 0

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss.log", "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                           cb_params.cur_epoch_num,
                                                                           cb_params.cur_step_num,
                                                                           str(cb_params.net_outputs)))
            f.write('\n')


def argparse_init():
    """
    Argparse init.
    """
    parser = argparse.ArgumentParser(description='transformer')
    parser.add_argument("--distribute", type=str, default="false", help="Run distribute, default is false.")
    parser.add_argument("--epoch_size", type=int, default=52, help="Epoch size, default is 52.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--enable_lossscale", type=str, default="true", help="Use lossscale or not, default is true.")
    parser.add_argument("--do_shuffle", type=str, default="true", help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="false", help="Enable data sink, default is false.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint file path")
    parser.add_argument("--enable_save_ckpt", type=str, default="true", help="Enable save checkpoint, "
                                                                             "default is true.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=2500, help="Save checkpoint steps, "
                                                                                "default is 2500.")
    parser.add_argument("--save_checkpoint_num", type=int, default=30, help="Save checkpoint numbers, default is 30.")
    parser.add_argument("--save_checkpoint_path", type=str, default="./checkpoint/", help="Save checkpoint file path, "
                                                                                          "default is ./checkpoint/")
    parser.add_argument("--data_path", type=str, default="", help="Data path, it is better to use absolute path")
    return parser

def run_transformer_train():
    """
    Transformer training.
    """
    parser = argparse_init()
    args, _ = parser.parse_known_args()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
    context.set_context(reserve_class_name_in_scope=False, enable_auto_mixed_precision=False)

    if args.distribute == "true":
        device_num = args.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                          parameter_broadcast=True, device_num=device_num)
        D.init()
        rank_id = args.device_id % device_num
    else:
        device_num = 1
        rank_id = 0
    dataset, repeat_count = create_transformer_dataset(epoch_count=args.epoch_size, rank_size=device_num,
                                                       rank_id=rank_id, do_shuffle=args.do_shuffle,
                                                       enable_data_sink=args.enable_data_sink,
                                                       dataset_path=args.data_path)

    netwithloss = TransformerNetworkWithLoss(transformer_net_cfg, True)

    if args.checkpoint_path:
        parameter_dict = load_checkpoint(args.checkpoint_path)
        load_param_into_net(netwithloss, parameter_dict)

    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size()*args.epoch_size,
                                  learning_rate=cfg.lr_schedule.learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  hidden_size=transformer_net_cfg.hidden_size,
                                  start_decay_step=cfg.lr_schedule.start_decay_step,
                                  min_lr=cfg.lr_schedule.min_lr), mstype.float32)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack()]
    if args.enable_save_ckpt == "true":
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,
                                       keep_checkpoint_max=args.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='transformer', directory=args.save_checkpoint_path, config=ckpt_config)
        callbacks.append(ckpoint_cb)

    if args.enable_lossscale == "true":
        scale_manager = DynamicLossScaleManager(init_loss_scale=cfg.init_loss_scale_value,
                                                scale_factor=cfg.scale_factor,
                                                scale_window=cfg.scale_window)
        update_cell = scale_manager.get_update_cell()
        netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                                scale_update_cell=update_cell)
    else:
        netwithgrads = TransformerTrainOneStepCell(netwithloss, optimizer=optimizer)

    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(repeat_count, dataset, callbacks=callbacks, dataset_sink_mode=(args.enable_data_sink == "true"))

if __name__ == '__main__':
    run_transformer_train()
