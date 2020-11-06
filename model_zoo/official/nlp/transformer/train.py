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

import os
import time
import argparse
import ast

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.common import set_seed

from src.transformer_for_train import TransformerTrainOneStepCell, TransformerNetworkWithLoss, \
                                      TransformerTrainOneStepWithLossScaleCell
from src.config import cfg, transformer_net_cfg, transformer_net_cfg_gpu
from src.dataset import create_transformer_dataset
from src.lr_schedule import create_dynamic_lr

set_seed(1)

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
    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')


def argparse_init():
    """
    Argparse init.
    """
    parser = argparse.ArgumentParser(description='transformer')
    parser.add_argument("--distribute", type=str, default="false", choices=['true', 'false'],
                        help="Run distribute, default is false.")
    parser.add_argument("--epoch_size", type=int, default=52, help="Epoch size, default is 52.")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="device where the code will be implemented, default is Ascend")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--enable_lossscale", type=str, default="true", choices=['true', 'false'],
                        help="Use lossscale or not, default is true.")
    parser.add_argument("--do_shuffle", type=str, default="true", choices=['true', 'false'],
                        help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint file path")
    parser.add_argument("--enable_save_ckpt", type=str, default="true", choices=['true', 'false'],
                        help="Enable save checkpoint, default is true.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=2500, help="Save checkpoint steps, "
                                                                                "default is 2500.")
    parser.add_argument("--save_checkpoint_num", type=int, default=30, help="Save checkpoint numbers, default is 30.")
    parser.add_argument("--save_checkpoint_path", type=str, default="./", help="Save checkpoint file path")
    parser.add_argument("--data_path", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--bucket_boundaries", type=ast.literal_eval, default=[16, 32, 48, 64, 128],
                        help="sequence length for different bucket")

    return parser

def run_transformer_train():
    """
    Transformer training.
    """
    parser = argparse_init()
    args, _ = parser.parse_known_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    context.set_context(reserve_class_name_in_scope=False, enable_auto_mixed_precision=False)

    if args.distribute == "true":
        if args.device_target == "Ascend":
            device_num = args.device_num
            D.init('hccl')
        else:
            D.init('nccl')
            device_num = D.get_group_size()
            rank = get_rank()
            args.device_id = rank
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        rank_id = args.device_id % device_num
        save_ckpt_path = os.path.join(args.save_checkpoint_path, 'ckpt_' + str(get_rank()) + '/')
    else:
        device_num = 1
        rank_id = 0
        save_ckpt_path = os.path.join(args.save_checkpoint_path, 'ckpt_0/')
    dataset = create_transformer_dataset(epoch_count=1, rank_size=device_num,
                                         rank_id=rank_id, do_shuffle=args.do_shuffle,
                                         dataset_path=args.data_path,
                                         bucket_boundaries=args.bucket_boundaries,
                                         device_target=args.device_target)
    if args.device_target == "Ascend":
        netwithloss = TransformerNetworkWithLoss(transformer_net_cfg, True)
    else:
        netwithloss = TransformerNetworkWithLoss(transformer_net_cfg_gpu, True)

    if args.checkpoint_path:
        parameter_dict = load_checkpoint(args.checkpoint_path)
        load_param_into_net(netwithloss, parameter_dict)

    hidden_size = transformer_net_cfg.hidden_size if args.device_target == "Ascend" \
        else transformer_net_cfg_gpu.hidden_size
    learning_rate = cfg.lr_schedule.learning_rate if args.device_target == "Ascend" \
        else 1.0
    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size()*args.epoch_size,
                                  learning_rate=learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  hidden_size=hidden_size,
                                  start_decay_step=cfg.lr_schedule.start_decay_step,
                                  min_lr=cfg.lr_schedule.min_lr), mstype.float32)

    if args.device_target == "GPU" and cfg.transformer_network == "large":
        optimizer = Adam(netwithloss.trainable_params(), lr, beta2=cfg.optimizer_adam_beta2)
    else:
        optimizer = Adam(netwithloss.trainable_params(), lr)

    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(rank_id=rank_id)]
    if args.enable_save_ckpt == "true":
        if device_num == 1 or (device_num > 1 and rank_id == 0):
            if args.device_target == "Ascend":
                ckpt_config = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,
                                               keep_checkpoint_max=args.save_checkpoint_num)
            else:
                ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(),
                                               keep_checkpoint_max=args.save_checkpoint_num)
            ckpoint_cb = ModelCheckpoint(prefix='transformer', directory=save_ckpt_path, config=ckpt_config)
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

    model.train(args.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=False)

if __name__ == '__main__':
    run_transformer_train()
