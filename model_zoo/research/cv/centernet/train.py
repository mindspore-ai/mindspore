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
"""
Train CenterNet and get network model files(.ckpt)
"""

import os
import argparse
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Adam
from mindspore import log as logger
from mindspore.common import set_seed
from mindspore.profiler import Profiler
from src.dataset import COCOHP
from src import CenterNetMultiPoseLossCell, CenterNetWithLossScaleCell
from src import CenterNetWithoutLossScaleCell
from src.utils import LossCallBack, CenterNetPolynomialDecayLR, CenterNetMultiEpochsDecayLR
from src.config import dataset_config, net_config, train_config

_current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='CenterNet training')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'CPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                    help="Run distribute, default is false.")
parser.add_argument("--need_profiler", type=str, default="false", choices=["true", "false"],
                    help="Profiling to parsing runtime info, default is false.")
parser.add_argument("--profiler_path", type=str, default=" ", help="The path to save profiling data")
parser.add_argument("--epoch_size", type=int, default="1", help="Epoch size, default is 1.")
parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1,"
                    "i.e. run all steps according to epoch number.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--enable_save_ckpt", type=str, default="true", choices=["true", "false"],
                    help="Enable save checkpoint, default is true.")
parser.add_argument("--do_shuffle", type=str, default="true", choices=["true", "false"],
                    help="Enable shuffle for dataset, default is true.")
parser.add_argument("--enable_data_sink", type=str, default="true", choices=["true", "false"],
                    help="Enable data sink, default is true.")
parser.add_argument("--data_sink_steps", type=int, default="1", help="Sink steps for each epoch, default is 1.")
parser.add_argument("--save_checkpoint_path", type=str, default="", help="Save checkpoint path")
parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
parser.add_argument("--save_checkpoint_steps", type=int, default=1000, help="Save checkpoint steps, default is 1000.")
parser.add_argument("--save_checkpoint_num", type=int, default=1, help="Save checkpoint numbers, default is 1.")
parser.add_argument("--mindrecord_dir", type=str, default="", help="Mindrecord dataset files directory")
parser.add_argument("--mindrecord_prefix", type=str, default="coco_hp.train.mind",
                    help="Prefix of MindRecord dataset filename.")
parser.add_argument("--visual_image", type=str, default="false", help="Visulize the ground truth and predicted image")
parser.add_argument("--save_result_dir", type=str, default="", help="The path to save the predict results")

args_opt = parser.parse_args()


def _set_parallel_all_reduce_split():
    """set centernet all_reduce fusion split"""
    if net_config.last_level == 5:
        context.set_auto_parallel_context(all_reduce_fusion_config=[16, 56, 96, 136, 175])
    elif net_config.last_level == 6:
        context.set_auto_parallel_context(all_reduce_fusion_config=[18, 59, 100, 141, 182])
    else:
        raise ValueError("The total num of allreduced grads for last level = {} is unknown,"
                         "please re-split after known the true value".format(net_config.last_level))


def _get_params_groups(network, optimizer):
    """
    Get param groups
    """
    params = network.trainable_params()
    decay_params = list(filter(lambda x: not optimizer.decay_filter(x), params))
    other_params = list(filter(optimizer.decay_filter, params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]
    return group_params


def _get_optimizer(network, dataset_size):
    """get optimizer, only support Adam right now."""
    if train_config.optimizer == 'Adam':
        group_params = _get_params_groups(network, train_config.Adam)
        if train_config.lr_schedule == "PolyDecay":
            lr_schedule = CenterNetPolynomialDecayLR(learning_rate=train_config.PolyDecay.learning_rate,
                                                     end_learning_rate=train_config.PolyDecay.end_learning_rate,
                                                     warmup_steps=train_config.PolyDecay.warmup_steps,
                                                     decay_steps=args_opt.train_steps,
                                                     power=train_config.PolyDecay.power)
            optimizer = Adam(group_params, learning_rate=lr_schedule, eps=train_config.PolyDecay.eps, loss_scale=1.0)
        elif train_config.lr_schedule == "MultiDecay":
            multi_epochs = train_config.MultiDecay.multi_epochs
            if not isinstance(multi_epochs, (list, tuple)):
                raise TypeError("multi_epochs must be list or tuple.")
            if not multi_epochs:
                multi_epochs = [args_opt.epoch_size]
            lr_schedule = CenterNetMultiEpochsDecayLR(learning_rate=train_config.MultiDecay.learning_rate,
                                                      warmup_steps=train_config.MultiDecay.warmup_steps,
                                                      multi_epochs=multi_epochs,
                                                      steps_per_epoch=dataset_size,
                                                      factor=train_config.MultiDecay.factor)
            optimizer = Adam(group_params, learning_rate=lr_schedule, eps=train_config.MultiDecay.eps, loss_scale=1.0)
        else:
            raise ValueError("Don't support lr_schedule {}, only support [PolynormialDecay, MultiEpochDecay]".
                             format(train_config.optimizer))
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, Adam]".
                         format(train_config.optimizer))
    return optimizer


def train():
    """training CenterNet"""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(reserve_class_name_in_scope=False)
    context.set_context(save_graphs=False)

    ckpt_save_dir = args_opt.save_checkpoint_path
    rank = 0
    device_num = 1
    num_workers = 8
    if args_opt.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=False)
        context.set_context(device_id=args_opt.device_id)
        if args_opt.distribute == "true":
            D.init()
            device_num = args_opt.device_num
            rank = args_opt.device_id % device_num
            ckpt_save_dir = args_opt.save_checkpoint_path + 'ckpt_' + str(get_rank()) + '/'

            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            _set_parallel_all_reduce_split()
    else:
        args_opt.distribute = "false"
        args_opt.need_profiler = "false"
        args_opt.enable_data_sink = "false"

    # Start create dataset!
    # mindrecord files will be generated at args_opt.mindrecord_dir such as centernet.mindrecord0, 1, ... file_num.
    logger.info("Begin creating dataset for CenterNet")
    coco = COCOHP(dataset_config, run_mode="train", net_opt=net_config,
                  enable_visual_image=(args_opt.visual_image == "true"), save_path=args_opt.save_result_dir)
    dataset = coco.create_train_dataset(args_opt.mindrecord_dir, args_opt.mindrecord_prefix,
                                        batch_size=train_config.batch_size, device_num=device_num, rank=rank,
                                        num_parallel_workers=num_workers, do_shuffle=args_opt.do_shuffle == 'true')
    dataset_size = dataset.get_dataset_size()
    logger.info("Create dataset done!")

    net_with_loss = CenterNetMultiPoseLossCell(net_config)

    new_repeat_count = args_opt.epoch_size * dataset_size // args_opt.data_sink_steps
    if args_opt.train_steps > 0:
        new_repeat_count = min(new_repeat_count, args_opt.train_steps // args_opt.data_sink_steps)
    else:
        args_opt.train_steps = args_opt.epoch_size * dataset_size
        logger.info("train steps: {}".format(args_opt.train_steps))

    optimizer = _get_optimizer(net_with_loss, dataset_size)

    enable_static_time = args_opt.device_target == "CPU"
    callback = [TimeMonitor(args_opt.data_sink_steps), LossCallBack(dataset_size, enable_static_time)]
    if args_opt.enable_save_ckpt == "true" and args_opt.device_id % min(8, device_num) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                     keep_checkpoint_max=args_opt.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_centernet',
                                     directory=None if ckpt_save_dir == "" else ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)

    if args_opt.load_checkpoint_path:
        param_dict = load_checkpoint(args_opt.load_checkpoint_path)
        load_param_into_net(net_with_loss, param_dict)
    if args_opt.device_target == "Ascend":
        net_with_grads = CenterNetWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                    sens=train_config.loss_scale_value)
    else:
        net_with_grads = CenterNetWithoutLossScaleCell(net_with_loss, optimizer=optimizer)

    model = Model(net_with_grads)
    model.train(new_repeat_count, dataset, callbacks=callback, dataset_sink_mode=(args_opt.enable_data_sink == "true"),
                sink_size=args_opt.data_sink_steps)


if __name__ == '__main__':
    if args_opt.need_profiler == "true":
        profiler = Profiler(output_path=args_opt.profiler_path)
    set_seed(0)
    train()
    if args_opt.need_profiler == "true":
        profiler.analyse()
