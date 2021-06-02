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
#################pre_train bert example on zh-wiki########################
python run_pretrain.py
"""

import argparse
import os
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.nn.optim import thor
from src import BertNetworkWithLoss, BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell
from src.dataset import create_bert_dataset
from src.config import cfg, bert_net_cfg
from src.utils import LossCallBack


_current_dir = os.path.dirname(os.path.realpath(__file__))


def _set_bert_all_reduce_split():
    """set bert all_reduce fusion split, support num_hidden_layers is 12 and 24."""
    from mindspore.parallel._auto_parallel_context import auto_parallel_context
    if bert_net_cfg.num_hidden_layers == 12:
        if bert_net_cfg.use_relative_positions:
            auto_parallel_context().set_all_reduce_fusion_split_indices([29, 58, 87, 116, 145, 174, 203, 217],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([29, 58, 87, 116, 145, 174, 203, 217],
                                                                        "hccl_world_groupsum3")
        else:
            auto_parallel_context().set_all_reduce_fusion_split_indices([28, 55, 82, 109, 136, 163, 190, 205],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([28, 55, 82, 109, 136, 163, 190, 205],
                                                                        "hccl_world_groupsum3")
    elif bert_net_cfg.num_hidden_layers == 24:
        if bert_net_cfg.use_relative_positions:
            auto_parallel_context().set_all_reduce_fusion_split_indices([30, 90, 150, 210, 270, 330, 390, 421],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([30, 90, 150, 210, 270, 330, 390, 421],
                                                                        "hccl_world_groupsum3")
        else:
            auto_parallel_context().set_all_reduce_fusion_split_indices([38, 93, 148, 203, 258, 313, 368, 397],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([38, 93, 148, 203, 258, 313, 368, 397],
                                                                        "hccl_world_groupsum3")


def _get_optimizer(args_opt, network):
    """get thor optimizer."""
    if cfg.optimizer == "Thor":
        from src.utils import get_bert_thor_lr, get_bert_thor_damping
        lr = get_bert_thor_lr(cfg.Thor.lr_max, cfg.Thor.lr_min, cfg.Thor.lr_power, cfg.Thor.lr_total_steps)
        damping = get_bert_thor_damping(cfg.Thor.damping_max, cfg.Thor.damping_min, cfg.Thor.damping_power,
                                        cfg.Thor.damping_total_steps)
        split_indices = None
        if bert_net_cfg.num_hidden_layers == 12 and not bert_net_cfg.use_relative_positions:
            split_indices = [28, 55, 77]
        elif bert_net_cfg.num_hidden_layers == 24 and not bert_net_cfg.use_relative_positions:
            split_indices = [38, 93, 149]
        optimizer = thor(network, lr, damping, cfg.Thor.momentum,
                         cfg.Thor.weight_decay, cfg.Thor.loss_scale, cfg.batch_size,
                         decay_filter=lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
                         split_indices=split_indices, enable_clip_grad=True, frequency=cfg.Thor.frequency)
    else:
        raise ValueError("Don't support optimizer {}, only support [Thor]".format(cfg.optimizer))
    return optimizer


def run_pretrain():
    """pre-train bert_clue"""
    parser = argparse.ArgumentParser(description='bert pre_training')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument("--distribute", type=str, default="false", help="Run distribute, default is false.")
    parser.add_argument("--epoch_size", type=int, default="1", help="Epoch size, default is 1.")
    parser.add_argument("--device_id", type=int, default=4, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--enable_save_ckpt", type=str, default="true", help="Enable save checkpoint, default is true.")
    parser.add_argument("--enable_lossscale", type=str, default="false", help="Use lossscale or not, default is not.")
    parser.add_argument("--do_shuffle", type=str, default="false", help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="true", help="Enable data sink, default is true.")
    parser.add_argument("--data_sink_steps", type=int, default="100", help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--save_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--save_checkpoint_steps", type=int, default=1000, help="Save checkpoint steps, "
                                                                                "default is 1000.")
    parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1, "
                                                                    "meaning run all steps according to epoch number.")
    parser.add_argument("--save_checkpoint_num", type=int, default=1, help="Save checkpoint numbers, default is 1.")
    parser.add_argument("--data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")

    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                        device_id=args_opt.device_id, save_graphs=False)
    context.set_context(reserve_class_name_in_scope=False)
    ckpt_save_dir = args_opt.save_checkpoint_path
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        ckpt_save_dir = args_opt.save_checkpoint_path + 'ckpt_' + str(rank) + '/'
        context.reset_auto_parallel_context()
        _set_bert_all_reduce_split()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)

    else:
        rank = 0
        device_num = 1

    if args_opt.device_target == 'GPU' and bert_net_cfg.compute_type != mstype.float32:
        logger.warning('Gpu only support fp32 temporarily, run with fp32.')
        bert_net_cfg.compute_type = mstype.float32

    ds = create_bert_dataset(device_num, rank, args_opt.do_shuffle, args_opt.data_dir, args_opt.schema_dir)
    net_with_loss = BertNetworkWithLoss(bert_net_cfg, True)

    new_repeat_count = args_opt.epoch_size * ds.get_dataset_size() // args_opt.data_sink_steps
    if args_opt.train_steps > 0:
        new_repeat_count = min(new_repeat_count, args_opt.train_steps // args_opt.data_sink_steps)
    else:
        args_opt.train_steps = args_opt.epoch_size * ds.get_dataset_size()
        logger.info("train steps: {}".format(args_opt.train_steps))

    optimizer = _get_optimizer(args_opt, net_with_loss)
    callback = [TimeMonitor(args_opt.data_sink_steps), LossCallBack(ds.get_dataset_size())]
    if args_opt.enable_save_ckpt == "true" and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                     keep_checkpoint_max=args_opt.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_bert', directory=ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)

    if args_opt.load_checkpoint_path:
        param_dict = load_checkpoint(args_opt.load_checkpoint_path)
        load_param_into_net(net_with_loss, param_dict)

    if args_opt.enable_lossscale == "true":
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)
        net_with_grads = BertTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                           scale_update_cell=update_cell)
    else:
        net_with_grads = BertTrainOneStepCell(net_with_loss, optimizer=optimizer, sens=cfg.Thor.loss_scale,
                                              enable_clip_grad=False)

    model = Model(net_with_grads)
    model = ConvertModelUtils().convert_to_thor_model(model, network=net_with_grads, optimizer=optimizer)
    model.train(new_repeat_count, ds, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == "true"), sink_size=args_opt.data_sink_steps)


if __name__ == '__main__':
    set_seed(0)
    run_pretrain()
