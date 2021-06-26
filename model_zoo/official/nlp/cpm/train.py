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
"""Train."""
import os
import ast
import argparse
import math
import numpy as np

import mindspore.dataset as ds
from mindspore import context
from mindspore.train.model import Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore.communication import management as MultiAscend
from mindspore.context import ParallelMode
from mindspore.common import set_seed

from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindspore.parallel import set_algo_parameters

from src.config import config_train_single_machine, config_train_multi_machine
from src.cpm_train import CPMWithLoss, CPMTrainOneStepWithLossScaleCell, VirtualDatasetOneInputCell, \
    CPMTrainAccuStepsWithLossScaleCell
from src.lr_schedule import CPMLearningRate
from src.loss_monitor import LossCallBack, TimeCallBack
from src.model_cpm import Model_ACCU as Model_CPM

device_id = int(os.getenv("DEVICE_ID"))

set_seed(23333)
context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend",
                    device_id=device_id)
context.set_context(variable_memory_max_size="30GB")


def collate(truth, input_ids, BatchInfo):
    """Collate operation for dataset."""
    bs = len(truth)
    max_size = np.size(input_ids, 1)

    attn_mask = np.tril(np.ones(shape=(max_size, max_size),))
    attention_mask = np.expand_dims(attn_mask, 0)
    attention_mask = np.tile(attention_mask, (bs, 1, 1)).astype(np.float32)

    position_ids = np.expand_dims(np.arange(max_size * 1), 0)
    position_ids = np.tile(position_ids, (bs, 1)).astype(np.int32)

    truth_list = np.zeros(bs, dtype=np.int32)

    for i in range(bs):
        truth_list[i] = truth[i]

    return input_ids, attention_mask, position_ids, truth_list


def _load_dataset(dataset_path, batch_size, rank_size=None, rank_id=None, shuffle=True, drop_remainder=True,
                  is_training=True):
    """Loader for data."""
    ds.config.set_seed(1)
    data = ds.MindDataset(dataset_file=dataset_path,
                          columns_list=["truth", "input_ids", "loss_mask", "labels", "size"],
                          shuffle=shuffle)

    type_cast_op = C.TypeCast(mstype.float32)
    type_cast_op_int = C.TypeCast(mstype.int32)
    data = data.map(input_columns="input_ids", operations=type_cast_op_int)
    data = data.map(input_columns="labels", operations=type_cast_op_int)
    data = data.map(input_columns="loss_mask", operations=type_cast_op)
    if is_training:
        data = data.batch(batch_size,
                          per_batch_map=collate,
                          input_columns=["truth", "input_ids"],
                          output_columns=["input_ids", "attention_mask", "position_ids", "truth"],
                          column_order=["input_ids", "attention_mask", "position_ids", "loss_mask", "labels"],
                          num_parallel_workers=4,
                          drop_remainder=drop_remainder)
    else:
        data = data.batch(batch_size,
                          per_batch_map=collate,
                          input_columns=["truth", "input_ids"],
                          output_columns=["input_ids", "attention_mask", "position_ids", "truth"],
                          column_order=["input_ids", "attention_mask", "position_ids", "loss_mask", "labels", "truth"],
                          num_parallel_workers=4,
                          drop_remainder=drop_remainder)

    return data


def load_dataset(dataset, batch_size,
                 rank_size=None, rank_id=None,
                 shuffle=True,
                 drop_remainder=True,
                 is_training=True):
    """
    Load dataset.

    Args:
        dataset (class): Dataset.
        batch_size (int): Batch size.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.
        drop_remainder (bool): Determines whether or not to drop the last possibly incomplete batch.
        is_training (bool): Whether training mode.

    Returns:
        Dataset, dataset instance.
    """
    return _load_dataset(dataset,
                         batch_size, rank_size=rank_size,
                         rank_id=rank_id, shuffle=shuffle,
                         drop_remainder=drop_remainder,
                         is_training=is_training)


def _build_training_pipeline(datasets, pretrain_ckpt_path, config_train):
    """
    Building training pipeline
    """
    net_with_loss = CPMWithLoss(batch_size=config_train.batch_size,
                                seq_length=config_train.seq_length,
                                vocab_size=config_train.vocab_size,
                                hidden_size=config_train.hidden_size,
                                config=config_train,
                                num_hidden_layers=config_train.num_hidden_layers,
                                num_attention_heads=config_train.num_attention_heads)

    net_with_loss = VirtualDatasetOneInputCell(net_with_loss)

    param_dict = load_checkpoint(pretrain_ckpt_path)

    can_be_loaded = {}
    for name, _ in param_dict.items():
        if 'cpm_model.' not in name:
            can_be_loaded['cpm_model.' + name] = param_dict[name]
        else:
            can_be_loaded[name] = param_dict[name]
    load_param_into_net(net_with_loss, parameter_dict=can_be_loaded)
    print("------->Load pretrained parameter successfully<------------")

    steps_per_epoch = datasets.get_dataset_size()
    print("++++++Dataset size= ", steps_per_epoch, flush=True)
    print("++++++MP= ", str(config_train.mp), flush=True)
    print("++++++DP= ", str(config_train.dp), flush=True)
    print("++++++Global_batch_size= ", str(config_train.batch_size), flush=True)
    lr_schedule = CPMLearningRate(learning_rate=config_train.lr,
                                  warmup_steps=int(steps_per_epoch * config_train.epoch * config_train.warmup_steps),
                                  end_steps=steps_per_epoch * config_train.epoch)
    params = net_with_loss.trainable_params()

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': config_train.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]
    optimizer = AdamWeightDecay(group_params, lr_schedule, eps=config_train.eps, beta1=0.9, beta2=0.95)

    callback_size = config_train.grad_accumulation_step if config_train.grad_accumulation_step > 1 \
        else config_train.sink_size
    actual_epoch_num = int(config_train.epoch * steps_per_epoch // callback_size)
    print("++++++actual_epoch_num= ", str(actual_epoch_num), flush=True)

    if config_train.grad_accumulation_step > 1:
        callback = [TimeCallBack(), LossCallBack(steps_per_epoch)]
    else:
        callback = [TimeMonitor(), LossCallBack(steps_per_epoch)]

    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch,
                                   integrated_save=False,
                                   keep_checkpoint_max=config_train.epoch)
    ckpt_model = ModelCheckpoint(prefix='cpm_rank_{}'.format(os.getenv("RANK_ID")),
                                 directory=os.path.join('./', 'ckpt_rank_{}'.format(os.getenv("RANK_ID"))),
                                 config=ckpt_config)
    callback.append(ckpt_model)

    dynamic_loss_cale = DynamicLossScaleUpdateCell(loss_scale_value=math.pow(2, 32),
                                                   scale_factor=2,
                                                   scale_window=1000)
    print(dynamic_loss_cale)
    print(" | Start pre-training job.")
    if config_train.grad_accumulation_step > 1:
        cpm_with_grads = CPMTrainAccuStepsWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                            scale_update_cell=dynamic_loss_cale)
        model = Model_CPM(cpm_with_grads)
        model.train(config_train.epoch, datasets, callbacks=callback,
                    dataset_sink_mode=True)
    else:
        cpm_with_grads = CPMTrainOneStepWithLossScaleCell(net_with_loss, optimizer, dynamic_loss_cale)

        model = Model(cpm_with_grads)
        model.train(epoch=actual_epoch_num,
                    train_dataset=datasets,
                    callbacks=callback,
                    sink_size=callback_size,
                    dataset_sink_mode=True)


def set_parallel_env(config_train):
    r"""
    Parallel environment.
    """
    context.reset_auto_parallel_context()
    MultiAscend.init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      device_num=MultiAscend.get_group_size(),
                                      gradients_mean=True,
                                      grad_accumulation_step=config_train.grad_accumulation_step,
                                      full_batch=True)
    context.set_auto_parallel_context(enable_parallel_optimizer=True)
    context.set_auto_parallel_context(strategy_ckpt_save_file='./train_strategy.ckpt')
    set_algo_parameters(elementwise_op_strategy_follow=True)


def train_single(input_file, pretrain_ckpt_path, config_train):
    """
    Training on single device
    """
    print("Staring training on single device")
    preprocessed_data = load_dataset(dataset=input_file,
                                     batch_size=config_train.batch_size)
    _build_training_pipeline(preprocessed_data, pretrain_ckpt_path, config_train)


def train_paralle(input_file, pretrain_ckpt_path, config_train):
    """
    Training on multi device
    """
    set_parallel_env(config_train)
    print("Staring training on multiple device")
    processed_data = load_dataset(dataset=input_file,
                                  batch_size=config_train.batch_size,
                                  rank_size=MultiAscend.get_group_size(),
                                  rank_id=MultiAscend.get_rank())
    _build_training_pipeline(processed_data, pretrain_ckpt_path, config_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CPM training.")
    parser.add_argument("--dataset", type=str, default="", help="CPM dataset path")
    parser.add_argument("--pretrain_ckpt_path", type=str, default="",
                        help="Load the checkpoint file path for train.")
    parser.add_argument("--multi_machine", type=ast.literal_eval, default=False, help='distributed training')

    args = parser.parse_args()
    if args.multi_machine:
        print("Training on multiple machines.")
        train_paralle(args.dataset, args.pretrain_ckpt_path, config_train_multi_machine)
    else:
        print("Training on single machine.")
        train_paralle(args.dataset, args.pretrain_ckpt_path, config_train_single_machine)
