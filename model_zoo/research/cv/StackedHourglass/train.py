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
"""
run model train
"""
import math
import os

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)

from src.config import parse_args
from src.dataset.DatasetGenerator import DatasetGenerator
from src.dataset.MPIIDataLoader import MPII
from src.models.loss import HeatmapLoss
from src.models.StackedHourglassNet import StackedHourglassNet

set_seed(1)

args = parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.img_dir) or not os.path.exists(args.annot_dir):
        print("Dataset not found.")
        exit()

    # Set context mode
    if args.context_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    if args.parallel:
        # Parallel mode
        context.reset_auto_parallel_context()
        init()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        args.rank_id = get_rank()
        args.group_size = get_group_size()
    else:
        args.rank_id = 0
        args.group_size = 1

    net = StackedHourglassNet(args.nstack, args.inp_dim, args.oup_dim)

    # Process dataset
    mpii = MPII()
    train, valid = mpii.setup_val_split()

    train_generator = DatasetGenerator(args.input_res, args.output_res, mpii, train)
    train_size = len(train_generator)
    train_sampler = ds.DistributedSampler(num_shards=args.group_size, shard_id=args.rank_id, shuffle=True)
    train_data = ds.GeneratorDataset(train_generator, ["data", "label"], sampler=train_sampler)
    train_data = train_data.batch(args.batch_size, True, args.group_size)

    print("train data size:", train_size)
    step_per_epoch = math.ceil(train_size / args.batch_size / args.group_size)

    # Define loss function
    loss_func = HeatmapLoss()
    # Define optimizer
    lr_decay = nn.exponential_decay_lr(
        args.initial_lr, args.decay_rate, args.num_epoch * step_per_epoch, step_per_epoch, args.decay_epoch
    )
    optimizer = nn.Adam(net.trainable_params(), lr_decay)

    # Define model
    model = Model(net, loss_func, optimizer, amp_level=args.amp_level, keep_batchnorm_fp32=False)

    # Define callback functions
    callbacks = []
    callbacks.append(LossMonitor(args.loss_log_interval))
    callbacks.append(TimeMonitor(train_size))

    # Save checkpoint file
    if args.rank_id == 0:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=args.save_checkpoint_epochs * step_per_epoch,
            keep_checkpoint_max=args.keep_checkpoint_max,
        )
        ckpoint = ModelCheckpoint("ckpt", config=config_ck)
        callbacks.append(ckpoint)

    model.train(args.num_epoch, train_data, callbacks=callbacks, dataset_sink_mode=True)
