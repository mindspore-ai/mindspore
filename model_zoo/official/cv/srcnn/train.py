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
"""srcnn training"""

import os
import argparse
import ast

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import ParallelMode

from src.config import srcnn_cfg as config
from src.dataset import create_train_dataset
from src.srcnn import SRCNN

set_seed(1)

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn training")
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument('--device_target', type=str, default='GPU', choices=("GPU"),
                        help="Device target, support GPU.")
    parser.add_argument('--pre_trained', type=str, default='', help='model_path, local pretrained model to load')
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default: false.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    args, _ = parser.parse_known_args()


    if args.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target,
                            save_graphs=False)
    else:
        raise ValueError("Unsupported device target.")

    rank = 0
    device_num = 1
    if args.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)

    train_dataset = create_train_dataset(args.dataset_path, batch_size=config.batch_size,
                                         shard_id=rank, num_shard=device_num)

    step_size = train_dataset.get_dataset_size()

    # define net
    net = SRCNN()

    # init weight
    if args.pre_trained:
        param_dict = load_checkpoint(args.pre_trained)
        if args.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)

    lr = Tensor(config.lr, ms.float32)

    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)
    loss = nn.MSELoss(reduction='mean')
    model = Model(net, loss_fn=loss, optimizer=opt)

    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if config.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(prefix="srcnn", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpt_cb)

    model.train(config.epoch_size, train_dataset, callbacks=callbacks)
