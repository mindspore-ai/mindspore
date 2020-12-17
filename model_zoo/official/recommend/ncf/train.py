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
"""Training entry file"""
import os

import argparse
from absl import logging

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import context, Model
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.ncf import NCFModel, NetWithLossClass, TrainStepWrap

from config import cfg

set_seed(1)

logging.set_verbosity(logging.INFO)

parser = argparse.ArgumentParser(description='NCF')
parser.add_argument("--data_path", type=str, default="./dataset/")  # The location of the input data.
parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "ml-20m"])  # Dataset to be trained and evaluated. ["ml-1m", "ml-20m"]
parser.add_argument("--train_epochs", type=int, default=14)  # The number of epochs used to train.
parser.add_argument("--batch_size", type=int, default=256)  # Batch size for training and evaluation
parser.add_argument("--num_neg", type=int, default=4)  # The Number of negative instances to pair with a positive instance.
parser.add_argument("--output_path", type=str, default="./output/")  # The location of the output file.
parser.add_argument("--loss_file_name", type=str, default="loss.log")  # Loss output file.
parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/")  # The location of the checkpoint file.
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--device_id', type=int, default=1, help='device id of GPU or Ascend. (Default: None)')
parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
args = parser.parse_args()

def test_train():
    """train entry method"""
    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
            context.set_context(device_id=args.device_id)
        elif args.device_target == "GPU":
            init()

        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, gradients_mean=True)
    else:
        context.set_context(device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    layers = cfg.layers
    num_factors = cfg.num_factors
    epochs = args.train_epochs

    ds_train, num_train_users, num_train_items = create_dataset(test_train=True, data_dir=args.data_path,
                                                                dataset=args.dataset, train_epochs=1,
                                                                batch_size=args.batch_size, num_neg=args.num_neg)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    ncf_net = NCFModel(num_users=num_train_users,
                       num_items=num_train_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)
    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net, ds_train.get_dataset_size() * (epochs + 1))

    train_net.set_train()

    model = Model(train_net)
    callback = LossMonitor(per_print_times=ds_train.get_dataset_size())
    ckpt_config = CheckpointConfig(save_checkpoint_steps=(4970845+args.batch_size-1)//(args.batch_size),
                                   keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix='NCF', directory=args.checkpoint_path, config=ckpt_config)
    model.train(epochs,
                ds_train,
                callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb],
                dataset_sink_mode=True)


if __name__ == '__main__':
    test_train()
