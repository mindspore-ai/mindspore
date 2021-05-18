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
"""train finetune"""
import os
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from src.args import args
from src.data.imagenet import ImgData
from src.data.srdata import SRData
from src.data.div2k import DIV2K
from src.data.bicubic import bicubic
from src.ipt_model import IPT
from src.utils import Trainer

def train_net(distribute, imagenet):
    """Train net with finetune"""
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

    if imagenet == 1:
        train_dataset = ImgData(args)
    elif not args.derain:
        train_dataset = DIV2K(args, name=args.data_train, train=True, benchmark=False)
        train_dataset.set_scale(args.task_id)
    else:
        train_dataset = SRData(args, name=args.data_train, train=True, benchmark=False)
        train_dataset.set_scale(args.task_id)

    if distribute:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.group_size, gradients_mean=True)
        print('Rank {}, group_size {}'.format(args.rank, args.group_size))
        if imagenet == 1:
            train_de_dataset = ds.GeneratorDataset(train_dataset,
                                                   ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
                                                   num_shards=args.group_size, shard_id=args.rank, shuffle=True)
        else:
            train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR", "idx", "filename"],
                                                   num_shards=args.group_size, shard_id=args.rank, shuffle=True)
    else:
        if imagenet == 1:
            train_de_dataset = ds.GeneratorDataset(train_dataset,
                                                   ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
                                                   shuffle=True)
        else:
            train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR", "idx", "filename"], shuffle=True)

    if args.imagenet == 1:
        resize_fuc = bicubic()
        train_de_dataset = train_de_dataset.batch(
            args.batch_size,
            input_columns=["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
            output_columns=["LR", "HR", "idx", "filename"], drop_remainder=True,
            per_batch_map=resize_fuc.forward)
    else:
        train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)

    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)
    net_m = IPT(args)
    print("Init net weights successfully")

    if args.pth_path:
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")

    train_func = Trainer(args, train_loader, net_m)

    for epoch in range(0, args.epochs):
        train_func.update_learning_rate(epoch)
        train_func.train()

if __name__ == "__main__":
    train_net(distribute=args.distribute, imagenet=args.imagenet)
