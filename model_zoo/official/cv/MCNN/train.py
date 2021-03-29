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
######################## train mcnn example ########################
train mcnn and get network model files(.ckpt) :
python train.py
"""
import os
import argparse
import ast
import numpy as np
from mindspore.communication.management import init
import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore import context, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train import Model
from src.data_loader import ImageDataLoader
from src.config import crowd_cfg as cfg
from src.dataset import create_dataset
from src.mcnn import MCNN
from src.generator_lr import get_lr_sha
from src.Mcnn_Callback import mcnn_callback

parser = argparse.ArgumentParser(description='MindSpore MCNN Example')
parser.add_argument('--run_offline', type=ast.literal_eval,
                    default=False, help='run in offline is False or True')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default="/cache/train_output", help='Location of ckpt.')

parser.add_argument('--data_url', default=None, help='Location of data.')
parser.add_argument('--train_url', default=None, help='Location of training outputs.')

parser.add_argument('--train_path', required=True, default=None, help='Location of data.')
parser.add_argument('--train_gt_path', required=True, default=None, help='Location of data.')
parser.add_argument('--val_path', required=True,
                    default='/lhb1234/mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/val',
                    help='Location of data.')
parser.add_argument('--val_gt_path', required=True,
                    default='/lhb1234/mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/val_den',
                    help='Location of data.')
args = parser.parse_args()
rand_seed = 64678
np.random.seed(rand_seed)

if __name__ == "__main__":
    device_num = int(os.getenv("RANK_SIZE"))
    device_id = int(os.getenv("DEVICE_ID"))

    print("device_id:", device_id)
    print("device_num:", device_num)
    device_target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(save_graphs=False)

    if device_target == "Ascend":
        context.set_context(device_id=device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            # local_data1_url=os.path.join(local_data1_url,str(device_id)) # 可以删除
            # local_data2_url=os.path.join(local_data2_url,str(device_id))
            # local_data3_url=os.path.join(local_data3_url,str(device_id))
            # local_data4_url=os.path.join(local_data4_url,str(device_id))
    else:
        raise ValueError("Unsupported platform.")
    if args.run_offline:
        local_data1_url = args.train_path
        local_data2_url = args.train_gt_path
        local_data3_url = args.val_path
        local_data4_url = args.val_gt_path
    else:
        import moxing as mox
        local_data1_url = '/cache/train_path'
        local_data2_url = '/cache/train_gt_path'
        local_data3_url = '/cache/val_path'
        local_data4_url = '/cache/val_gt_path'

        mox.file.copy_parallel(src_url=args.train_path, dst_url=local_data1_url) # pcl
        mox.file.copy_parallel(src_url=args.train_gt_path, dst_url=local_data2_url) # pcl
        mox.file.copy_parallel(src_url=args.val_path, dst_url=local_data3_url) # pcl
        mox.file.copy_parallel(src_url=args.val_gt_path, dst_url=local_data4_url) # pcl

    data_loader = ImageDataLoader(local_data1_url, local_data2_url, shuffle=True, gt_downsample=True, pre_load=True)
    data_loader_val = ImageDataLoader(local_data3_url, local_data4_url,
                                      shuffle=False, gt_downsample=True, pre_load=True)
    ds_train = create_dataset(data_loader, target=args.device_target)
    ds_val = create_dataset(data_loader_val, target=args.device_target, train=False)

    ds_train = ds_train.batch(cfg['batch_size'])
    ds_val = ds_val.batch(1)

    network = MCNN()
    net_loss = nn.MSELoss(reduction='mean')
    lr = Tensor(get_lr_sha(0, cfg['lr'], cfg['epoch_size'], ds_train.get_dataset_size()))
    net_opt = nn.Adam(list(filter(lambda p: p.requires_grad, network.get_parameters())), learning_rate=lr)

    if args.device_target != "Ascend":
        model = Model(network, net_loss, net_opt)
    else:
        model = Model(network, net_loss, net_opt, amp_level="O2")

    print("============== Starting Training ==============")
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    eval_callback = mcnn_callback(network, ds_val, args.run_offline, args.ckpt_path)
    model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, eval_callback, LossMonitor(1)])
    if not args.run_offline:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url="obs://lhb1234/MCNN/ckpt")
