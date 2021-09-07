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
train network.
"""

import os
import argparse
import ast
import pandas as pd
from sklearn import preprocessing

from mindspore.common import dtype as mstype
import mindspore.nn as nn

from mindspore import context
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor
from mindspore.common import set_seed

from src.config import stgcn_chebconv_45min_cfg, stgcn_chebconv_30min_cfg, stgcn_chebconv_15min_cfg, stgcn_gcnconv_45min_cfg, stgcn_gcnconv_30min_cfg, stgcn_gcnconv_15min_cfg
from src import dataloader, utility
from src.model import models, metric

set_seed(1)

parser = argparse.ArgumentParser('mindspore stgcn training')

# The way of training
parser.add_argument('--device_target', type=str, default='Ascend', \
 help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run on modelarts')
parser.add_argument('--save_check_point', type=bool, default=True, help='Whether save checkpoint')

# Path for data and checkpoint
parser.add_argument('--data_url', type=str, required=True, help='Train dataset directory.')
parser.add_argument('--train_url', type=str, required=True, help='Save checkpoint directory.')
parser.add_argument('--data_path', type=str, default="vel.csv", help='Dataset file of vel.')
parser.add_argument('--wam_path', type=str, default="adj_mat.csv", help='Dataset file of warm.')

# Super parameters for training
parser.add_argument('--n_pred', type=int, default=3, help='The number of time interval for predcition, default as 3')
parser.add_argument('--opt', type=str, default='AdamW', help='optimizer, default as AdamW')

#network
parser.add_argument('--graph_conv_type', type=str, default="gcnconv", help='Grapg convolution type')

args, _ = parser.parse_known_args()

if args.graph_conv_type == "chebconv":
    if args.n_pred == 9:
        cfg = stgcn_chebconv_45min_cfg
    elif args.n_pred == 6:
        cfg = stgcn_chebconv_30min_cfg
    elif args.n_pred == 3:
        cfg = stgcn_chebconv_15min_cfg
    else:
        raise ValueError("Unsupported n_pred.")
elif args.graph_conv_type == "gcnconv":
    if args.n_pred == 9:
        cfg = stgcn_gcnconv_45min_cfg
    elif args.n_pred == 6:
        cfg = stgcn_gcnconv_30min_cfg
    elif args.n_pred == 3:
        cfg = stgcn_gcnconv_15min_cfg
    else:
        raise ValueError("Unsupported pred.")
else:
    raise ValueError("Unsupported graph_conv_type.")

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)

if ((cfg.Kt - 1) * 2 * cfg.stblock_num > cfg.n_his) or ((cfg.Kt - 1) * 2 * cfg.stblock_num <= 0):
    raise ValueError(f'ERROR: {cfg.Kt} and {cfg.stblock_num} are unacceptable.')

Ko = cfg.n_his - (cfg.Kt - 1) * 2 * cfg.stblock_num

if (cfg.graph_conv_type != "chebconv") and (cfg.graph_conv_type != "gcnconv"):
    raise NotImplementedError(f'ERROR: {cfg.graph_conv_type} is not implemented.')

if (cfg.graph_conv_type == 'gcnconv') and (cfg.Ks != 2):
    cfg.Ks = 2

# blocks: settings of channel size in st_conv_blocks and output layer,
# using the bottleneck design in st_conv_blocks
blocks = []
blocks.append([1])
for l in range(cfg.stblock_num):
    blocks.append([64, 16, 64])
if Ko == 0:
    blocks.append([128])
elif Ko > 0:
    blocks.append([128, 128])
blocks.append([1])


day_slot = int(24 * 60 / cfg.time_intvl)
cfg.n_pred = cfg.n_pred

time_pred = cfg.n_pred * cfg.time_intvl
time_pred_str = str(time_pred) + '_mins'

if args.run_modelarts:
    import moxing as mox
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    cfg.batch_size = cfg.batch_size*int(8/device_num)
    context.set_context(device_id=device_id)
    local_data_url = '/cache/data'
    local_train_url = '/cache/train'
    #mox.file.make_dirs(local_train_url)
    mox.file.copy_parallel(args.data_url, local_data_url)
    if device_num > 1:
        init()
        #context.set_auto_parallel_context(parameter_broadcast=True)
        context.set_auto_parallel_context(device_num=device_num, \
         parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    data_dir = local_data_url + '/'
else:
    if args.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        cfg.batch_size = cfg.batch_size*int(8/device_num)
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        #context.set_auto_parallel_context(parameter_broadcast=True)
        context.set_auto_parallel_context(device_num=device_num, \
         parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        context.set_context(device_id=args.device_id)
        device_num = 1
        cfg.batch_size = cfg.batch_size*int(8/device_num)
        device_id = args.device_id
    data_dir = args.data_url + '/'
    model_save_path = args.train_url + cfg.graph_conv_type + '_' + time_pred_str

adj_mat = dataloader.load_weighted_adjacency_matrix(data_dir+args.wam_path)

n_vertex_vel = pd.read_csv(data_dir+args.data_path, header=None).shape[1]
n_vertex_adj = pd.read_csv(data_dir+args.wam_path, header=None).shape[1]
if n_vertex_vel == n_vertex_adj:
    n_vertex = n_vertex_vel
else:
    raise ValueError(f"ERROR: number of vertices in dataset is not equal to number \
     of vertices in weighted adjacency matrix.")

mat = utility.calculate_laplacian_matrix(adj_mat, cfg.mat_type)
conv_matrix = Tensor(Tensor.from_numpy(mat), mstype.float32)
if cfg.graph_conv_type == "chebconv":
    if (cfg.mat_type != "wid_sym_normd_lap_mat") and (cfg.mat_type != "wid_rw_normd_lap_mat"):
        raise ValueError(f'ERROR: {cfg.mat_type} is wrong.')
elif cfg.graph_conv_type == "gcnconv":
    if (cfg.mat_type != "hat_sym_normd_lap_mat") and (cfg.mat_type != "hat_rw_normd_lap_mat"):
        raise ValueError(f'ERROR: {cfg.mat_type} is wrong.')

stgcn_conv = models.STGCN_Conv(cfg.Kt, cfg.Ks, blocks, cfg.n_his, n_vertex, \
    cfg.gated_act_func, cfg.graph_conv_type, conv_matrix, cfg.drop_rate)
net = stgcn_conv

if __name__ == "__main__":
    #start training

    zscore = preprocessing.StandardScaler()
    if args.run_modelarts or args.run_distribute:
        dataset = dataloader.create_dataset(data_dir+args.data_path, cfg.batch_size, cfg.n_his, \
         cfg.n_pred, zscore, False, device_num, device_id, mode=0)
    else:
        dataset = dataloader.create_dataset(data_dir+args.data_path, cfg.batch_size, cfg.n_his, \
         cfg.n_pred, zscore, True, device_num, device_id, mode=0)
    data_len = dataset.get_dataset_size()

    learning_rate = nn.exponential_decay_lr(learning_rate=cfg.learning_rate, decay_rate=cfg.gamma, \
     total_step=data_len*cfg.epochs, step_per_epoch=data_len, decay_epoch=cfg.decay_epoch)
    if args.opt == "RMSProp":
        optimizer = nn.RMSProp(net.trainable_params(), learning_rate=learning_rate)
    elif args.opt == "Adam":
        optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate, \
         weight_decay=cfg.weight_decay_rate)
    elif args.opt == "AdamW":
        optimizer = nn.AdamWeightDecay(net.trainable_params(), learning_rate=learning_rate, \
         weight_decay=cfg.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {args.opt} is undefined.')

    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=data_len)
    callbacks = [time_cb, loss_cb]

    #save training results
    if args.save_check_point and (device_num == 1 or device_id == 0):
        config_ck = CheckpointConfig(
            save_checkpoint_steps=data_len*cfg.epochs, keep_checkpoint_max=cfg.epochs)
        if args.run_modelarts:
            ckpoint_cb = ModelCheckpoint(prefix='STGCN'+cfg.graph_conv_type+str(cfg.n_pred)+'-', \
             directory=local_train_url, config=config_ck)
        else:
            ckpoint_cb = ModelCheckpoint(prefix='STGCN', directory=model_save_path, config=config_ck)
        callbacks += [ckpoint_cb]

    net = metric.LossCellWithNetwork(net)
    model = Model(net, optimizer=optimizer, amp_level='O3')

    model.train(cfg.epochs, dataset, callbacks=callbacks)
    if args.run_modelarts:
        mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
