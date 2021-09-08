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
testing network performance.
"""

import os
import ast
import argparse

import pandas as pd
from sklearn import preprocessing

from mindspore.common import dtype as mstype

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode

from src.model import models
from src.config import stgcn_chebconv_45min_cfg, stgcn_chebconv_30min_cfg, stgcn_chebconv_15min_cfg, stgcn_gcnconv_45min_cfg, stgcn_gcnconv_30min_cfg, stgcn_gcnconv_15min_cfg
from src import dataloader, utility

os.system("export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")
parser = argparse.ArgumentParser('mindspore stgcn testing')
parser.add_argument('--device_target', type=str, default='Ascend', \
 help='device where the code will be implemented. (Default: Ascend)')

# The way of testing
parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run on modelarts.')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_id', type=int, default=0, help='Device id.')

# Path for data and checkpoint
parser.add_argument('--data_url', type=str, default='', help='Test dataset directory.')
parser.add_argument('--train_url', type=str, default='', help='Output directory.')
parser.add_argument('--data_path', type=str, default="vel.csv", help='Dataset file of vel.')
parser.add_argument('--wam_path', type=str, default="adj_mat.csv", help='Dataset file of warm.')
parser.add_argument('--ckpt_url', type=str, default='', help='The path of checkpoint.')
parser.add_argument('--ckpt_name', type=str, default="", help='the name of checkpoint.')

# Super parameters for testing
parser.add_argument('--n_pred', type=int, default=3, help='The number of time interval for predcition')

#network
parser.add_argument('--graph_conv_type', type=str, default="gcnconv", help='Grapg convolution type')
#dataset


args, _ = parser.parse_known_args()
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

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
    local_data_url = '/cache/data'
    local_ckpt_url = '/cache/ckpt'
    mox.file.copy_parallel(args.data_url, local_data_url)
    mox.file.copy_parallel(args.ckpt_url, local_ckpt_url)
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num, \
         parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    data_dir = local_data_url + '/'
    local_ckpt_url = local_ckpt_url + '/'
else:
    if args.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        cfg.batch_size = cfg.batch_size*int(8/device_num)
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, \
         parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        device_num = 1
        device_id = args.device_id
        context.set_context(device_id=args.device_id)
    data_dir = args.data_url + '/'
    local_ckpt_url = args.ckpt_url + '/'

adj_mat = dataloader.load_weighted_adjacency_matrix(data_dir+args.wam_path)

n_vertex_vel = pd.read_csv(data_dir+args.data_path, header=None).shape[1]
n_vertex_adj = pd.read_csv(data_dir+args.wam_path, header=None).shape[1]
if n_vertex_vel == n_vertex_adj:
    n_vertex = n_vertex_vel
else:
    raise ValueError(f'ERROR: number of vertices in dataset is not equal to \
     number of vertices in weighted adjacency matrix.')

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

    zscore = preprocessing.StandardScaler()
    if args.run_modelarts or args.run_distribute:
        dataset = dataloader.create_dataset(data_dir+args.data_path, \
         cfg.batch_size, cfg.n_his, cfg.n_pred, zscore, False, device_num, device_id, mode=2)
    else:
        dataset = dataloader.create_dataset(data_dir+args.data_path, \
         cfg.batch_size, cfg.n_his, cfg.n_pred, zscore, True, device_num, device_id, mode=2)
    data_len = dataset.get_dataset_size()

    param_dict = load_checkpoint(local_ckpt_url+args.ckpt_name)
    load_param_into_net(net, param_dict)

    test_MAE, test_RMSE, test_MAPE = utility.evaluate_metric(net, dataset, zscore)
    print(f'MAE {test_MAE:.2f} | MAPE {test_MAPE*100:.2f} | RMSE {test_RMSE:.2f}')
