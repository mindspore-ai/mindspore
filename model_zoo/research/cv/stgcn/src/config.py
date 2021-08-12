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
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict

stgcn_chebconv_45min_cfg = edict({
    'learning_rate': 0.003,
    'n_his': 12,
    'n_pred': 9,
    'epochs': 500,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.7,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func': "glu",
    'graph_conv_type': "chebconv",
    'mat_type': "wid_sym_normd_lap_mat",
})

stgcn_chebconv_30min_cfg = edict({
    'learning_rate': 0.003,
    'n_his': 12,
    'n_pred': 6,
    'epochs': 500,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.7,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func': "glu",
    'graph_conv_type': "chebconv",
    'mat_type': "wid_sym_normd_lap_mat",
})

stgcn_chebconv_15min_cfg = edict({
    'learning_rate': 0.002,
    'n_his': 12,
    'n_pred': 3,
    'epochs': 100,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.999,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func': "glu",
    'graph_conv_type': "chebconv",
    'mat_type': "wid_rw_normd_lap_mat",
})

stgcn_gcnconv_45min_cfg = edict({
    'learning_rate': 0.003,
    'n_his': 12,
    'n_pred': 9,
    'epochs': 500,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.7,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func': "glu",
    'graph_conv_type': "gcnconv",
    'mat_type': "hat_sym_normd_lap_mat",
})

stgcn_gcnconv_30min_cfg = edict({
    'learning_rate': 0.003,
    'n_his': 12,
    'n_pred': 6,
    'epochs': 500,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.7,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func': "glu",
    'graph_conv_type': "gcnconv",
    'mat_type': "hat_sym_normd_lap_mat",
})

stgcn_gcnconv_15min_cfg = edict({
    'learning_rate': 0.002,
    'n_his': 12,
    'n_pred': 3,
    'epochs': 100,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.9999,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func': "glu",
    'graph_conv_type': "gcnconv",
    'mat_type': "hat_rw_normd_lap_mat",
})
