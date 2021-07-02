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
utils, will be used in train.py
"""
import mindspore.nn as nn
from src.config import awa_cfg, cub_cfg
from src.demnet import DEMNet1, DEMNet2, DEMNet3, DEMNet4, MyWithLossCell

def acc_cfg(args):
    if args.dataset == 'CUB':
        pred_len = 2933
    elif args.dataset == 'AwA':
        pred_len = 6180
    return pred_len

def backbone_cfg(args):
    """set backbone"""
    if args.dataset == 'CUB':
        net = DEMNet1()
    elif args.dataset == 'AwA':
        if args.train_mode == 'att':
            net = DEMNet2()
        elif args.train_mode == 'word':
            net = DEMNet3()
        elif args.train_mode == 'fusion':
            net = DEMNet4()
    return net

def param_cfg(args):
    """set Hyperparameter"""
    if args.dataset == 'CUB':
        lr = cub_cfg.lr_att
        weight_decay = cub_cfg.wd_att
        clip_param = cub_cfg.clip_att
    elif args.dataset == 'AwA':
        if args.train_mode == 'att':
            lr = awa_cfg.lr_att
            weight_decay = awa_cfg.wd_att
            clip_param = awa_cfg.clip_att
        elif args.train_mode == 'word':
            lr = awa_cfg.lr_word
            weight_decay = awa_cfg.wd_word
            clip_param = awa_cfg.clip_word
        elif args.train_mode == 'fusion':
            lr = awa_cfg.lr_fusion
            weight_decay = awa_cfg.wd_fusion
            clip_param = awa_cfg.clip_fusion
    return lr, weight_decay, clip_param

def withlosscell_cfg(args):
    if args.train_mode == 'fusion':
        return MyWithLossCell
    return nn.WithLossCell

def save_min_acc_cfg(args):
    if args.train_mode == 'att':
        save_min_acc = 0.5
    elif args.train_mode == 'word':
        save_min_acc = 0.7
    elif args.train_mode == 'fusion':
        save_min_acc = 0.7
    return save_min_acc
