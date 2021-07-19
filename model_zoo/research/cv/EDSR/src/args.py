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
"""args parser"""
import argparse
parser = argparse.ArgumentParser(description='EDSR')
# Data specifications
parser.add_argument('--dir_data', type=str, default='/cache/data/',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-900',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='input patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.1,
                    help='residual scaling')
# Training specifications
parser.add_argument('--test_every', type=int, default=8000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--loss_scale', type=float, default=1024.0,
                    help='init loss scale')
# ckpt specifications
parser.add_argument('--ckpt_save_path', type=str, default='./ckpt/',
                    help='path to save ckpt')
parser.add_argument('--ckpt_save_interval', type=int, default=10,
                    help='save ckpt frequency, unit is epoch')
parser.add_argument('--ckpt_save_max', type=int, default=5,
                    help='max number of saved ckpt')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='path of saved ckpt')
# alltask
parser.add_argument('--task_id', type=int, default=0)

args, unparsed = parser.parse_known_args()
args.scale = [int(x) for x in args.scale.split("+")]
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')
if args.epochs == 0:
    args.epochs = 1e4
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
