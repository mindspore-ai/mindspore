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
"""
network config setting, will be used in train.py
"""
import argparse


def parser_args():
    """Config for BGCF"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="Beauty")
    parser.add_argument("-dpath", "--datapath", type=str, default="./scripts/data_mr")
    parser.add_argument("-de", "--device", type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Ks', type=list, default=[5, 10, 20, 100])
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=None)
    parser.add_argument('-w', '--workers', type=int, default=10)

    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-l2", "--l2", type=float, default=0.03)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01)
    parser.add_argument("-act", "--activation", type=str, default='tanh', choices=['relu', 'tanh'])
    parser.add_argument("-ndrop", "--neighbor_dropout", type=list, default=[0.0, 0.2, 0.3])
    parser.add_argument("-log", "--log_name", type=str, default='test')

    parser.add_argument("-e", "--num_epoch", type=int, default=600)
    parser.add_argument('-input', '--input_dim', type=int, default=64, choices=[64, 128])
    parser.add_argument("-b", "--batch_pairs", type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=20)

    parser.add_argument("-neg", "--num_neg", type=int, default=10)
    parser.add_argument('-max', '--max_degree', type=str, default='[128,128]')
    parser.add_argument("-g1", "--raw_neighs", type=int, default=40)
    parser.add_argument("-g2", "--gnew_neighs", type=int, default=20)
    parser.add_argument("-emb", "--embedded_dimension", type=int, default=64)
    parser.add_argument('-dist', '--distance', type=str, default='iou')
    parser.add_argument('--dist_reg', type=float, default=0.003)

    parser.add_argument('-ng', '--num_graphs', type=int, default=5)
    parser.add_argument('-geps', '--graph_epsilon', type=float, default=0.01)

    return parser.parse_args()
