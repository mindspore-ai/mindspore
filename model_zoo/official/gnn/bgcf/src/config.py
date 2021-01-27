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
network config setting
"""
import argparse


def parser_args():
    """Config for BGCF"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="Beauty", help="choose which dataset")
    parser.add_argument("-dpath", "--datapath", type=str, default="./scripts/data_mr", help="minddata path")
    parser.add_argument("-de", "--device", type=str, default='0', help="device id")
    parser.add_argument('--Ks', type=list, default=[5, 10, 20, 100], help="top K")
    parser.add_argument('-w', '--workers', type=int, default=8, help="number of process to generate data")
    parser.add_argument("-ckpt", "--ckptpath", type=str, default="./ckpts", help="checkpoint path")

    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="optimizer parameter")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-l2", "--l2", type=float, default=0.03, help="l2 coefficient")
    parser.add_argument("-act", "--activation", type=str, default='tanh', choices=['relu', 'tanh'],
                        help="activation function")
    parser.add_argument("-ndrop", "--neighbor_dropout", type=list, default=[0.0, 0.2, 0.3],
                        help="dropout ratio for different aggregation layer")
    parser.add_argument("-log", "--log_name", type=str, default='test', help="log name")

    parser.add_argument("-e", "--num_epoch", type=int, default=600, help="epoch sizes for training")
    parser.add_argument('-input', '--input_dim', type=int, default=64, choices=[64, 128],
                        help="user and item embedding dimension")
    parser.add_argument("-b", "--batch_pairs", type=int, default=5000, help="batch size")
    parser.add_argument('--eval_interval', type=int, default=20, help="evaluation interval")

    parser.add_argument("-neg", "--num_neg", type=int, default=10, help="negative sampling rate ")
    parser.add_argument("-g1", "--raw_neighs", type=int, default=40, help="num of sampling neighbors in raw graph")
    parser.add_argument("-g2", "--gnew_neighs", type=int, default=20, help="num of sampling neighbors in sample graph")
    parser.add_argument("-emb", "--embedded_dimension", type=int, default=64, help="output embedding dim")
    parser.add_argument('--dist_reg', type=float, default=0.003, help="distance loss coefficient")

    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='device target')
    return parser.parse_args()
