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
""" config. """
import argparse


def argparse_init():
    """
    argparse_init
    """
    parser = argparse.ArgumentParser(description='WideDeep')
    parser.add_argument("--data_path", type=str, default="./test_raw_data/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16000)
    parser.add_argument("--eval_batch_size", type=int, default=16000)
    parser.add_argument("--field_size", type=int, default=39)
    parser.add_argument("--vocab_size", type=int, default=184965)
    parser.add_argument("--emb_dim", type=int, default=80)
    parser.add_argument("--deep_layer_dim", type=int, nargs='+', default=[1024, 512, 256, 128])
    parser.add_argument("--deep_layer_act", type=str, default='relu')
    parser.add_argument("--keep_prob", type=float, default=1.0)

    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--eval_file_name", type=str, default="eval.log")
    parser.add_argument("--loss_file_name", type=str, default="loss.log")
    return parser


class WideDeepConfig():
    """
    WideDeepConfig
    """
    def __init__(self):
        self.data_path = "./test_raw_data/"
        self.epochs = 15
        self.batch_size = 16000
        self.eval_batch_size = 16000
        self.field_size = 39
        self.vocab_size = 184965
        self.emb_dim = 80
        self.deep_layer_dim = [1024, 512, 256, 128]
        self.deep_layer_act = 'relu'
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.keep_prob = 1.0
        self.l2_coef = 8e-5

        self.output_path = "./output"
        self.eval_file_name = "eval.log"
        self.loss_file_name = "loss.log"
        self.ckpt_path = "./checkpoints/"

    def argparse_init(self):
        """
        argparse_init
        """
        parser = argparse_init()
        args, _ = parser.parse_known_args()
        self.data_path = args.data_path
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.field_size = args.field_size
        self.vocab_size = args.vocab_size
        self.emb_dim = args.emb_dim
        self.deep_layer_dim = args.deep_layer_dim
        self.deep_layer_act = args.deep_layer_act
        self.keep_prob = args.keep_prob
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.l2_coef = 8e-5

        self.output_path = args.output_path
        self.eval_file_name = args.eval_file_name
        self.loss_file_name = args.loss_file_name
        self.ckpt_path = args.ckpt_path
