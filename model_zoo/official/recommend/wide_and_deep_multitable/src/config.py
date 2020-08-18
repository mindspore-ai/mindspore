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

    parser.add_argument("--data_path", type=str, default="./test_raw_data/")  # The location of the input data.
    parser.add_argument("--epochs", type=int, default=200)  # The number of epochs used to train.
    parser.add_argument("--batch_size", type=int, default=131072)  # Batch size for training and evaluation
    parser.add_argument("--eval_batch_size", type=int, default=131072)  # The batch size used for evaluation.
    parser.add_argument("--deep_layers_dim", type=int, nargs='+', default=[1024, 512, 256, 128])  # The sizes of hidden layers for MLP
    parser.add_argument("--deep_layers_act", type=str, default='relu')  # The act of hidden layers for MLP
    parser.add_argument("--keep_prob", type=float, default=1.0)  # The Embedding size of MF model.
    parser.add_argument("--adam_lr", type=float, default=0.003)  # The Adam lr
    parser.add_argument("--ftrl_lr", type=float, default=0.1)  # The ftrl lr.
    parser.add_argument("--l2_coef", type=float, default=0.0)  # The l2 coefficient.
    parser.add_argument("--is_tf_dataset", type=bool, default=True)  # The l2 coefficient.

    parser.add_argument("--output_path", type=str, default="./output/")  # The location of the output file.
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")  # The location of the checkpoints file.
    parser.add_argument("--eval_file_name", type=str, default="eval.log")  # Eval output file.
    parser.add_argument("--loss_file_name", type=str, default="loss.log")  # Loss output file.
    return parser


class WideDeepConfig():
    """
    WideDeepConfig
    """
    def __init__(self):
        self.data_path = ''
        self.epochs = 200
        self.batch_size = 131072
        self.eval_batch_size = 131072
        self.deep_layers_act = 'relu'
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.keep_prob = 1.0
        self.l2_coef = 0.0

        self.adam_lr = 0.003

        self.ftrl_lr = 0.1

        self.is_tf_dataset = True
        self.input_emb_dim = 0
        self.output_path = "./output/"
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
        self.deep_layers_act = args.deep_layers_act
        self.keep_prob = args.keep_prob
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.l2_coef = args.l2_coef
        self.ftrl_lr = args.ftrl_lr
        self.adam_lr = args.adam_lr
        self.is_tf_dataset = args.is_tf_dataset

        self.output_path = args.output_path
        self.eval_file_name = args.eval_file_name
        self.loss_file_name = args.loss_file_name
        self.ckpt_path = args.ckpt_path
