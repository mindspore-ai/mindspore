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
"""deep and cross config"""


import argparse


def argparse_init():
    """
    argparse_init
    """
    parser = argparse.ArgumentParser(description='DeepCross')
    parser.add_argument("--device_target", type=str, default="GPU", choices=["Ascend", "GPU"],
                        help="device where the code will be implemented. (Default: GPU)")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--data_path", type=str, default="./data/mindrecord",
                        help="This should be set to the same directory given to the data_download's data_dir argument")
    parser.add_argument("--epochs", type=int, default=10, help="Total train epochs")
    parser.add_argument("--full_batch", type=int, default=0, help="Enable loading the full batch ")
    parser.add_argument("--batch_size", type=int, default=16000, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16000, help="Eval batch size.")
    parser.add_argument("--field_size", type=int, default=39, help="The number of features.")
    parser.add_argument("--vocab_size", type=int, default=200000, help="The total features of dataset.")
    parser.add_argument("--emb_dim", type=int, default=30, help="The dense embedding dimension of sparse feature.")
    parser.add_argument("--deep_layer_dim", type=int, nargs='+', default=[1024, 1024],
                        help="The dimension of all deep layers.")
    parser.add_argument("--cross_layer_num", type=int, default=6, help="Cross layer num")
    parser.add_argument("--deep_layer_act", type=str, default='relu',
                        help="The activation function of all deep layers.")
    parser.add_argument("--keep_prob", type=float, default=1.0, help="The keep rate in dropout layer.")
    parser.add_argument("--dropout_flag", type=int, default=0, help="Enable dropout")
    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/", help="The location of the checkpoint file.")
    parser.add_argument("--eval_file_name", type=str, default="eval.log", help="Eval output file.")
    parser.add_argument("--loss_file_name", type=str, default="loss.log", help="Loss output file.")
    parser.add_argument("--host_device_mix", type=int, default=0, help="Enable host device mode or not")
    parser.add_argument("--dataset_type", type=str, default="mindrecord", help="tfrecord/mindrecord/hd5")
    parser.add_argument("--parameter_server", type=int, default=0, help="Open parameter server of not")
    parser.add_argument("--is_distributed", type=int, default=0, help="0/1")
    return parser


class DeepCrossConfig():
    """
    DeepCrossConfig
    """

    def __init__(self):
        self.device_target = "GPU"
        self.device_id = 0
        self.data_path = "./test_raw_data/"
        self.full_batch = False
        self.epochs = 10
        self.batch_size = 16000
        self.eval_batch_size = 16000
        self.field_size = 39
        self.vocab_size = 200000
        self.emb_dim = 27
        self.deep_layer_dim = [1024, 1024]
        self.cross_layer_num = 6
        self.deep_layer_act = 'relu'
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.keep_prob = 1.0

        self.output_path = "./output"
        self.eval_file_name = "eval.log"
        self.loss_file_name = "loss.log"
        self.ckpt_path = "./checkpoints/"
        self.host_device_mix = 0
        self.dataset_type = "mindrecord"
        self.parameter_server = 0
        self.is_distributed = 0


    def argparse_init(self):
        """
        argparse_init
        """
        parser = argparse_init()
        args, _ = parser.parse_known_args()
        self.device_target = args.device_target
        self.device_id = args.device_id
        self.data_path = args.data_path
        self.epochs = args.epochs
        self.full_batch = bool(args.full_batch)
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
        self.dropout_flag = bool(args.dropout_flag)
        self.cross_layer_num = args.cross_layer_num

        self.output_path = args.output_path
        self.eval_file_name = args.eval_file_name
        self.loss_file_name = args.loss_file_name
        self.ckpt_path = args.ckpt_path
        self.host_device_mix = args.host_device_mix
        self.dataset_type = args.dataset_type
        self.parameter_server = args.parameter_server
        self.is_distributed = args.is_distributed
