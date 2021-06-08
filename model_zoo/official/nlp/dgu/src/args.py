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
Args used in Bert finetune and evaluation.
"""
import argparse

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--task_name",
        default="udc",
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--device_target",
        default="GPU",
        type=str,
        help="The device to train.")
    parser.add_argument(
        "--device_id",
        default=0,
        type=int,
        help="The device id to use.")
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-uncased.ckpt',
        type=str,
        help="Path to pre-trained bert model or shortcut name.")
    parser.add_argument(
        "--local_model_name_or_path",
        default='/cache/pretrainModel/bert-BertCLS-111.ckpt',
        type=str,
        help="local Path to pre-trained bert model or shortcut name, for online work.")
    parser.add_argument(
        "--checkpoints_path",
        default=None,
        type=str,
        help="The output directory where the checkpoints will be saved.")
    parser.add_argument(
        "--eval_ckpt_path",
        default=None,
        type=str,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--max_seq_len",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization for trainng.\
        Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        "--eval_max_seq_len",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization for evaling.\
        Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        "--learning_rate",
        default=None,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--save_steps",
        default=None,
        type=int,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="The proportion of warmup.")
    parser.add_argument(
        "--do_train", default="true", type=str, help="Whether training.")
    parser.add_argument(
        "--do_eval", default="true", type=str, help="Whether evaluation.")

    parser.add_argument(
        "--train_data_shuffle", type=str, default="true", choices=["true", "false"],
        help="Enable train data shuffle, default is true")
    parser.add_argument(
        "--train_data_file_path", type=str, default="",
        help="Data path, it is better to use absolute path")
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size, default is 32")
    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Eval batch size, default is None. if the eval_batch_size parameter is not passed in,\
        It will be assigned the same value as train_batch_size")
    parser.add_argument(
        "--eval_data_file_path", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument(
        "--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
        help="Enable eval data shuffle, default is false")

    parser.add_argument(
        "--is_modelarts_work", type=str, default="false", help="Whether modelarts online work.")
    parser.add_argument(
        "--train_url", type=str, default="",
        help="save_model path, it is better to use absolute path, for modelarts online work.")
    parser.add_argument(
        "--data_url", type=str, default="", help="data path, for modelarts online work")
    args = parser.parse_args()
    return args


def set_default_args(args):
    """set default args."""
    args.task_name = args.task_name.lower()
    if args.task_name == 'udc':
        if not args.save_steps:
            args.save_steps = 1000
        if not args.epochs:
            args.epochs = 2
        if not args.max_seq_len:
            args.max_seq_len = 224
        if not args.eval_batch_size:
            args.eval_batch_size = 100
    elif args.task_name == 'atis_intent':
        if not args.save_steps:
            args.save_steps = 100
        if not args.epochs:
            args.epochs = 20
    elif args.task_name == 'mrda':
        if not args.save_steps:
            args.save_steps = 500
        if not args.epochs:
            args.epochs = 7
    elif args.task_name == 'swda':
        if not args.save_steps:
            args.save_steps = 500
        if not args.epochs:
            args.epochs = 3
    else:
        raise ValueError('Not support task: %s.' % args.task_name)

    if not args.checkpoints_path:
        args.checkpoints_path = './checkpoints/' + args.task_name
    if not args.learning_rate:
        args.learning_rate = 2e-5
    if not args.max_seq_len:
        args.max_seq_len = 128
    if not args.eval_max_seq_len:
        args.eval_max_seq_len = args.max_seq_len
    if not args.eval_batch_size:
        args.eval_batch_size = args.train_batch_size
