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
"""parse args"""
import argparse
import ast
import os
import math
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_rank
from .config import get_dataset_config

def get_args(phase):
    """Define the common options that are used in both training and test."""
    parser = argparse.ArgumentParser(description='Configuration')

    # Hardware specifications
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument("--device_id", type=int, default=0, help="device id, default is 0.")
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument('--platform', type=str, default="Ascend", \
                        help='run platform, only support Ascend')
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False, \
                        help='whether save graphs, default is False.')
    parser.add_argument('--dataset', type=str, default="large", choices=("large", "small", "demo"), \
                        help='MIND dataset, support large, small and demo.')
    parser.add_argument('--dataset_path', type=str, default=None, help='MIND dataset path.')

    # Model specifications
    parser.add_argument('--n_browsed_news', type=int, default=50, help='number of browsed news per user')
    parser.add_argument('--n_words_title', type=int, default=16, help='number of words per title')
    parser.add_argument('--n_words_abstract', type=int, default=48, help='number of words per abstract')
    parser.add_argument('--word_embedding_dim', type=int, default=304, help='dimension of word embedding vector')
    parser.add_argument('--category_embedding_dim', type=int, default=112, \
                        help='dimension of category embedding vector')
    parser.add_argument('--query_vector_dim', type=int, default=208, help='dimension of the query vector in attention')
    parser.add_argument('--n_filters', type=int, default=400, help='number of filters in CNN')
    parser.add_argument('--window_size', type=int, default=3, help='size of filter in CNN')
    parser.add_argument("--checkpoint_path", type=str, default=None, \
                        help="Pre trained checkpoint path, default is None.")
    parser.add_argument('--batch_size', type=int, default=64, help='size of each batch')
    # Training specifications
    if phase == "train":
        parser.add_argument('--epochs', type=int, default=None, help='number of epochs for training')
        parser.add_argument('--lr', type=float, default=None, help='learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
        parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
        parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
        parser.add_argument('--neg_sample', type=int, default=4, help='number of negative samples in negative sampling')
        parser.add_argument("--mixed", type=ast.literal_eval, default=True, \
                            help="whether use mixed precision, default is True.")
        parser.add_argument("--sink_mode", type=ast.literal_eval, default=True, \
                            help="whether use dataset sink, default is True.")
        parser.add_argument('--print_times', type=int, default=None, help='number of print times, default is None')
        parser.add_argument("--weight_decay", type=ast.literal_eval, default=True, \
                            help="whether use weight decay, default is True.")
        parser.add_argument('--save_checkpoint', type=ast.literal_eval, default=True, \
                            help='whether save checkpoint, default is True.')
        parser.add_argument("--save_checkpoint_path", type=str, default="./checkpoint", \
                            help="Save checkpoint path, default is checkpoint.")
        parser.add_argument('--dropout_ratio', type=float, default=0.2, help='ratio of dropout')
    if phase == "eval":
        parser.add_argument('--neg_sample', type=int, default=-1, \
                            help='number of negative samples in negative sampling')
    if phase == "export":
        parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', \
                            help='file format')
        parser.add_argument('--neg_sample', type=int, default=-1, \
                            help='number of negative samples in negative sampling')
    args = parser.parse_args()
    if args.device_num > 1:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=args.save_graphs)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=args.device_num)
        init()
        args.rank = get_rank()
        args.save_checkpoint_path = os.path.join(args.save_checkpoint_path, "ckpt_" + str(args.rank))
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, device_id=args.device_id,
                            save_graphs=args.save_graphs, save_graphs_path="naml_ir")
        args.rank = 0
        args.device_num = 1
    args.phase = phase
    cfg = get_dataset_config(args.dataset)
    args.n_categories = cfg.n_categories
    args.n_sub_categories = cfg.n_sub_categories
    args.n_words = cfg.n_words
    if phase == "train":
        args.epochs = cfg.epochs * math.ceil(args.device_num ** 0.5) if args.epochs is None else args.epochs
        args.lr = cfg.lr if args.lr is None else args.lr
        args.print_times = cfg.print_times if args.print_times is None else args.print_times
    args.embedding_file = cfg.embedding_file.format(args.dataset_path)
    args.word_dict_path = cfg.word_dict_path.format(args.dataset_path)
    args.category_dict_path = cfg.category_dict_path.format(args.dataset_path)
    args.subcategory_dict_path = cfg.subcategory_dict_path.format(args.dataset_path)
    args.uid2index_path = cfg.uid2index_path.format(args.dataset_path)
    args.train_dataset_path = cfg.train_dataset_path.format(args.dataset_path)
    args.eval_dataset_path = cfg.eval_dataset_path.format(args.dataset_path)
    args_dict = vars(args)
    for key in args_dict.keys():
        print('--> {}:{}'.format(key, args_dict[key]), flush=True)
    return args
