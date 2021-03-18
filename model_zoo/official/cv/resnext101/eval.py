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
"""Eval"""
import os
import time
import argparse
import datetime
import glob
import numpy as np
import mindspore.nn as nn

from mindspore import Tensor, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size, release
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from src.utils.logging import get_logger
from src.utils.auto_mixed_precision import auto_mixed_precision
from src.utils.var_init import load_pretrain_model
from src.image_classification import get_network
from src.dataset import classification_dataset
from src.config import config


class ParameterReduce(nn.Cell):
    """ParameterReduce"""
    def __init__(self):
        super(ParameterReduce, self).__init__()
        self.cast = P.Cast()
        self.reduce = P.AllReduce()

    def construct(self, x):
        one = self.cast(F.scalar_to_array(1.0), mstype.float32)
        out = x * one
        ret = self.reduce(out)
        return ret


def parse_args(cloud_args=None):
    """parse_args"""
    parser = argparse.ArgumentParser('mindspore classification test')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='/opt/npu/datasets/classification/val', help='eval data dir')
    parser.add_argument('--per_batch_size', default=32, type=int, help='batch size for per npu')
    # network related
    parser.add_argument('--graph_ckpt', action='store_true', default=True, help='graph ckpt or feed ckpt')
    parser.add_argument('--pretrained', default='', type=str, help='fully path of pretrained model to load. '
                        'If it is a direction, it will test all ckpt')

    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='path to save log')
    parser.add_argument('--is_distributed', action='store_true', default=False, help='if multi device')

    # roma obs
    parser.add_argument('--train_url', type=str, default="", help='train url')

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)
    args.image_size = config.image_size
    args.num_classes = config.num_classes
    args.rank = config.rank
    args.group_size = config.group_size

    args.image_size = list(map(int, args.image_size.split(',')))

    # init distributed
    if args.is_distributed:
        if args.platform == "Ascend":
            init()
        elif args.platform == "GPU":
            init("nccl")
        args.rank = get_rank()
        args.group_size = get_group_size()
    else:
        args.rank = 0
        args.group_size = 1

    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    args.logger = get_logger(args.outputs_dir, args.rank)
    return args


def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count

def merge_args(args, cloud_args):
    """merge_args"""
    args_dict = vars(args)
    if isinstance(cloud_args, dict):
        for key in cloud_args.keys():
            val = cloud_args[key]
            if key in args_dict and val:
                arg_type = type(args_dict[key])
                if arg_type is not type(None):
                    val = arg_type(val)
                args_dict[key] = val
    return args


def get_result(args, model, top1_correct, top5_correct, img_tot):
    """calculate top1 and top5 value."""
    results = [[top1_correct], [top5_correct], [img_tot]]
    args.logger.info('before results={}'.format(results))
    if args.is_distributed:
        model_md5 = model.replace('/', '')
        tmp_dir = '/cache'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        top1_correct_npy = '/cache/top1_rank_{}_{}.npy'.format(args.rank, model_md5)
        top5_correct_npy = '/cache/top5_rank_{}_{}.npy'.format(args.rank, model_md5)
        img_tot_npy = '/cache/img_tot_rank_{}_{}.npy'.format(args.rank, model_md5)
        np.save(top1_correct_npy, top1_correct)
        np.save(top5_correct_npy, top5_correct)
        np.save(img_tot_npy, img_tot)
        while True:
            rank_ok = True
            for other_rank in range(args.group_size):
                top1_correct_npy = '/cache/top1_rank_{}_{}.npy'.format(other_rank, model_md5)
                top5_correct_npy = '/cache/top5_rank_{}_{}.npy'.format(other_rank, model_md5)
                img_tot_npy = '/cache/img_tot_rank_{}_{}.npy'.format(other_rank, model_md5)
                if not os.path.exists(top1_correct_npy) or not os.path.exists(top5_correct_npy) or \
                   not os.path.exists(img_tot_npy):
                    rank_ok = False
            if rank_ok:
                break

        top1_correct_all = 0
        top5_correct_all = 0
        img_tot_all = 0
        for other_rank in range(args.group_size):
            top1_correct_npy = '/cache/top1_rank_{}_{}.npy'.format(other_rank, model_md5)
            top5_correct_npy = '/cache/top5_rank_{}_{}.npy'.format(other_rank, model_md5)
            img_tot_npy = '/cache/img_tot_rank_{}_{}.npy'.format(other_rank, model_md5)
            top1_correct_all += np.load(top1_correct_npy)
            top5_correct_all += np.load(top5_correct_npy)
            img_tot_all += np.load(img_tot_npy)
        results = [[top1_correct_all], [top5_correct_all], [img_tot_all]]
        results = np.array(results)
    else:
        results = np.array(results)

    args.logger.info('after results={}'.format(results))
    return results


def test(cloud_args=None):
    """test"""
    args = parse_args(cloud_args)
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args.platform, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # init distributed
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.group_size,
                                          gradients_mean=True)

    args.logger.save_args(args)

    # network
    args.logger.important_info('start create network')
    if os.path.isdir(args.pretrained):
        models = list(glob.glob(os.path.join(args.pretrained, '*.ckpt')))
        print(models)
        if args.graph_ckpt:
            f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])
        else:
            f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('_')[-1])
        args.models = sorted(models, key=f)
    else:
        args.models = [args.pretrained,]

    for model in args.models:
        de_dataset = classification_dataset(args.data_dir, image_size=args.image_size,
                                            per_batch_size=args.per_batch_size,
                                            max_epoch=1, rank=args.rank, group_size=args.group_size,
                                            mode='eval')
        eval_dataloader = de_dataset.create_tuple_iterator(output_numpy=True, num_epochs=1)
        network = get_network(num_classes=args.num_classes, platform=args.platform)

        load_pretrain_model(model, network, args)

        img_tot = 0
        top1_correct = 0
        top5_correct = 0
        if args.platform == "Ascend":
            network.to_float(mstype.float16)
        else:
            auto_mixed_precision(network)
        network.set_train(False)
        t_end = time.time()
        it = 0
        for data, gt_classes in eval_dataloader:
            output = network(Tensor(data, mstype.float32))
            output = output.asnumpy()

            top1_output = np.argmax(output, (-1))
            top5_output = np.argsort(output)[:, -5:]

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, gt_classes)
            img_tot += args.per_batch_size

            if args.rank == 0 and it == 0:
                t_end = time.time()
                it = 1
        if args.rank == 0:
            time_used = time.time() - t_end
            fps = (img_tot - args.per_batch_size) * args.group_size / time_used
            args.logger.info('Inference Performance: {:.2f} img/sec'.format(fps))
        results = get_result(args, model, top1_correct, top5_correct, img_tot)
        top1_correct = results[0, 0]
        top5_correct = results[1, 0]
        img_tot = results[2, 0]
        acc1 = 100.0 * top1_correct / img_tot
        acc5 = 100.0 * top5_correct / img_tot
        args.logger.info('after allreduce eval: top1_correct={}, tot={},'
                         'acc={:.2f}%(TOP1)'.format(top1_correct, img_tot, acc1))
        args.logger.info('after allreduce eval: top5_correct={}, tot={},'
                         'acc={:.2f}%(TOP5)'.format(top5_correct, img_tot, acc5))
    if args.is_distributed:
        release()


if __name__ == "__main__":
    test()
