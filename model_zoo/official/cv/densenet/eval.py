# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
##############test densenet example#################
python eval.py --net densenet121 --dataset imagenet --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/CHECKPOINT
"""

import os
import argparse
import datetime
import glob
import numpy as np
from mindspore import context

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size, release
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from src.utils.logging import get_logger

class ParameterReduce(nn.Cell):
    """
    reduce parameter
    """
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
    """
    parse args
    """
    parser = argparse.ArgumentParser('mindspore classification test')

    # network and dataset choices
    parser.add_argument('--net', type=str, default='', help='Densenet Model, densenet100 or densenet121')
    parser.add_argument('--dataset', type=str, default='', help='Dataset, either cifar10 or imagenet')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='eval data dir')

    # network related
    parser.add_argument('--backbone', default='resnet50', help='backbone')
    parser.add_argument('--pretrained', default='', type=str, help='fully path of pretrained model to load.'
                                                                   'If it is a direction, it will test all ckpt')

    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='path to save log')
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    # roma obs
    parser.add_argument('--train_url', type=str, default="", help='train url')

    # platform
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'),
                        help='device target')

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)

    if args.net == "densenet100":
        from src.config import config_100 as config
    else:
        from src.config import config_121 as config

    args.per_batch_size = config.per_batch_size
    args.image_size = config.image_size
    args.num_classes = config.num_classes

    args.image_size = list(map(int, args.image_size.split(',')))

    return args


def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count

def merge_args(args, cloud_args):
    """
    merge args and cloud_args
    """
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

def generate_results(model, rank, group_size, top1_correct, top5_correct, img_tot):
    model_md5 = model.replace('/', '')
    tmp_dir = '../cache'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    top1_correct_npy = '{}/top1_rank_{}_{}.npy'.format(tmp_dir, rank, model_md5)
    top5_correct_npy = '{}/top5_rank_{}_{}.npy'.format(tmp_dir, rank, model_md5)
    img_tot_npy = '{}/img_tot_rank_{}_{}.npy'.format(tmp_dir, rank, model_md5)
    np.save(top1_correct_npy, top1_correct)
    np.save(top5_correct_npy, top5_correct)
    np.save(img_tot_npy, img_tot)
    while True:
        rank_ok = True
        for other_rank in range(group_size):
            top1_correct_npy = '{}/top1_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
            top5_correct_npy = '{}/top5_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
            img_tot_npy = '{}/img_tot_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
            if not os.path.exists(top1_correct_npy) or not os.path.exists(top5_correct_npy) \
               or not os.path.exists(img_tot_npy):
                rank_ok = False
        if rank_ok:
            break

    top1_correct_all = 0
    top5_correct_all = 0
    img_tot_all = 0
    for other_rank in range(group_size):
        top1_correct_npy = '{}/top1_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
        top5_correct_npy = '{}/top5_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
        img_tot_npy = '{}/img_tot_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
        top1_correct_all += np.load(top1_correct_npy)
        top5_correct_all += np.load(top5_correct_npy)
        img_tot_all += np.load(img_tot_npy)
    return [[top1_correct_all], [top5_correct_all], [img_tot_all]]

def test(cloud_args=None):
    """
    network eval function. Get top1 and top5 ACC from classification for imagenet,
    and top1 ACC for cifar10.
    The result will be save at [./outputs] by default.
    """
    args = parse_args(cloud_args)

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=True)
    if args.device_target == 'Ascend':
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=devid)

    # init distributed
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    args.logger = get_logger(args.outputs_dir, args.rank)
    args.logger.save_args(args)

    # network
    args.logger.important_info('start create network')
    if os.path.isdir(args.pretrained):
        models = list(glob.glob(os.path.join(args.pretrained, '*.ckpt')))

        f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])

        args.models = sorted(models, key=f)
    else:
        args.models = [args.pretrained,]

    if args.net == "densenet100":
        from src.network.densenet import DenseNet100 as DenseNet
    else:
        from src.network.densenet import DenseNet121 as DenseNet

    if args.dataset == "cifar10":
        from src.datasets import classification_dataset_cifar10 as classification_dataset
    else:
        from src.datasets import classification_dataset_imagenet as classification_dataset

    for model in args.models:
        de_dataset = classification_dataset(args.data_dir, image_size=args.image_size,
                                            per_batch_size=args.per_batch_size,
                                            max_epoch=1, rank=args.rank, group_size=args.group_size,
                                            mode='eval')
        eval_dataloader = de_dataset.create_tuple_iterator()
        network = DenseNet(args.num_classes)

        param_dict = load_checkpoint(model)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load model {} success'.format(model))

        if args.device_target == 'Ascend':
            network.add_flags_recursive(fp16=True)

        img_tot = 0
        top1_correct = 0
        top5_correct = 0
        network.set_train(False)
        for data, gt_classes in eval_dataloader:
            output = network(Tensor(data, mstype.float32))
            output = output.asnumpy()
            gt_classes = gt_classes.asnumpy()

            top1_output = np.argmax(output, (-1))
            top5_output = np.argsort(output)[:, -5:]

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, gt_classes)
            img_tot += args.per_batch_size

        results = [[top1_correct], [top5_correct], [img_tot]]
        args.logger.info('before results={}'.format(results))
        if args.is_distributed:
            results = generate_results(model, args.rank, args.group_size, top1_correct,
                                       top5_correct, img_tot)
            results = np.array(results)
        else:
            results = np.array(results)

        args.logger.info('after results={}'.format(results))
        top1_correct = results[0, 0]
        top5_correct = results[1, 0]
        img_tot = results[2, 0]
        acc1 = 100.0 * top1_correct / img_tot
        acc5 = 100.0 * top5_correct / img_tot
        args.logger.info('after allreduce eval: top1_correct={}, tot={}, acc={:.2f}%'.format(top1_correct, img_tot,
                                                                                             acc1))
        if args.dataset == 'imagenet':
            args.logger.info('after allreduce eval: top5_correct={}, tot={}, acc={:.2f}%'.format(top5_correct, img_tot,
                                                                                                 acc5))
    if args.is_distributed:
        release()

if __name__ == "__main__":
    test()
