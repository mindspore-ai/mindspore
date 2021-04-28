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
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from src.utils.logging import get_logger
from src.vgg import vgg19
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset


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
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    # dataset related
    parser.add_argument('--dataset', type=str, choices=["cifar10", "imagenet2012"], default="cifar10")
    parser.add_argument('--data_path', type=str, default='', help='eval data dir')
    parser.add_argument('--per_batch_size', default=32, type=int, help='batch size for per npu')
    # network related
    parser.add_argument('--graph_ckpt', type=int, default=1, help='graph ckpt or feed ckpt')
    parser.add_argument('--pre_trained', default='', type=str, help='fully path of pretrained model to load. '
                        'If it is a direction, it will test all ckpt')

    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='path to save log')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    args_opt = parser.parse_args()
    args_opt = merge_args(args_opt, cloud_args)

    if args_opt.dataset == "cifar10":
        from src.config import cifar_cfg as cfg
    else:
        from src.config import imagenet_cfg as cfg

    args_opt.image_size = cfg.image_size
    args_opt.num_classes = cfg.num_classes
    args_opt.per_batch_size = cfg.batch_size
    args_opt.momentum = cfg.momentum
    args_opt.weight_decay = cfg.weight_decay
    args_opt.buffer_size = cfg.buffer_size
    args_opt.pad_mode = cfg.pad_mode
    args_opt.padding = cfg.padding
    args_opt.has_bias = cfg.has_bias
    args_opt.batch_norm = cfg.batch_norm
    args_opt.initialize_mode = cfg.initialize_mode
    args_opt.has_dropout = cfg.has_dropout

    args_opt.image_size = list(map(int, args_opt.image_size.split(',')))

    return args_opt


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


def test(cloud_args=None):
    """test"""
    args = parse_args(cloud_args)
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args.device_target, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit() and args.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    args.logger = get_logger(args.outputs_dir, args.rank)
    args.logger.save_args(args)

    if args.dataset == "cifar10":
        net = vgg19(num_classes=args.num_classes, args=args)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, args.momentum,
                       weight_decay=args.weight_decay)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

        param_dict = load_checkpoint(args.pre_trained)
        load_param_into_net(net, param_dict)
        net.set_train(False)
        dataset = vgg_create_dataset(args.data_path, args.image_size, args.per_batch_size, training=False)
        res = model.eval(dataset)
        print("result: ", res)
    else:
        # network
        args.logger.important_info('start create network')
        if os.path.isdir(args.pre_trained):
            models = list(glob.glob(os.path.join(args.pre_trained, '*.ckpt')))
            print(models)
            if args.graph_ckpt:
                f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])
            else:
                f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('_')[-1])
            args.models = sorted(models, key=f)
        else:
            args.models = [args.pre_trained,]

        for model in args.models:
            dataset = classification_dataset(args.data_path, args.image_size, args.per_batch_size, mode='eval')
            eval_dataloader = dataset.create_tuple_iterator(output_numpy=True, num_epochs=1)
            network = vgg19(args.num_classes, args, phase="test")

            # pre_trained
            load_param_into_net(network, load_checkpoint(model))
            network.add_flags_recursive(fp16=True)

            img_tot = 0
            top1_correct = 0
            top5_correct = 0

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
            results = [[top1_correct], [top5_correct], [img_tot]]
            args.logger.info('before results={}'.format(results))
            results = np.array(results)

            args.logger.info('after results={}'.format(results))
            top1_correct = results[0, 0]
            top5_correct = results[1, 0]
            img_tot = results[2, 0]
            acc1 = 100.0 * top1_correct / img_tot
            acc5 = 100.0 * top5_correct / img_tot
            args.logger.info('after allreduce eval: top1_correct={}, tot={},'
                             'acc={:.2f}%(TOP1)'.format(top1_correct, img_tot, acc1))
            args.logger.info('after allreduce eval: top5_correct={}, tot={},'
                             'acc={:.2f}%(TOP5)'.format(top5_correct, img_tot, acc5))


if __name__ == "__main__":
    test()
