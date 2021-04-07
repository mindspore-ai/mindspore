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
"""Inference Interface"""
import sys
import logging
import argparse

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore import context
from mindspore import Tensor

from src.dataset import create_dataset_cifar10
from src.loss import LabelSmoothingCrossEntropy
from src.resnet import resnet20

from easydict import EasyDict as edict

import numpy as np

root = logging.getLogger()
root.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data_path', type=str, default='/home/workspace/mindspore_dataset/',
                    metavar='DIR', help='path to dataset')
parser.add_argument('--model', default='hournas_f_c10', type=str, metavar='MODEL',
                    help='Name of model to train (default: "hournas_f_c10")')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 10)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--ckpt', type=str, default='./resnet20.ckpt',
                    help='model checkpoint to load')
parser.add_argument('--GPU', action='store_true', default=False,
                    help='Use GPU for training (default: False)')
parser.add_argument('--dataset_sink', action='store_true', default=False,
                    help='Data sink (default: False)')
parser.add_argument('--device_id', type=int, default=0,
                    help='Device ID (default: 0)')
parser.add_argument('--image-size', type=int, default=32, metavar='N',
                    help='input image size (default: 32)')

def main():
    """Main entrance for training"""
    args = parser.parse_args()
    print(sys.argv)

    context.set_context(mode=context.GRAPH_MODE)
    # context.set_context(mode=context.PYNATIVE_MODE)

    if args.GPU:
        context.set_context(device_target='GPU', device_id=args.device_id)

    # parse model argument
    assert args.model.startswith(
        "hournas"), "Only Tinynet models are supported."
    #_, sub_name = args.model.split("_")
    thres = np.load('thres.npy')
    thres = Tensor(thres.astype(np.float32))
    net = resnet20(thres=thres)

    cfg = edict({
        'image_height': args.image_size,
        'image_width': args.image_size,
    })
    #cfg.rank = 0
    #cfg.group_size = 1
    cfg.batch_size = args.batch_size

    #input_size = net.default_cfg['input_size'][1]
    val_data_url = args.data_path #os.path.join(args.data_path, 'val')
    val_dataset = create_dataset_cifar10(val_data_url, repeat_num=1, training=False, cifar_cfg=cfg)

    loss = LabelSmoothingCrossEntropy(smooth_factor=args.smoothing,
                                      num_classes=args.num_classes)

    loss.add_flags_recursive(fp32=True, fp16=False)
    eval_metrics = {'Validation-Loss': Loss(),
                    'Top1-Acc': Top1CategoricalAccuracy(),
                    'Top5-Acc': Top5CategoricalAccuracy()}

    ckpt = load_checkpoint(args.ckpt)
    load_param_into_net(net, ckpt)
    net.set_train(False)

    model = Model(net, loss, metrics=eval_metrics)

    metrics = model.eval(val_dataset, dataset_sink_mode=False)
    print(metrics)


if __name__ == '__main__':
    main()
