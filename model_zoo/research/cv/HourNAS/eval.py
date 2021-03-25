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
import argparse

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore import context
from mindspore import nn

from src.dataset import create_dataset_cifar10
from src.utils import count_params
from src.hournasnet import hournasnet

from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data_path', type=str, default='/home/workspace/mindspore_dataset/',
                    metavar='DIR', help='path to dataset')
parser.add_argument('--model', default='hournas_f_c10', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tinynet_c"')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 10)')
parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--ckpt', type=str, default='./ms_hournas_f_c10.ckpt',
                    help='model checkpoint to load')
parser.add_argument('--GPU', action='store_true', default=True,
                    help='Use GPU for training (default: True)')
parser.add_argument('--dataset_sink', action='store_true', default=True)
parser.add_argument('--image-size', type=int, default=32, metavar='N',
                    help='input image size (default: 32)')

def main():
    """Main entrance for training"""
    args = parser.parse_args()
    print(sys.argv)

    #context.set_context(mode=context.GRAPH_MODE)
    context.set_context(mode=context.PYNATIVE_MODE)

    if args.GPU:
        context.set_context(device_target='GPU')

    # parse model argument
    assert args.model.startswith(
        "hournas"), "Only Tinynet models are supported."
    #_, sub_name = args.model.split("_")
    net = hournasnet(args.model,
                     num_classes=args.num_classes,
                     drop_rate=0.0,
                     drop_connect_rate=0.0,
                     global_pool="avg",
                     bn_tf=False,
                     bn_momentum=None,
                     bn_eps=None)
    print(net)
    print("Total number of parameters:", count_params(net))
    cfg = edict({'image_height': args.image_size, 'image_width': args.image_size,})
    cfg.batch_size = args.batch_size
    print(cfg)

    #input_size = net.default_cfg['input_size'][1]
    val_data_url = args.data_path #os.path.join(args.data_path, 'val')
    val_dataset = create_dataset_cifar10(val_data_url, repeat_num=1, training=False, cifar_cfg=cfg)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

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
