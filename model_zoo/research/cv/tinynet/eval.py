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
# ============================================================================
"""Inference Interface"""
import sys
import os
import argparse

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore import context

from src.dataset import create_dataset_val
from src.utils import count_params
from src.loss import LabelSmoothingCrossEntropy
from src.tinynet import tinynet

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data_path', type=str, default='/home/dataset/imagenet_jpeg/',
                    metavar='DIR', help='path to dataset')
parser.add_argument('--model', default='tinynet_c', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tinynet_c"')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--ckpt', type=str, default=None,
                    help='model checkpoint to load')
parser.add_argument('--GPU', action='store_true', default=True,
                    help='Use GPU for training (default: True)')
parser.add_argument('--dataset_sink', action='store_true', default=True)


def main():
    """Main entrance for training"""
    args = parser.parse_args()
    print(sys.argv)

    context.set_context(mode=context.GRAPH_MODE)

    if args.GPU:
        context.set_context(device_target='GPU')

    # parse model argument
    assert args.model.startswith(
        "tinynet"), "Only Tinynet models are supported."
    _, sub_name = args.model.split("_")
    net = tinynet(sub_model=sub_name,
                  num_classes=args.num_classes,
                  drop_rate=0.0,
                  drop_connect_rate=0.0,
                  global_pool="avg",
                  bn_tf=False,
                  bn_momentum=None,
                  bn_eps=None)
    print("Total number of parameters:", count_params(net))

    input_size = net.default_cfg['input_size'][1]
    val_data_url = os.path.join(args.data_path, 'val')
    val_dataset = create_dataset_val(args.batch_size,
                                     val_data_url,
                                     workers=args.workers,
                                     distributed=False,
                                     input_size=input_size)

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
