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
python eval.py
"""
import argparse
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet_ibn import resnet50_ibn_a
from src.loss import SoftmaxCrossEntropyExpand
from src.dataset import create_dataset_ImageNet as create_dataset
from src.lr_generator import lr_generator
from src.config import cfg

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('--eval_url', required=True, type=str, help='val data path')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=256, type=int, metavar='N',
                    help='train batch size (default: 256)')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batch size (default: 100)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', required=True, type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# Device options
parser.add_argument('--device_target', type=str, default='Ascend', choices=['GPU', 'Ascend'])
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()


if __name__ == "__main__":
    train_epoch = 1
    step = 60
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    context.set_context(device_id=args.device_id, enable_auto_mixed_precision=True)

    lr = lr_generator(cfg.lr, train_epoch, steps_per_epoch=step)
    net = resnet50_ibn_a(num_classes=cfg.class_num)
    criterion = SoftmaxCrossEntropyExpand(sparse=True)
    optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr,
                       momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={"top_1_accuracy", "top_5_accuracy"})

    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    param_dict = load_checkpoint(args.checkpoint)
    # load parameter to the network
    load_param_into_net(net, param_dict)
    # load testing dataset
    ds_eval = create_dataset(os.path.join(args.eval_url), do_train=False, repeat_num=1,
                             batch_size=cfg.test_batch, target=target)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))
