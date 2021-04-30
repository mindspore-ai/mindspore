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
"""evaluate_imagenet"""
import argparse
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.config import config_ascend, config_gpu, config_cpu
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inceptionv4 import Inceptionv4

CFG_DICT = {
    "Ascend": config_ascend,
    "GPU": config_gpu,
    "CPU": config_cpu,
}

DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}


def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser(description='image classification evaluation')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'), help='run platform')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint of inceptionV4')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':
    args = parse_args()

    if args.platform == 'Ascend':
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    config = CFG_DICT[args.platform]
    create_dataset = DS_DICT[config.ds_type]

    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform)
    net = Inceptionv4(classes=config.num_classes)
    ckpt = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net, ckpt)
    net.set_train(False)
    config.rank = 0
    config.group_size = 1
    dataset = create_dataset(dataset_path=args.dataset_path, do_train=False, cfg=config)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss, optimizer=None, metrics=eval_metrics)
    print('=' * 20, 'Evalute start', '=' * 20)
    metrics = model.eval(dataset, dataset_sink_mode=config.ds_sink_mode)
    print("metric: ", metrics)
