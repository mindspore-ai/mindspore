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
"""
######################## eval alexnet example ########################
eval alexnet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

import ast
import argparse
from src.config import alexnet_cifar10_cfg, alexnet_imagenet_cfg
from src.dataset import create_dataset_cifar10, create_dataset_imagenet
from src.alexnet import AlexNet
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet', 'cifar10'],
                        help='dataset name.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--dataset_sink_mode', type=ast.literal_eval,
                        default=True, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    print("============== Starting Testing ==============")

    if args.dataset_name == 'cifar10':
        cfg = alexnet_cifar10_cfg
        network = AlexNet(cfg.num_classes)
        loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
        opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
        ds_eval = create_dataset_cifar10(cfg.val_data_path, cfg.batch_size, status="test")

        param_dict = load_checkpoint(cfg.ckpt_path)
        print("load checkpoint from [{}].".format(cfg.ckpt_path))
        load_param_into_net(network, param_dict)
        network.set_train(False)

        model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})

    elif args.dataset_name == 'imagenet':
        cfg = alexnet_imagenet_cfg
        network = AlexNet(cfg.num_classes)
        if not cfg.label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean",
                                                smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)
        ds_eval = create_dataset_imagenet(cfg.val_data_path, cfg.batch_size, training=False)

        param_dict = load_checkpoint(cfg.ckpt_path)
        print("load checkpoint from [{}].".format(cfg.ckpt_path))
        load_param_into_net(network, param_dict)
        network.set_train(False)

        model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    else:
        raise ValueError("Unsupport dataset.")

    result = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("result : {}".format(result))
