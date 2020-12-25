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
##############test tinydarknet example on cifar10#################
python eval.py
"""
import argparse


from mindspore import context

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.config import imagenet_cfg
from src.dataset import create_dataset_imagenet

from src.tinydarknet import TinyDarkNet
from src.CrossEntropySmooth import CrossEntropySmooth

set_seed(1)

parser = argparse.ArgumentParser(description='tinydarknet')
parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet', 'cifar10'],
                    help='dataset name.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
args_opt = parser.parse_args()

if __name__ == '__main__':

    if args_opt.dataset_name == "imagenet":
        cfg = imagenet_cfg
        dataset = create_dataset_imagenet(cfg.val_data_path, 1, False)
        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)
        net = TinyDarkNet(num_classes=cfg.num_classes)
        model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    else:
        raise ValueError("dataset is not support.")

    device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)

    if args_opt.checkpoint_path is not None:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        print("load checkpoint from [{}].".format(args_opt.checkpoint_path))
    else:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        print("load checkpoint from [{}].".format(cfg.checkpoint_path))

    load_param_into_net(net, param_dict)
    net.set_train(False)

    acc = model.eval(dataset)
    print("accuracy: ", acc)
