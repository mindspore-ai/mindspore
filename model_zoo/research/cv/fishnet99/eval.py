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
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.config import imagenet_cfg
from src.dataset import create_dataset_imagenet

import src.fishnet as net_ms

set_seed(1)

parser = argparse.ArgumentParser(description='fishnet99')
parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet', 'cifar10'],
                    help='dataset name.')
parser.add_argument('--device_type', type=str, default=None, help='GPU or Ascend. (Default: None)')
parser.add_argument('--checkpoint_path', type=str, default='./ckpt_0', help='Checkpoint file path')
args_opt = parser.parse_args()


class CrossEntropySmooth(_Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss_ = self.ce(logit, label)
        return loss_


if __name__ == '__main__':

    if args_opt.dataset_name == "imagenet":
        cfg = imagenet_cfg
        dataset = create_dataset_imagenet(cfg.val_data_path, 1, False)
        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)
        net = net_ms.fish99()
        model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    else:
        raise ValueError("dataset is not support.")

    if not args_opt.device_type:
        device_target = args_opt.device_type
    else:
        device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    context.set_context(device_id=cfg.device_id)

    file_list = os.listdir(args_opt.checkpoint_path)
    for filename in file_list:
        de_path = os.path.join(args_opt.checkpoint_path, filename)
        if de_path.endswith('.ckpt'):
            param_dict = load_checkpoint(de_path)
            load_param_into_net(net, param_dict)
            net.set_train(False)

            acc = model.eval(dataset)
            print(f"model {de_path}'s accuracy is {acc}")
