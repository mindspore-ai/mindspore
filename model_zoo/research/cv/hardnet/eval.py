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
##############test hardnet example on imagenet#################
python3 eval.py
"""
import argparse
import random
import numpy as np
from mindspore import context
from mindspore import dataset
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model

from src.dataset import create_dataset_ImageNet
from src.HarDNet import HarDNet85
from src.EntropyLoss import CrossEntropySmooth
from src.config import config

random.seed(1)
np.random.seed(1)
dataset.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')

parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='if mode is test, must provide path where the trained ckpt file')
parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='label_smooth_factor')
parser.add_argument('--device_id', type=int, default=0, help='device_id')
args = parser.parse_args()

def test(ckpt_path):
    """run eval"""
    target = args.device_target
    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        save_graphs=False)

    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    # dataset
    predict_data = create_dataset_ImageNet(dataset_path=args.dataset_path,
                                           do_train=False,
                                           repeat_num=1,
                                           batch_size=config.batch_size,
                                           target=target)
    step_size = predict_data.get_dataset_size()
    if step_size == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    # define net
    network = HarDNet85(num_classes=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)

    # define loss, model
    loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                              num_classes=config.class_num)

    model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    print("Dataset path: {}".format(args.dataset_path))
    print("Ckpt path :{}".format(ckpt_path))
    print("Class num: {}".format(config.class_num))
    print("Backbone hardnet")
    print("============== Starting Testing ==============")
    acc = model.eval(predict_data)
    print("==============Acc: {} ==============".format(acc))


if __name__ == '__main__':
    path = args.ckpt_path
    test(path)
