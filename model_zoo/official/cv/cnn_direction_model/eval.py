# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""test direction model."""
import argparse
import os
import random

import numpy as np
from src.cnn_direction_model import CNNDirectionModel
from src.config import config1 as config
from src.dataset import create_dataset_eval

from mindspore import context
from mindspore import dataset as de
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='Image classification')

parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if __name__ == '__main__':
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

    # create dataset
    dataset_name = config.dataset_name
    dataset_lr, dataset_rl = create_dataset_eval(args_opt.dataset_path +  "/" + dataset_name +
                                                 ".mindrecord0", config=config, dataset_name=dataset_name)
    step_size = dataset_lr.get_dataset_size()

    print("step_size ", step_size)

    # define net
    net = CNNDirectionModel([3, 64, 48, 48, 64], [64, 48, 48, 64, 64], [256, 64], [64, 512])

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="sum")

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy'})

    # eval model
    res_lr = model.eval(dataset_lr, dataset_sink_mode=False)
    res_rl = model.eval(dataset_rl, dataset_sink_mode=False)
    print("result on upright images:", res_lr, "ckpt=", args_opt.checkpoint_path)
    print("result on 180 degrees rotated images:", res_rl, "ckpt=", args_opt.checkpoint_path)
