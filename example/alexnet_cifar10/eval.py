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

import argparse
from config import alexnet_cfg as cfg
from dataset import create_dataset
import mindspore.nn as nn
from mindspore import context
from mindspore.model_zoo.alexnet import AlexNet
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./", help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    network = AlexNet(cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    repeat_size = cfg.epoch_size
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(args.data_path,
                             cfg.batch_size,
                             1,
                             "test")
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== Accuracy:{} ==============".format(acc))
