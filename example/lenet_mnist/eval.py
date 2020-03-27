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
######################## train and test lenet example ########################
test lenet according to model file:
python main.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""
import os
import argparse
import mindspore.nn as nn
from dataset import create_dataset
from config import mnist_cfg as cfg
from mindspore.model_zoo.lenet import LeNet5
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore MNIST Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./MNIST_Data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, enable_mem_reuse=False)

    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    repeat_size = cfg.epoch_size
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(args.data_path, "test"), 32, 1)
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== Accuracy:{} ==============".format(acc))
