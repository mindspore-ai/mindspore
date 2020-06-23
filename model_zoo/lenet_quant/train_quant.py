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
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.quant import quant
from src.dataset import create_dataset
from src.config import mnist_cfg as cfg
from src.lenet_fusion import LeNet5 as LeNet5Fusion

parser = argparse.ArgumentParser(description='MindSpore MNIST Example')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--data_path', type=str, default="./MNIST_Data",
                    help='path where the dataset is saved')
parser.add_argument('--ckpt_path', type=str, default="",
                    help='if mode is test, must provide path where the trained ckpt file')
parser.add_argument('--dataset_sink_mode', type=bool, default=True,
                    help='dataset_sink_mode is False or True')
args = parser.parse_args()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"), cfg.batch_size, cfg.epoch_size)
    step_size = ds_train.get_dataset_size()

    # define fusion network
    network = LeNet5Fusion(cfg.num_classes)
    # load quantization aware network checkpoint
    param_dict = load_checkpoint(args.ckpt_path, network.type)
    load_param_into_net(network, param_dict)
    # convert fusion network to quantization aware network
    network = quant.convert_quant_network(network, quant_delay=0, bn_fold=False, freeze_bn=10000)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)

    # call back and monitor
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ckpt = CheckpointConfig(save_checkpoint_steps=cfg.epoch_size * step_size,
                                   keep_checkpoint_max=cfg.keep_checkpoint_max,
                                   model_type="quant")
    ckpt_callback = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ckpt)

    # define model
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, ckpt_callback, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode)
    print("============== End Training ==============")
