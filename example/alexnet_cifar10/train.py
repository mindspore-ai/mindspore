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
######################## train alexnet example ########################
train alexnet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import argparse
from config import alexnet_cfg as cfg
from dataset import create_dataset
import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.model_zoo.alexnet import AlexNet
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./", help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, enable_mem_reuse=False)

    network = AlexNet(cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    print("============== Starting Training ==============")
    ds_train = create_dataset(data_path=args.data_path,
                              batch_size=cfg.batch_size,
                              repeat_size=cfg.epoch_size,
                              status="train")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=args.ckpt_path, config=config_ck)
    model.train(cfg.epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode)
