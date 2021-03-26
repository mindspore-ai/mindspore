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
train and infer lenet quantization network
"""

import os
import pytest
from mindspore import context
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import load_checkpoint, load_param_into_net, export
from mindspore.train import Model
from mindspore.compression.quant import QuantizationAwareTraining
from mindspore.compression.quant.quant_utils import load_nonquant_param_into_quant_net
from dataset import create_dataset
from config import nonquant_cfg, quant_cfg
from lenet import LeNet5
from lenet_fusion import LeNet5 as LeNet5Fusion
import numpy as np

device_target = 'GPU'
data_path = "/home/workspace/mindspore_dataset/mnist"


def train_lenet():
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    cfg = nonquant_cfg
    ds_train = create_dataset(os.path.join(data_path, "train"),
                              cfg.batch_size)

    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="ckpt_lenet_noquant", config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training Lenet==============")
    model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=True)


def train_lenet_quant():
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    cfg = quant_cfg
    ckpt_path = './ckpt_lenet_noquant-10_1875.ckpt'
    ds_train = create_dataset(os.path.join(data_path, "train"), cfg.batch_size, 1)
    step_size = ds_train.get_dataset_size()

    # define fusion network
    network = LeNet5Fusion(cfg.num_classes)

    # load quantization aware network checkpoint
    param_dict = load_checkpoint(ckpt_path)
    load_nonquant_param_into_quant_net(network, param_dict)

    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(quant_delay=900,
                                          bn_fold=False,
                                          per_channel=[True, False],
                                          symmetric=[True, False])
    network = quantizer.quantize(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)

    # call back and monitor
    config_ckpt = CheckpointConfig(save_checkpoint_steps=cfg.epoch_size * step_size,
                                   keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix="ckpt_lenet_quant", config=config_ckpt)

    # define model
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(cfg['epoch_size'], ds_train, callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=True)
    print("============== End Training ==============")


def eval_quant():
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    cfg = quant_cfg
    ds_eval = create_dataset(os.path.join(data_path, "test"), cfg.batch_size, 1)
    ckpt_path = './ckpt_lenet_quant-10_937.ckpt'
    # define fusion network
    network = LeNet5Fusion(cfg.num_classes)
    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(quant_delay=0,
                                          bn_fold=False,
                                          freeze_bn=10000,
                                          per_channel=[True, False],
                                          symmetric=[True, False])
    network = quantizer.quantize(network)

    # define loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)

    # call back and monitor
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # load quantization aware network checkpoint
    param_dict = load_checkpoint(ckpt_path)
    not_load_param = load_param_into_net(network, param_dict)
    if not_load_param:
        raise ValueError("Load param into net fail!")

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval, dataset_sink_mode=True)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.98

def export_lenet():
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    cfg = quant_cfg
    # define fusion network
    network = LeNet5Fusion(cfg.num_classes)
    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(quant_delay=0,
                                          bn_fold=False,
                                          freeze_bn=10000,
                                          per_channel=[True, False],
                                          symmetric=[True, False])
    network = quantizer.quantize(network)

    # export network
    inputs = Tensor(np.ones([1, 1, cfg.image_height, cfg.image_width]), mstype.float32)
    export(network, inputs, file_name="lenet_quant", file_format='MINDIR', quant_mode='AUTO')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lenet_quant():
    train_lenet()
    train_lenet_quant()
    eval_quant()
    export_lenet()


if __name__ == "__main__":
    train_lenet_quant()
