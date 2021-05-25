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
"""Train Mobilenetv2_quant gpu on Cifar10"""


import pytest
import numpy as np
from easydict import EasyDict as ed

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.train.model import Model
from mindspore.compression.quant import QuantizationAwareTraining
from mindspore.common import set_seed

from dataset import create_dataset
from lr_generator import get_lr
from utils import Monitor, CrossEntropyWithLabelSmooth
from mobilenetV2 import mobilenetV2

config_ascend_quant = ed({
    "num_classes": 10,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 300,
    "step_threshold": 10,
    "data_load_mode": "mindata",
    "epoch_size": 1,
    "start_epoch": 200,
    "warmup_epochs": 1,
    "lr": 0.05,
    "momentum": 0.997,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "loss_scale": 1024,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 300,
    "save_checkpoint_path": "./checkpoint",
})

dataset_path = "/home/workspace/mindspore_dataset/cifar-10-batches-bin/"

@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_mobilenetv2_quant():
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    config = config_ascend_quant
    print("training configure: {}".format(config))

    epoch_size = config.epoch_size

    # define network
    network = mobilenetV2(num_classes=config.num_classes)
    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define dataset
    dataset = create_dataset(dataset_path=dataset_path,
                             config=config,
                             repeat_num=1,
                             batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(bn_fold=True,
                                          per_channel=[True, False],
                                          symmetric=[False, False])
    network = quantizer.quantize(network)

    # get learning rate
    lr = Tensor(get_lr(global_step=config.start_epoch * step_size,
                       lr_init=0,
                       lr_end=0,
                       lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size + config.start_epoch,
                       steps_per_epoch=step_size))

    # define optimization
    opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), lr, config.momentum,
                      config.weight_decay)
    # define model
    model = Model(network, loss_fn=loss, optimizer=opt)

    print("============== Starting Training ==============")
    monitor = Monitor(lr_init=lr.asnumpy(),
                      step_threshold=config.step_threshold)
    callback = [monitor]
    model.train(epoch_size, dataset, callbacks=callback,
                dataset_sink_mode=False)
    print("============== End Training ==============")
    train_time = monitor.step_mseconds
    print('train_time_used:{}'.format(train_time))
    avg_step_loss = np.mean(np.array(monitor.losses))
    print("average step loss:{}".format(avg_step_loss))
    expect_avg_step_loss = 2.32
    assert avg_step_loss < expect_avg_step_loss
    export_time_used = 960
    assert train_time < export_time_used

if __name__ == '__main__':
    test_mobilenetv2_quant()
