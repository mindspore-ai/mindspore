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
"""Train Resnet50_quant on Cifar10"""

import pytest
import numpy as np
from easydict import EasyDict as ed

from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.compression.quant import QuantizationAwareTraining
from mindspore import set_seed

from resnet_quant_manual import resnet50_quant
from dataset import create_dataset
from lr_generator import get_lr
from utils import Monitor, CrossEntropy


config_quant = ed({
    "class_num": 10,
    "batch_size": 128,
    "step_threshold": 20,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 1,
    "pretrained_epoch_size": 90,
    "buffer_size": 1000,
    "image_height": 224,
    "image_width": 224,
    "data_load_mode": "original",
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 50,
    "save_checkpoint_path": "./",
    "warmup_epochs": 0,
    "lr_decay_mode": "cosine",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0,
    "lr_max": 0.005,
})

dataset_path = "/home/workspace/mindspore_dataset/cifar-10-batches-bin/"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resnet50_quant():
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    config = config_quant
    print("training configure: {}".format(config))
    epoch_size = config.epoch_size

    # define network
    net = resnet50_quant(class_num=config.class_num)
    net.set_train(True)

    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(
        smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    #loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    # define dataset
    dataset = create_dataset(dataset_path=dataset_path,
                             config=config,
                             repeat_num=1,
                             batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(bn_fold=True,
                                          per_channel=[True, False],
                                          symmetric=[True, False])
    net = quantizer.quantize(net)

    # get learning rate
    lr = Tensor(get_lr(lr_init=config.lr_init,
                       lr_end=0.0,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size,
                       lr_decay_mode='cosine'))

    # define optimization
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)

    # define model
    #model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
    model = Model(net, loss_fn=loss, optimizer=opt)

    print("============== Starting Training ==============")
    monitor = Monitor(lr_init=lr.asnumpy(),
                      step_threshold=config.step_threshold)

    callbacks = [monitor]
    model.train(epoch_size, dataset, callbacks=callbacks,
                dataset_sink_mode=False)
    print("============== End Training ==============")

    expect_avg_step_loss = 2.60
    avg_step_loss = np.mean(np.array(monitor.losses))

    print("average step loss:{}".format(avg_step_loss))
    assert avg_step_loss < expect_avg_step_loss


if __name__ == '__main__':
    test_resnet50_quant()
