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
"""train."""
import argparse
import time
import pytest
import numpy as np
from mindspore import context, Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore import Model
from mindspore.train.callback import Callback
from src.md_dataset import create_dataset
from src.losses import OhemLoss
from src.deeplabv3 import deeplabv3_resnet50
from src.config import config

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
#--train
#--eval
#  --Images
#    --2008_001135.jpg
#    --2008_001404.jpg
#  --SegmentationClassRaw
#    --2008_001135.png
#    --2008_001404.png
data_url = "/home/workspace/mindspore_dataset/voc/voc2012"
class LossCallBack(Callback):
    """
    Monitor the loss in training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, data_size, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self.data_size = data_size
        self._per_print_times = per_print_times
        self.time = 1000
        self.loss = 0
    def epoch_begin(self, run_context):
        self.epoch_time = time.time()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.time = epoch_mseconds / self.data_size
        self.loss = cb_params.net_outputs
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)))

def model_fine_tune(train_net, fix_weight_layer):
    for para in train_net.trainable_params():
        para.set_parameter_data(Tensor(np.ones(para.data.shape).astype(np.float32) * 0.02))
        if fix_weight_layer in para.name:
            para.requires_grad = False

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_deeplabv3_1p():
    start_time = time.time()
    epoch_size = 100
    args_opt = argparse.Namespace(base_size=513, crop_size=513, batch_size=2)
    args_opt.base_size = config.crop_size
    args_opt.crop_size = config.crop_size
    args_opt.batch_size = config.batch_size
    train_dataset = create_dataset(args_opt, data_url, epoch_size, config.batch_size,
                                   usage="eval")
    dataset_size = train_dataset.get_dataset_size()
    callback = LossCallBack(dataset_size)
    net = deeplabv3_resnet50(config.seg_num_classes, [config.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)
    net.set_train()
    model_fine_tune(net, 'layer')
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    opt = Momentum(filter(lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'depth' not in x.name and 'bias' not in x.name, net.trainable_params()), learning_rate=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    model = Model(net, loss, opt)
    model.train(epoch_size, train_dataset, callback)
    print(time.time() - start_time)
    print("expect loss: ", callback.loss)
    print("expect time: ", callback.time)
    expect_loss = 0.92
    expect_time = 40
    assert callback.loss.asnumpy() <= expect_loss
    assert callback.time <= expect_time
