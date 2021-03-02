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
######################## train YOLOv3 example ########################
train YOLOv3 and get network model files(.ckpt) :
python train.py --image_dir /data --anno_path /data/coco/train_coco.txt --mindrecord_dir=/data/Mindrecord_train

If the mindrecord_dir is empty, it wil generate mindrecord file by image_dir and anno_path.
Note if mindrecord_dir isn't empty, it will use mindrecord_dir rather than image_dir and anno_path.
"""

import os
import time
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.common.initializer import initializer
from mindspore.train.callback import Callback

from src.yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
from src.dataset import create_yolo_dataset
from src.config import ConfigYOLOV3ResNet18

np.random.seed(1)
def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate."""
    lr_each_step = []
    for i in range(global_step):
        if steps:
            lr_each_step.append(learning_rate * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(learning_rate * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step


def init_net_param(network, init_value='ones'):
    """Init:wq the parameters in network."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_data(initializer(init_value, p.data.shape, p.data.dtype))

class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs.asnumpy())
        print("epoch: {}, outputs are: {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))

class TimeMonitor(Callback):
    """Time Monitor."""
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_mseconds_list = []
        self.per_step_mseconds_list = []
    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds_list.append(epoch_mseconds)
        self.per_step_mseconds_list.append(epoch_mseconds / self.data_size)

DATA_DIR = "/home/workspace/mindspore_dataset/coco/coco2017/mindrecord_train/yolov3"

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_yolov3():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    rank = 0
    device_num = 1
    lr_init = 0.001
    epoch_size = 5
    batch_size = 32
    loss_scale = 1024
    mindrecord_dir = DATA_DIR

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is yolo.mindrecord0, 1, ... file_num.
    if not os.path.isdir(mindrecord_dir):
        raise KeyError("mindrecord path is not exist.")

    prefix = "yolo.mindrecord"
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("yolov3 mindrecord is ", mindrecord_file)
    if not os.path.exists(mindrecord_file):
        print("mindrecord file is not exist.")
        assert False
    else:
        loss_scale = float(loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
        dataset = create_yolo_dataset(mindrecord_file, repeat_num=1,
                                      batch_size=batch_size, device_num=device_num, rank=rank)
        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        net = yolov3_resnet18(ConfigYOLOV3ResNet18())
        net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())

        total_epoch_size = 60
        lr = Tensor(get_lr(learning_rate=lr_init, start_step=0,
                           global_step=total_epoch_size * dataset_size,
                           decay_step=1000, decay_rate=0.95, steps=True))
        opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

        model_callback = ModelCallback()
        time_monitor_callback = TimeMonitor(data_size=dataset_size)
        callback = [model_callback, time_monitor_callback]

        model = Model(net)
        print("Start train YOLOv3, the first epoch will be slower because of the graph compilation.")
        model.train(epoch_size, dataset, callbacks=callback, dataset_sink_mode=True,
                    sink_size=dataset.get_dataset_size())
        # assertion occurs while the loss value, overflow state or loss_scale value is wrong
        loss_value = np.array(model_callback.loss_list)

        expect_loss_value = [6850, 4250, 2750]
        print("loss value: {}".format(loss_value))
        assert loss_value[0] < expect_loss_value[0]
        assert loss_value[1] < expect_loss_value[1]
        assert loss_value[2] < expect_loss_value[2]

        epoch_mseconds0 = np.array(time_monitor_callback.epoch_mseconds_list)[2]
        epoch_mseconds1 = np.array(time_monitor_callback.epoch_mseconds_list)[3]
        epoch_mseconds2 = np.array(time_monitor_callback.epoch_mseconds_list)[4]
        expect_epoch_mseconds = 1250
        print("epoch mseconds: {}".format(epoch_mseconds0))
        assert epoch_mseconds0 <= expect_epoch_mseconds or \
               epoch_mseconds1 <= expect_epoch_mseconds or \
               epoch_mseconds2 <= expect_epoch_mseconds

        per_step_mseconds0 = np.array(time_monitor_callback.per_step_mseconds_list)[2]
        per_step_mseconds1 = np.array(time_monitor_callback.per_step_mseconds_list)[3]
        per_step_mseconds2 = np.array(time_monitor_callback.per_step_mseconds_list)[4]
        expect_per_step_mseconds = 130
        print("per step mseconds: {}".format(per_step_mseconds0))
        assert per_step_mseconds0 <= expect_per_step_mseconds or \
               per_step_mseconds1 <= expect_per_step_mseconds or \
               per_step_mseconds2 <= expect_per_step_mseconds
        print("yolov3 test case passed.")
