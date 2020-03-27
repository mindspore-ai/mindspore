# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
######################## train YOLOv3 example ########################
train YOLOv3 and get network model files(.ckpt) :
python train.py --image_dir dataset/coco/coco/train2017 --anno_path dataset/coco/train_coco.txt
"""

import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer

from mindspore.model_zoo.yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
from dataset import create_yolo_dataset
from config import ConfigYOLOV3ResNet18


def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate"""
    lr_each_step = []
    lr = learning_rate
    for i in range(global_step):
        if steps:
            lr_each_step.append(lr * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(lr * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step


def init_net_param(net, init='ones'):
    """Init the parameters in net."""
    params = net.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_parameter_data(initializer(init, p.data.shape(), p.data.dtype()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv3")
    parser.add_argument("--distribute", type=bool, default=False, help="Run distribute, default is false.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--mode", type=str, default="graph", help="Run graph mode or feed mode, default is graph")
    parser.add_argument("--epoch_size", type=int, default=10, help="Epoch size, default is 10")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint file path")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5, help="Save checkpoint epochs, default is 5.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--image_dir", type=str, required=True, help="Dataset image dir.")
    parser.add_argument("--anno_path", type=str, required=True, help="Dataset anno path.")
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    context.set_context(enable_task_sink=True, enable_loop_sink=True, enable_mem_reuse=True)
    if args_opt.distribute:
        device_num = args_opt.device_num
        context.reset_auto_parallel_context()
        context.set_context(enable_hccl=True)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                          device_num=device_num)
        init()
        rank = args_opt.device_id
    else:
        context.set_context(enable_hccl=False)
        rank = 0
        device_num = 1

    loss_scale = float(args_opt.loss_scale)
    dataset = create_yolo_dataset(args_opt.image_dir, args_opt.anno_path, repeat_num=args_opt.epoch_size,
                                  batch_size=args_opt.batch_size, device_num=device_num, rank=rank)
    dataset_size = dataset.get_dataset_size()
    net = yolov3_resnet18(ConfigYOLOV3ResNet18())
    net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
    init_net_param(net, "XavierUniform")

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs)
    ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=None, config=ckpt_config)
    if args_opt.checkpoint_path != "":
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)

    lr = Tensor(get_lr(learning_rate=0.001, start_step=0, global_step=args_opt.epoch_size * dataset_size,
                       decay_step=1000, decay_rate=0.95))
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)
    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]

    model = Model(net)
    dataset_sink_mode = False
    if args_opt.mode == "graph":
        dataset_sink_mode = True
    print("Start train YOLOv3.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
