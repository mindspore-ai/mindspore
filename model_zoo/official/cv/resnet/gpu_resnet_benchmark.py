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
"""train resnet."""
import argparse
import ast
import time
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_group_size
from mindspore.common import set_seed
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
from src.resnet_gpu_benchmark import resnet50 as resnet

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--batch_size', type=str, default="256", help='Batch_size: default 256.')
parser.add_argument('--epoch_size', type=str, default="2", help='Epoch_size: default 2')
parser.add_argument('--print_per_steps', type=str, default="20", help='Print loss and time per steps: default 20')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--dataset_path', type=str, default=None, help='Imagenet dataset path')
parser.add_argument('--dtype', type=str, choices=["fp32", "fp16", "FP16", "FP32"], default="fp16",\
     help='Compute data type fp32 or fp16: default fp16')
args_opt = parser.parse_args()

set_seed(1)

class MyTimeMonitor(Callback):
    def __init__(self, batch_size, sink_size):
        super(MyTimeMonitor, self).__init__()
        self.batch_size = batch_size
        self.size = sink_size
    def step_begin(self, run_context):
        self.step_time = time.time()
    def step_end(self, run_context):
        step_mseconds = (time.time() - self.step_time) * 1000
        fps = self.batch_size / step_mseconds *1000 * self.size
        print("Epoch time: {:5.3f} ms, fps: {:d} img/sec.".format(step_mseconds, int(fps)), flush=True, end=" ")

def pad(image):
    zeros = np.zeros([224, 224, 1], dtype=np.uint8)
    output = np.concatenate((image, zeros), axis=2)
    return output

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="GPU", dtype="fp16"):
    ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=4, shuffle=True)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
        ]
    if dtype == "fp32":
        trans.append(C.HWC2CHW())
    ds = ds.map(operations=trans, input_columns="image", num_parallel_workers=4)
    if dtype == "fp16":
        ds = ds.map(operations=pad, input_columns="image", num_parallel_workers=4)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    if repeat_num > 1:
        ds = ds.repeat(repeat_num)

    return ds

def get_liner_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    for i in range(total_steps):
        if i < warmup_steps:
            lr_ = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr_ = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr_)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step

if __name__ == '__main__':
    # set args
    dev = "GPU"
    epoch_size = int(args_opt.epoch_size)
    total_batch = int(args_opt.batch_size)
    print_per_steps = int(args_opt.print_per_steps)
    compute_type = str(args_opt.dtype).lower()

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=dev, save_graphs=False)
    if args_opt.run_distribute:
        init()
        context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=[85, 160])

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=total_batch, target=dev, dtype=compute_type)
    step_size = dataset.get_dataset_size()
    if (print_per_steps > step_size or print_per_steps < 1):
        print("Arg: print_per_steps should lessequal to dataset_size ", step_size)
        print("Change to default: 20")
        print_per_steps = 20
    # define net
    net = resnet(class_num=1001, dtype=compute_type)

    # init weight
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

    # init lr
    lr = get_liner_lr(lr_init=0, lr_end=0, lr_max=0.8, warmup_epochs=0, total_epochs=epoch_size,
                      steps_per_epoch=step_size)
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': 1e-4},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9, 1e-4, 1024)
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
    # Mixed precision
    if compute_type == "fp16":
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)
    # define callbacks
    time_cb = MyTimeMonitor(total_batch, print_per_steps)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    # train model
    print("========START RESNET50 GPU BENCHMARK========")
    model.train(int(epoch_size * step_size / print_per_steps), dataset, callbacks=cb, sink_size=print_per_steps)
