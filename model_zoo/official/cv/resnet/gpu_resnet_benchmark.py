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
from mindspore.train.callback import Callback, ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from src.resnet_gpu_benchmark import resnet50 as resnet
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--batch_size', type=str, default="256", help='Batch_size: default 256.')
parser.add_argument('--epoch_size', type=str, default="2", help='Epoch_size: default 2')
parser.add_argument('--print_per_steps', type=str, default="20", help='Print loss and time per steps: default 20')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--save_ckpt', type=ast.literal_eval, default=False, help='Save ckpt or not: default False')
parser.add_argument('--eval', type=ast.literal_eval, default=False, help='Eval ckpt : default False')
parser.add_argument('--dataset_path', type=str, default=None, help='Imagenet dataset path')
parser.add_argument('--ckpt_path', type=str, default="./", help='The path to save ckpt if save_ckpt is True;\
                    Or the ckpt model file when eval is True')
parser.add_argument('--mode', type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"], help='Execute mode')
parser.add_argument('--dtype', type=str, choices=["fp32", "fp16", "FP16", "FP32"], default="fp16", \
                    help='Compute data type fp32 or fp16: default fp16')
args_opt = parser.parse_args()

set_seed(1)


class MyTimeMonitor(Callback):
    def __init__(self, batch_size, sink_size, dataset_size, mode):
        super(MyTimeMonitor, self).__init__()
        self.batch_size = batch_size
        self.size = sink_size
        self.data_size = dataset_size
        self.mode = mode

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_epoch_num = int(cb_params.cur_epoch_num / (self.data_size / self.size) +1)
        cur_step_in_epoch = int(self.size * (cb_params.cur_epoch_num % (self.data_size / self.size)))
        total_epochs = int((cb_params.epoch_num - 1) / (self.data_size / self.size) + 1)
        if self.mode == context.PYNATIVE_MODE:
            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            cur_epoch_num = cb_params.cur_epoch_num
            total_epochs = cb_params.epoch_num

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cur_epoch_num, cur_step_in_epoch))
        step_mseconds = (time.time() - self.step_time) * 1000
        fps = self.batch_size / step_mseconds * 1000 * self.size
        print("epoch: [%s/%s] step: [%s/%s], loss is %s" % (cur_epoch_num, total_epochs,\
            cur_step_in_epoch, self.data_size, loss),\
                "Epoch time: {:5.3f} ms, fps: {:d} img/sec.".format(step_mseconds, int(fps)), flush=True)


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="GPU", dtype="fp16",
                   device_num=1):
    if args_opt.mode == "GRAPH":
        ds_num_parallel_worker = 4
        map_num_parallel_worker = 8
        batch_num_parallel_worker = None
    else:
        ds_num_parallel_worker = 2
        map_num_parallel_worker = 3
        batch_num_parallel_worker = 2
    ds.config.set_numa_enable(True)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=ds_num_parallel_worker, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=ds_num_parallel_worker, shuffle=True,
                                         num_shards=device_num, shard_id=get_rank())
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    normalize_op = C.Normalize(mean=mean, std=std)
    if dtype == "fp16":
        if args_opt.eval:
            x_dtype = "float32"
        else:
            x_dtype = "float16"
        normalize_op = C.NormalizePad(mean=mean, std=std, dtype=x_dtype)
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            normalize_op,
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            normalize_op,
        ]
    if dtype == "fp32":
        trans.append(C.HWC2CHW())
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=map_num_parallel_worker)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True, num_parallel_workers=batch_num_parallel_worker)
    # apply dataset repeat operation
    if repeat_num > 1:
        data_set = data_set.repeat(repeat_num)

    return data_set


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


def train():
    # set args
    dev = "GPU"
    epoch_size = int(args_opt.epoch_size)
    total_batch = int(args_opt.batch_size)
    print_per_steps = int(args_opt.print_per_steps)
    compute_type = str(args_opt.dtype).lower()
    ckpt_save_dir = str(args_opt.ckpt_path)
    save_ckpt = bool(args_opt.save_ckpt)
    device_num = 1
    # init context
    if args_opt.mode == "GRAPH":
        mode = context.GRAPH_MODE
        all_reduce_fusion_config = [85, 160]
    else:
        mode = context.PYNATIVE_MODE
        all_reduce_fusion_config = [30, 90, 160]
    context.set_context(mode=mode, device_target=dev, save_graphs=False)
    if args_opt.run_distribute:
        init()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=all_reduce_fusion_config)
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=total_batch, target=dev, dtype=compute_type, device_num=device_num)
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

    # define loss, model
    loss = CrossEntropySmooth(sparse=True, reduction='mean', smooth_factor=0.1, num_classes=1001)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9, 1e-4)
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
    # Mixed precision
    if compute_type == "fp16":
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9, 1e-4, 1024)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)
    # define callbacks
    if mode == context.PYNATIVE_MODE:
        print_per_steps = 1
    time_cb = MyTimeMonitor(total_batch, print_per_steps, step_size, mode)
    cb = [time_cb]
    if save_ckpt:
        config_ck = CheckpointConfig(save_checkpoint_steps=5 * step_size, keep_checkpoint_max=5)
        ckpt_cb = ModelCheckpoint(prefix="resnet_benchmark", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    # train model
    print("========START RESNET50 GPU BENCHMARK========")
    if mode == context.GRAPH_MODE:
        model.train(int(epoch_size * step_size / print_per_steps), dataset, callbacks=cb, sink_size=print_per_steps)
    else:
        model.train(epoch_size, dataset, callbacks=cb)


def eval_():
    # set args
    dev = "GPU"
    compute_type = str(args_opt.dtype).lower()
    ckpt_dir = str(args_opt.ckpt_path)
    total_batch = int(args_opt.batch_size)
    # init context
    if args_opt.mode == "GRAPH":
        mode = context.GRAPH_MODE
    else:
        mode = context.PYNATIVE_MODE
    context.set_context(mode=mode, device_target=dev, save_graphs=False)
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, repeat_num=1,
                             batch_size=total_batch, target=dev, dtype=compute_type)
    # define net
    net = resnet(class_num=1001, dtype=compute_type)
    # load checkpoint
    param_dict = load_checkpoint(ckpt_dir)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # define loss, model
    loss = CrossEntropySmooth(sparse=True, reduction='mean', smooth_factor=0.1, num_classes=1001)
    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    # eval model
    print("========START EVAL RESNET50 ON GPU ========")
    res = model.eval(dataset)
    print("result:", res, "ckpt=", ckpt_dir)


if __name__ == '__main__':
    if not args_opt.eval:
        train()
    else:
        eval_()
