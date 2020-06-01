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
"""train_imagenet."""
import os
import time
import argparse
import random
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine as de
from mindspore.communication.management import init, get_group_size
from src.dataset import create_dataset
from src.lr_generator import get_lr
from src.config import config_gpu, config_ascend
from src.mobilenetV3 import mobilenet_v3_large

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--platform', type=str, default=None, help='run platform')
args_opt = parser.parse_args()

if args_opt.platform == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id = int(os.getenv('RANK_ID'))
    rank_size = int(os.getenv('RANK_SIZE'))
    run_distribute = rank_size > 1
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id, save_graphs=False)
elif args_opt.platform == "GPU":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="GPU", save_graphs=False)
else:
    raise ValueError("Unsupport platform.")


class CrossEntropyWithLabelSmooth(_Loss):
    """
    CrossEntropyWith LabelSmooth.

    Args:
        smooth_factor (float): smooth factor, default=0.
        num_classes (int): num classes

    Returns:
        None.

    Examples:
        >>> CrossEntropyWithLabelSmooth(smooth_factor=0., num_classes=1000)
    """

    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor /
                                (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


if __name__ == '__main__':
    if args_opt.platform == "GPU":
        # train on gpu
        print("train args: ", args_opt, "\ncfg: ", config_gpu)

        init('nccl')
        context.set_auto_parallel_context(parallel_mode="data_parallel",
                                          mirror_mean=True,
                                          device_num=get_group_size())

        # define net
        net = mobilenet_v3_large(num_classes=config_gpu.num_classes)
        # define loss
        if config_gpu.label_smooth > 0:
            loss = CrossEntropyWithLabelSmooth(
                smooth_factor=config_gpu.label_smooth, num_classes=config_gpu.num_classes)
        else:
            loss = SoftmaxCrossEntropyWithLogits(
                is_grad=False, sparse=True, reduction='mean')
        # define dataset
        epoch_size = config_gpu.epoch_size
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 config=config_gpu,
                                 platform=args_opt.platform,
                                 repeat_num=epoch_size,
                                 batch_size=config_gpu.batch_size)
        step_size = dataset.get_dataset_size()
        # resume
        if args_opt.pre_trained:
            param_dict = load_checkpoint(args_opt.pre_trained)
            load_param_into_net(net, param_dict)
        # define optimizer
        loss_scale = FixedLossScaleManager(
            config_gpu.loss_scale, drop_overflow_update=False)
        lr = Tensor(get_lr(global_step=0,
                           lr_init=0,
                           lr_end=0,
                           lr_max=config_gpu.lr,
                           warmup_epochs=config_gpu.warmup_epochs,
                           total_epochs=epoch_size,
                           steps_per_epoch=step_size))
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config_gpu.momentum,
                       config_gpu.weight_decay, config_gpu.loss_scale)
        # define model
        model = Model(net, loss_fn=loss, optimizer=opt,
                      loss_scale_manager=loss_scale)

        cb = [Monitor(lr_init=lr.asnumpy())]
        if config_gpu.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config_gpu.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config_gpu.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(
                prefix="mobilenetV3", directory=config_gpu.save_checkpoint_path, config=config_ck)
            cb += [ckpt_cb]
        # begine train
        model.train(epoch_size, dataset, callbacks=cb)
    elif args_opt.platform == "Ascend":
        # train on ascend
        print("train args: ", args_opt, "\ncfg: ", config_ascend,
              "\nparallel args: rank_id {}, device_id {}, rank_size {}".format(rank_id, device_id, rank_size))

        if run_distribute:
            context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              parameter_broadcast=True, mirror_mean=True)
            auto_parallel_context().set_all_reduce_fusion_split_indices([140])
            init()

        epoch_size = config_ascend.epoch_size
        net = mobilenet_v3_large(num_classes=config_ascend.num_classes)
        net.to_float(mstype.float16)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)
        if config_ascend.label_smooth > 0:
            loss = CrossEntropyWithLabelSmooth(
                smooth_factor=config_ascend.label_smooth, num_classes=config.num_classes)
        else:
            loss = SoftmaxCrossEntropyWithLogits(
                is_grad=False, sparse=True, reduction='mean')
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 config=config_ascend,
                                 platform=args_opt.platform,
                                 repeat_num=epoch_size,
                                 batch_size=config_ascend.batch_size)
        step_size = dataset.get_dataset_size()
        if args_opt.pre_trained:
            param_dict = load_checkpoint(args_opt.pre_trained)
            load_param_into_net(net, param_dict)

        loss_scale = FixedLossScaleManager(
            config_ascend.loss_scale, drop_overflow_update=False)
        lr = Tensor(get_lr(global_step=0,
                           lr_init=0,
                           lr_end=0,
                           lr_max=config_ascend.lr,
                           warmup_epochs=config_ascend.warmup_epochs,
                           total_epochs=epoch_size,
                           steps_per_epoch=step_size))
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config_ascend.momentum,
                       config_ascend.weight_decay, config_ascend.loss_scale)

        model = Model(net, loss_fn=loss, optimizer=opt,
                      loss_scale_manager=loss_scale)

        cb = None
        if rank_id == 0:
            cb = [Monitor(lr_init=lr.asnumpy())]
            if config_ascend.save_checkpoint:
                config_ck = CheckpointConfig(save_checkpoint_steps=config_ascend.save_checkpoint_epochs * step_size,
                                             keep_checkpoint_max=config_ascend.keep_checkpoint_max)
                ckpt_cb = ModelCheckpoint(
                    prefix="mobilenetV3", directory=config_ascend.save_checkpoint_path, config=config_ck)
                cb += [ckpt_cb]
        model.train(epoch_size, dataset, callbacks=cb)
    else:
        raise Exception
