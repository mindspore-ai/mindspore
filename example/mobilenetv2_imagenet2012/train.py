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
from dataset import create_dataset
from lr_generator import get_lr
from config import config
from mindspore import context
from mindspore import Tensor
from mindspore.model_zoo.mobilenet import mobilenet_v2
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindspore.train.model import Model, ParallelMode

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset.engine as de
from mindspore.communication.management import init

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
rank_id = int(os.getenv('RANK_ID'))
rank_size = int(os.getenv('RANK_SIZE'))
run_distribute = rank_size > 1

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id, save_graphs=False)
context.set_context(enable_task_sink=True)
context.set_context(enable_loop_sink=True)
context.set_context(enable_mem_reuse=True)


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None.

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
                                                                                      np.mean(self.losses)
                                                                                      ), flush=True)

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
            cb_params.cur_epoch_num - 1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]), flush=True)


if __name__ == '__main__':
    if run_distribute:
        context.set_context(enable_hccl=True)
        context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, mirror_mean=True)
        auto_parallel_context().set_all_reduce_fusion_split_indices([140])
        init()
    else:
        context.set_context(enable_hccl=False)

    epoch_size = config.epoch_size
    net = mobilenet_v2(num_classes=config.num_classes)
    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')

    print("train args: ", args_opt, "\ncfg: ", config,
          "\nparallel args: rank_id {}, device_id {}, rank_size {}".format(rank_id, device_id, rank_size))

    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True,
                             repeat_num=epoch_size, batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(global_step=0, lr_init=0, lr_end=0, lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size))
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)

    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale)

    cb = None
    if rank_id == 0:
        cb = [Monitor(lr_init=lr.asnumpy())]
        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="mobilenet", directory=config.save_checkpoint_path, config=config_ck)
            cb += [ckpt_cb]
    model.train(epoch_size, dataset, callbacks=cb)
