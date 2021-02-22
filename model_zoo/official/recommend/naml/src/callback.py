# Copyright 2021 Huawei Technologies Co., Ltd
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
"""NAML Callback"""
import os
import time
import numpy as np
from mindspore import Tensor, save_checkpoint
from mindspore.train.callback import Callback

class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(args)
    """

    def __init__(self, args):
        super(Monitor, self).__init__()
        self.cur_step = 1
        self.cur_epoch = 1
        self.epochs = args.epochs
        self.sink_size = args.print_times
        self.sink_mode = args.sink_mode
        self.dataset_size = args.dataset_size
        self.save_checkpoint_path = args.save_checkpoint_path
        self.save_checkpoint = args.save_checkpoint
        self.losses = []
        if args.sink_mode:
            self.epoch_steps = self.sink_size
        else:
            self.epoch_steps = args.dataset_size
        if self.save_checkpoint and not os.path.isdir(self.save_checkpoint_path):
            os.makedirs(self.save_checkpoint_path)

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        epoch_end_f = True
        if self.sink_mode:
            self.cur_step += self.epoch_steps
            epoch_end_f = False
            if self.cur_step >= self.dataset_size:
                epoch_end_f = True
                self.cur_step = self.cur_step % self.dataset_size
            cb_params = run_context.original_args()
            epoch_mseconds = (time.time() - self.epoch_time) * 1000
            per_step_mseconds = epoch_mseconds / cb_params.batch_num
            step_loss = cb_params.net_outputs
            if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                step_loss = step_loss[0]
            if isinstance(step_loss, Tensor):
                step_loss = np.mean(step_loss.asnumpy())
            self.losses.append(step_loss)
        if epoch_end_f:
            print("epoch: {:3d}/{:3d}, avg loss:{:5.3f}".format(
                self.cur_epoch, self.epochs, np.mean(self.losses)), flush=True)
            self.losses = []
            self.cur_epoch += 1
        if self.sink_mode:
            print("epoch: {:3d}/{:3d}, step:{:5d}/{:5d}, loss:{:5.3f}, per step time:{:5.3f} ms".format(
                self.cur_epoch, self.epochs, self.cur_step, self.dataset_size, step_loss, per_step_mseconds),
                  flush=True)
        if epoch_end_f and self.save_checkpoint:
            save_checkpoint(cb_params.train_network,
                            os.path.join(self.save_checkpoint_path, f"naml_{self.cur_epoch-1}.ckpt"))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        """Callback when step end."""
        if not self.sink_mode:
            cb_params = run_context.original_args()
            self.cur_step += 1
            self.cur_step = self.cur_step % self.dataset_size
            step_loss = cb_params.net_outputs
            if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                step_loss = step_loss[0]
            if isinstance(step_loss, Tensor):
                step_loss = np.mean(step_loss.asnumpy())
            step_mseconds = (time.time() - self.step_time) * 1000
            print("epoch: {:3d}/{:3d}, step:{:5d}/{:5d}, loss:{:5.3f}, per step time:{:5.3f} ms".format(
                self.cur_epoch, self.epochs, self.cur_step, self.dataset_size, step_loss, step_mseconds), flush=True)

    def end(self, run_context):
        cb_params = run_context.original_args()
        print("epoch: {:3d}/{:3d}, avg loss:{:5.3f}".format(
            self.epochs, self.epochs, np.mean(self.losses)), flush=True)
        if self.save_checkpoint:
            save_checkpoint(cb_params.train_network, os.path.join(self.save_checkpoint_path, f"naml_last.ckpt"))
