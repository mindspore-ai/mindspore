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
"""LossMonitor Callback class."""

import time
import numpy as np
from mindspore.common.tensor import Tensor

from ._callback import Callback


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
        lr_init (numpy array): train learning rate. Default: None.

    Raises:
        ValueError: If print_step is not int or less than zero.

    Examples:
        >>> LossMonitor(100, lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, per_print_times=1, lr_init=None):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.lr_init = lr_init

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("Epoch time: {:5.3f}, per step time: {:5.3f}, "
              "avg loss: {:5.3f}".format(epoch_mseconds,
                                         per_step_mseconds,
                                         np.mean(self.losses)))
        print("*" * 60)

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
        cur_step_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num) + 1

        if isinstance(step_loss, float) and (np.isnan(step_loss) or np.isinf(step_loss)):
            raise ValueError("Epoch: [{:3d}/{:3d}], step: [{:5d}/{:5d}]. "
                             "Invalid loss, terminating training.".format(
                                 cb_params.cur_epoch_num - 1, cb_params.epoch_num,
                                 cur_step_in_epoch, cb_params.batch_num))

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("Epoch: [{:3d}/{:3d}], step: [{:5d}/{:5d}], "
                  "loss: [{:5.4f}/{:5.4f}], time: [{:5.4f}]".format(
                      cb_params.cur_epoch_num, cb_params.epoch_num,
                      cur_step_in_epoch, int(cb_params.batch_num),
                      step_loss, np.mean(self.losses),
                      step_mseconds), flush=True)
