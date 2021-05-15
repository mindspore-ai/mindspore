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
"""callbacks"""
import time
from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
        Monitor the loss in training.

        If the loss is NAN or INF terminating training.

        Note:
            If per_print_times is 0 do not print loss.

        Args:
            per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, data_size, per_print_times=1, logger=None):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.logger = logger
        self._per_print_times = per_print_times
        self._loss = 0
        self.data_size = data_size
        self.step_cnt = 0
        self.loss_sum = 0

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """record loss and time"""
        epoch_seconds = time.time() - self.epoch_time
        self._per_step_seconds = epoch_seconds / self.data_size
        self._loss = self.loss_sum / self.step_cnt
        self.step_cnt = 0
        self.loss_sum = 0

        cb_params = run_context.original_args()
        epoch_idx = (cb_params.cur_step_num - 1) // cb_params.batch_num + 1
        print("the {} epoch's resnet result: "
              "training loss {},"
              "training per step cost {:.2f} s, total_cost {:.2f} s".format(
                  epoch_idx, self._loss, self._per_step_seconds, self._per_step_seconds * cb_params.batch_num))

        self.logger.info("the {} epoch's resnet result: "
                         "training loss {},"
                         "training per step cost {:.2f} s, total_cost {:.2f} s".format(
                             epoch_idx, self._loss, self._per_step_seconds,
                             self._per_step_seconds * cb_params.batch_num))

    def get_loss(self):
        return self._loss

    def get_per_step_time(self):
        return self._per_step_seconds

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if not isinstance(cb_params.net_outputs, list):
            loss = cb_params.net_outputs.asnumpy()
        else:
            loss = cb_params.net_outputs[0].asnumpy()

        self.loss_sum += loss
        self.step_cnt += 1
