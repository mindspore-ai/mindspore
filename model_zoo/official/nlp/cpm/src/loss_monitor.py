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
"""Loss monitor."""
import time
import math

from mindspore import log as logger
from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    """
    time_stamp_init = False
    time_stamp_first = 0

    def __init__(self, per_print_times=-1):
        super(LossCallBack, self).__init__()
        self._per_print_times = per_print_times

        if not self.time_stamp_init:
            self.time_stamp_first = self._get_ms_timestamp()
            self.time_stamp_init = True

    def step_end(self, run_context):
        """Print loss after each step."""
        cb_params = run_context.original_args()
        file_name = "./loss.log"
        with open(file_name, "a+") as f:
            time_stamp_current = self._get_ms_timestamp()
            if self._per_print_times > 0:
                _, epoch_num = math.modf(cb_params.cur_step_num / self._per_print_times)
                f.write("time: {},epoch: {},step: {},outputs:[loss: {},overflow: {}, loss scale value: {} ].\n".format(
                    time_stamp_current - self.time_stamp_first,
                    int(epoch_num),
                    cb_params.cur_step_num,
                    str(cb_params.net_outputs[0].asnumpy()),
                    str(cb_params.net_outputs[1].asnumpy()),
                    str(cb_params.net_outputs[2].asnumpy())
                ))
            else:
                f.write("time: {},epoch: {},step: {},outputs: [loss: {},overflow: {},loss scale value: {} ].\n".format(
                    time_stamp_current - self.time_stamp_first,
                    cb_params.cur_epoch_num,
                    cb_params.cur_step_num,
                    str(cb_params.net_outputs[0].asnumpy()),
                    str(cb_params.net_outputs[1].asnumpy()),
                    str(cb_params.net_outputs[2].asnumpy())
                ))

    @staticmethod
    def _get_ms_timestamp():
        """Get timestamp."""
        t = time.time()
        return int(round(t * 1000))


class TimeCallBack(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): Dataset size. Default: None.
    """

    def __init__(self, data_size=None):
        super(TimeCallBack, self).__init__()
        self.data_size = data_size

    def step_begin(self, run_context):
        """Step begin."""
        self.epoch_time = time.time()

    def step_end(self, run_context):
        """Step end."""
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            logger.error("data_size must be positive int.")
            return

        print("epoch time: {:5.3f} ms".format(epoch_seconds), flush=True)

    def epoch_begin(self, run_context):
        """Epoch begin."""
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Epoch end."""
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            logger.error("data_size must be positive int.")
            return

        print("epoch time: {:5.3f} ms".format(epoch_seconds), flush=True)
