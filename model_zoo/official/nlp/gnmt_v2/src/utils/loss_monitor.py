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
"""Loss monitor."""
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
    time_stamp_init = False
    time_stamp_first = 0

    def __init__(self, config, per_print_times: int = 1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.config = config
        self._per_print_times = per_print_times

        if not self.time_stamp_init:
            self.time_stamp_first = self._get_ms_timestamp()
            self.time_stamp_init = True

    def step_end(self, run_context):
        """step end."""
        cb_params = run_context.original_args()
        file_name = "./loss.log"
        with open(file_name, "a+") as f:
            time_stamp_current = self._get_ms_timestamp()
            f.write("time: {}, epoch: {}, step: {}, outputs: [loss: {}, overflow: {}, loss scale value: {} ].\n".format(
                time_stamp_current - self.time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())
            ))

    @staticmethod
    def _get_ms_timestamp():
        t = time.time()
        return int(round(t * 1000))
