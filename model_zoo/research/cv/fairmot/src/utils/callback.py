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
"""
custom callback
"""
import time
from mindspore.train.callback import Callback


class LossCallback(Callback):
    """StopAtTime"""

    def __init__(self, bach_size):
        """init"""
        super(LossCallback, self).__init__()
        self.bach_size = bach_size
        self.time_start = time.time()

    def begin(self, run_context):
        """train begin"""
        cb_params = run_context.original_args()
        batch_num = cb_params.batch_num
        epoch_num = cb_params.epoch_num
        device_number = cb_params.device_number

        print("Starting Training : device_number={},per_step_size={},batch_size={}, epoch={}".format(device_number,
                                                                                                     batch_num,
                                                                                                     self.bach_size,
                                                                                                     epoch_num))

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        cur_step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs[0].asnumpy()
        batch_num = cb_params.batch_num
        if cur_step_num > batch_num:
            cur_step_num = cur_step_num % batch_num + 1
        epoch_num = cb_params.epoch_num
        print("epoch: {}/{}, step: {}/{}, loss is {}".format(cur_epoch_num, epoch_num, cur_step_num, batch_num, loss))

    def end(self, run_context):
        """train end"""
        cb_params = run_context.original_args()
        device_number = cb_params.device_number
        time_end = time.time()
        seconds = time_end - self.time_start
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        print(device_number, "device totally cost:%02d:%02d:%02d" % (hour, minutes, seconds))
