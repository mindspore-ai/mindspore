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
# ===========================================================================
""" Callback"""
import time
import numpy as np
from mindspore import Tensor
from mindspore.train.callback import Callback


def add_write(file_path, out_str):
    """ Write info"""
    with open(file_path, 'a+', encoding='utf-8') as file_out:
        file_out.write(out_str + '\n')


class AUCCallBack(Callback):
    """ AUCCallBack"""
    def __init__(self, model, eval_dataset, eval_file_path):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_file_path = eval_file_path

    def epoch_end(self, run_context):
        start_time = time.time()
        out = self.model.eval(self.eval_dataset)
        eval_time = int(time.time() - start_time)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        out_str = "{} AUC:{}; eval_time{}s".format(
            time_str, out.values(), eval_time)
        print(out_str)
        add_write(self.eval_file_path, out_str)


class LossCallback(Callback):
    """ LossCallBack"""
    def __init__(self, loss_file_path, per_print_times=1):
        super(LossCallback, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.loss_file_path = loss_file_path

    def step_end(self, run_context):
        """ run after step_end """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            with open(self.loss_file_path, "a+") as loss_file:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss_file.write("{} epoch: {} step: {}, loss is {}\n".format(
                    time_str, cb_params.cur_epoch_num, cur_step_in_epoch, loss))
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)


class TimeMonitor(Callback):
    """ TimeMonitor"""
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = None
        self.step_time = None

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size
        print("epoch time: {0}, per step time: {1}".format(epoch_mseconds, per_step_mseconds), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_mseconds = (time.time() - self.step_time) * 1000
        print(f"step time {step_mseconds}", flush=True)
