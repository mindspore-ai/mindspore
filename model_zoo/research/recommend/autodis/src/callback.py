# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Defined callback for DeepFM.
"""
import time
from mindspore.train.callback import Callback


def add_write(file_path, out_str):
    with open(file_path, 'a+', encoding='utf-8') as file_out:
        file_out.write(out_str + '\n')


class EvalCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note
        If per_print_times is 0 do not print loss.
    """
    def __init__(self, model, eval_dataset, auc_metric, eval_file_path):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.aucMetric = auc_metric
        self.aucMetric.clear()
        self.eval_file_path = eval_file_path

    def epoch_end(self, run_context):
        start_time = time.time()
        out = self.model.eval(self.eval_dataset)
        eval_time = int(time.time() - start_time)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        out_str = "{} EvalCallBack metric{}; eval_time{}s".format(
            time_str, out.values(), eval_time)
        print(out_str)
        add_write(self.eval_file_path, out_str)


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note
        If per_print_times is 0 do not print loss.
    Args
        loss_file_path (str) The file absolute path, to save as loss_file;
        per_print_times (int) Print loss every times. Default 1.
    """
    def __init__(self, loss_file_path, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.loss_file_path = loss_file_path
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        """Monitor the loss in training."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num
        if self._per_print_times != 0 and cur_num % self._per_print_times == 0:
            with open(self.loss_file_path, "a+") as loss_file:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss_file.write("{} epoch: {} step: {}, loss is {}\n".format(
                    time_str, cb_params.cur_epoch_num, cur_step_in_epoch, loss))
            print("epoch: {} step: {}, loss is {}\n".format(
                cb_params.cur_epoch_num, cur_step_in_epoch, loss))


class TimeMonitor(Callback):
    """
    Time monitor for calculating cost of each epoch.
    Args
        data_size (int) step size of an epoch.
    """
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size

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
