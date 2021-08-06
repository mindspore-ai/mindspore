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
"""callbacks"""


import time
from mindspore.train.callback import Callback
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank

def add_write(file_path, out_str):
    """
    add lines to the file
    """
    with open(file_path, 'a+', encoding="utf-8") as file_out:
        file_out.write(out_str + "\n")


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, terminate the training.

    Note:
        If per_print_times is 0, do NOT print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, config=None, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("per_print_times must be in and >= 0.")
        self._per_print_times = per_print_times
        self.config = config

    def step_end(self, run_context):
        """Monitor the loss in training."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num
        rank_id = 0
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL,
                             ParallelMode.DATA_PARALLEL):
            rank_id = get_rank()

        print("===loss===", rank_id, cb_params.cur_epoch_num, cur_step_in_epoch,
              loss, flush=True)
        # raise ValueError
        if self._per_print_times != 0 and cur_num % self._per_print_times == 0 and self.config is not None:
            loss_file = open(self.config.loss_file_name, "a+")
            loss_file.write("epoch: %s, step: %s, loss: %s" %
                            (cb_params.cur_epoch_num, cur_step_in_epoch, loss))
            loss_file.write("\n")
            loss_file.close()
            print("epoch: %s, step: %s, loss: %s" %
                  (cb_params.cur_epoch_num, cur_step_in_epoch, loss))


class EvalCallBack(Callback):
    """
    Monitor the loss in evaluating.

    If the loss is NAN or INF, terminate evaluating.

    Note:
        If per_print_times is 0, do NOT print loss.

    Args:
        print_per_step (int): Print loss every times. Default: 1.
    """
    def __init__(self, model, eval_dataset, auc_metric, config, print_per_step=1, host_device_mix=False):
        super(EvalCallBack, self).__init__()
        if not isinstance(print_per_step, int) or print_per_step < 0:
            raise ValueError("print_per_step must be int and >= 0.")
        self.print_per_step = print_per_step
        self.model = model
        self.eval_dataset = eval_dataset
        self.aucMetric = auc_metric
        self.aucMetric.clear()
        self.eval_file_name = config.eval_file_name
        self.eval_values = []
        self.host_device_mix = host_device_mix

    def epoch_end(self, run_context):
        """
        epoch end
        """
        self.aucMetric.clear()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            context.set_auto_parallel_context(strategy_ckpt_save_file="",
                                              strategy_ckpt_load_file="./strategy_train.ckpt")
        rank_id = 0
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL,
                             ParallelMode.DATA_PARALLEL):
            rank_id = get_rank()
        start_time = time.time()
        out = self.model.eval(self.eval_dataset, dataset_sink_mode=(not self.host_device_mix))
        end_time = time.time()
        eval_time = int(end_time - start_time)

        time_str = time.strftime("%Y-%m-%d %H:%M%S", time.localtime())
        out_str = "{} == Rank: {} == EvalCallBack model.eval(): {}; eval_time: {}s".\
            format(time_str, rank_id, out.values(), eval_time)
        print(out_str)
        self.eval_values = out.values()
        add_write(self.eval_file_name, out_str)
