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
Callbacks
"""


import time
import math
from mindspore.train.callback import Callback
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, dataset_size=-1, local_rank=0, has_trained_epoch=0, has_trained_step=0, micro_size=1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step
        self.micro_size = micro_size
        print("load has trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0 and self.local_rank % 8 == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            loss_value = cb_params.net_outputs[0].asnumpy() / self.micro_size
            print("time: {} local_rank: {}, epoch: {}, step: {}, output is {}, overflow is {}, scale is {}".
                  format(date, int(self.local_rank), int(epoch_num) + int(self.has_trained_epoch),
                         cb_params.cur_step_num + int(self.has_trained_step), loss_value,
                         cb_params.net_outputs[1].asnumpy(), cb_params.net_outputs[2].asnumpy()))

class EvalCallBack(Callback):
    """
    Monitor the ppl loss in evaluating.
    Note:
        If per_print_times is 0, do NOT print loss.

    Args:
        print_per_step (int): Print loss every times. Default: 1.
    """
    def __init__(self, model, eval_dataset, ppl_metric, print_per_step=100, has_trained_step=0):
        super(EvalCallBack, self).__init__()
        if not isinstance(print_per_step, int) or print_per_step < 0:
            raise ValueError("print_per_step must be int and >= 0.")
        self.print_per_step = print_per_step
        self.model = model
        self.eval_dataset = eval_dataset
        self.pplMetric = ppl_metric
        self.has_trained_step = has_trained_step
        self.pplMetric.clear()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.strategy_ckpt_save_file = context.get_auto_parallel_context("strategy_ckpt_save_file")
        self.strategy_ckpt_load_file = context.get_auto_parallel_context("strategy_ckpt_load_file")

    def step_end(self, run_context):
        """
        step end
        """
        cb_params = run_context.original_args()
        current_step = cb_params.cur_step_num + self.has_trained_step
        if current_step % self.print_per_step != 0:
            return
        self.pplMetric.clear()
        if self.parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            context.set_auto_parallel_context(strategy_ckpt_save_file="",
                                              strategy_ckpt_load_file=self.strategy_ckpt_save_file)
        rank_id = 0
        if self.parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL,
                                  ParallelMode.AUTO_PARALLEL, ParallelMode.DATA_PARALLEL):
            rank_id = get_rank()
        start_time = time.time()
        out = self.model.eval(self.eval_dataset, dataset_sink_mode=True)
        end_time = time.time()
        eval_time = int(end_time - start_time)

        time_str = time.strftime("%Y-%m-%d %H:%M%S", time.localtime())
        out_str = "{} == Rank: {} == EvalCallBack model.eval(): {}; eval_time: {}s". \
            format(time_str, rank_id, out.values(), eval_time)
        print(out_str)
        context.set_auto_parallel_context(strategy_ckpt_save_file=self.strategy_ckpt_save_file,
                                          strategy_ckpt_load_file=self.strategy_ckpt_load_file)
