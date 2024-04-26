# Copyright 2024 Huawei Technologies Co., Ltd
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
import os
import time
from copy import deepcopy

import mindspore as ms
from mindspore import Callback
import numpy as np
from mindformers.llama_utils import get_real_group_size

__all__ = ['LlamaCallback']

_cur_dir = os.getcwd()
SAVE_DIR = _cur_dir


class LlamaCallback(Callback):
    def __init__(self,
                 learning_rate: float = 0.001,
                 per_print_times: int = 1,
                 micro_batch_num: int = 1,
                 micro_batch_interleave_num: int = 1,
                 origin_epochs: int = None,
                 dataset_size: int = None,
                 initial_epoch: int = 0,
                 initial_step: int = 0,
                 global_batch_size: int = 0,
                 gradient_accumulation_steps: int = 1,
                 stop_steps: int = 20):

        super(LlamaCallback, self).__init__()
        self.per_print_times = per_print_times
        self.learning_rate = deepcopy(learning_rate)
        self.last_print_time = 0
        self.mirco_size = micro_batch_num
        self.print_warning_flag = True
        self.loss_list = []
        self.step_time = time.time()
        self.epoch_time = time.time()
        self.run_context = None
        self.steps_per_epoch = dataset_size
        self.micro_batch_interleave_num = micro_batch_interleave_num
        self.origin_epochs = origin_epochs
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.global_batch_size = global_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_num = get_real_group_size()
        self.stop_steps = stop_steps
        self.overflow_list = []
        self.throughput_list = []
        self.loss_collector = []

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.loss_list = []
        self.epoch_time = time.time()
        self.run_context = run_context

    def epoch_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """

    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.run_context = run_context

    def step_end(self, run_context):
        """
        Print training info at the end of step.
        Args:
            run_context (RunContext): Context of the process running.
        """
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        auto_parallel = parallel_mode in ['semi_auto_parallel', 'auto_parallel']
        if auto_parallel:
            ms.context.set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
        cb_params = run_context.original_args()
        step_seconds = (time.time() - self.step_time) * 1000
        net_outputs = cb_params.net_outputs
        loss, overflow, _, learning_rate = _get_loss_output(net_outputs)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        loss = self._fix_loss_for_parallel(loss)
        self.loss_list.append(loss)
        self.loss_collector.append(loss)
        self.overflow_list.append(overflow)

        if cb_params.dataset_sink_mode:
            per_step_seconds = step_seconds / cb_params.batch_num
            steps_per_epoch = self.steps_per_epoch
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1
        else:
            per_step_seconds = step_seconds
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % cb_params.batch_num + 1

        # compute throughput
        throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000)
        self.throughput_list.append(throughput)

        if auto_parallel:
            ms.context.set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)

        if cur_step_num >= self.stop_steps:
            run_context.request_stop()

    def _fix_loss_for_parallel(self, loss):
        """Fix loss value in pipeline or double parallel mode."""
        pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
        if pipeline_stages > 1:
            loss = loss / self.mirco_size
        if self.micro_batch_interleave_num > 1:
            loss = loss / self.micro_batch_interleave_num
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        return loss


def _get_loss_output(output):
    """Get output of task for MFLossMonitor."""
    overflow = False
    scaling_sens = False
    loss = output
    learning_rate = None
    if isinstance(output, (tuple, list)):
        if len(output) == 3:
            loss, overflow, scaling_sens = output
            if isinstance(scaling_sens, ms.Tensor):
                scaling_sens = scaling_sens.asnumpy()
        elif len(output) == 4:
            loss, overflow, scaling_sens, learning_rate = output
            if isinstance(scaling_sens, ms.Tensor):
                scaling_sens = scaling_sens.asnumpy()
        else:
            if isinstance(output[0], ms.Tensor) and isinstance(output[0].asnumpy(), np.ndarray):
                loss = output[0]

    if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())

    # Boundary check.
    if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
        invalid_loss_info = "NaN" if np.isnan(loss) else "Inf"
        raise ValueError(f"The current value of loss is {invalid_loss_info}, terminate training.")

    return loss, overflow, scaling_sens, learning_rate
