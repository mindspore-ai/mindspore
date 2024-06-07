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
"""Training checker"""
import os
import sys
import time

import mindspore as ms
from mindspore import Callback
from mindspore.communication import get_rank, get_group_size

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from mindformers.core.callback.callback import _get_loss_output


class TrainingChecker(Callback):
    """
    Callback function for precision and performance checking. Raise an AssertionError once the difference
    between a step's loss and the corresponding expected value is greater than the error value or the
    difference ratio between average step time and expected value is greater than the error ratio.

    Args:
        loss_list_std (list[float]):
            A list of expected loss values.
        avg_step_time_std (float):
            Expected average step time value (in millisecond). Defaults to None.
        loss_error (float, optional):
            Allowable loss error between true and expected values. Defaults to 1e-3.
        time_error_ratio (float, optional):
            Allowable time error ratio between true and expected values. Defaults to 0.1.
        skip_step_num (int, optional):
            Skip a certain number of steps before counting the time. Defaults to 2.
        skip_time_num (int, optional):
            Remove the largest values in collected step time list. Defaults to 5.
        micro_batch_num (int, optional):
            The number of micro-batch in a pipeline stage. Defaults to 1.
        micro_batch_interleave_num (int, optional):
            Multi-copy parallel configuration. Defaults to 1.
        gradient_accumulation_steps (int, optional):
            The number of gradient accumulation steps. Defaults to 1.
        loss_mode (bool, optional):
            The mode of checking loss, 'abs' and 'relative' are supported. Defaults to 'abs'.
        experiment_mode (bool, optional):
            Enables or disables the developer debugging mode. Defaults to False.
            If set True, will not check values of loss and time.

    Raises:
        AssertionError
    """

    def __init__(self, loss_list_std: list, avg_step_time_std: float = None,
                 loss_error: float = 1e-3, time_error_ratio: float = 0.1,
                 skip_step_num: int = 2, skip_time_num: int = 5, micro_batch_num: int = 1,
                 micro_batch_interleave_num: int = 1, gradient_accumulation_steps: int = 1,
                 loss_mode: str = 'abs', experiment_mode: bool = False):
        super(TrainingChecker, self).__init__()
        self.loss_list_std = loss_list_std
        self.avg_step_time_std = avg_step_time_std
        self.loss_error = loss_error
        self.time_error_ratio = time_error_ratio
        self.step_time = time.time()
        self.total_time = []
        self.skip_step_num = skip_step_num
        self.skip_time_num = skip_time_num

        # init pipeline parallel status
        self.pipeline_parallel = False
        self.is_last_stage = True
        self.micro_size = micro_batch_num

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_batch_interleave_num = micro_batch_interleave_num

        self.loss_mode = loss_mode
        self.experiment_mode = experiment_mode
        self.loss_recoder = []
        self.time_recoder = []

    def on_train_begin(self, run_context):
        """Called once before the network training."""
        self.begin(run_context)

        # Check pipeline parallel training status.
        pipeline_stages = ms.get_auto_parallel_context('pipeline_stages')
        self.pipeline_parallel = pipeline_stages > 1

        if self.pipeline_parallel:
            rank_id = get_rank()
            device_num = get_group_size()

            per_stage_device_num = device_num // pipeline_stages
            stage_id = rank_id // per_stage_device_num
            self.is_last_stage = (stage_id == pipeline_stages - 1)

    def on_train_step_begin(self, run_context):
        """Called on each training step begin."""
        _ = run_context
        self.step_time = time.time()

    def on_train_step_end(self, run_context):
        """Called on each training step end."""
        cb_params = run_context.original_args()
        net_outputs = cb_params.net_outputs
        loss = _get_loss_output(net_outputs)[0]
        cur_step_num = cb_params.cur_step_num
        cur_step_time = (time.time() - self.step_time) * 1000

        if cur_step_num > self.skip_step_num:
            self.total_time.append(cur_step_time)

        if self.pipeline_parallel:
            loss = loss / self.micro_size
        if self.micro_batch_interleave_num > 1:
            loss = loss / self.micro_batch_interleave_num
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        self.loss_recoder.append(loss)

        # when enable pp, loss will be only available on the last card
        if (not self.pipeline_parallel or self.is_last_stage) and not self.experiment_mode:
            real_loss = self.loss_list_std[cur_step_num - 1]
            if self.loss_mode == 'abs':
                loss_diff = abs(loss - real_loss)
            elif self.loss_mode == 'relative':
                loss_diff = abs(loss - real_loss / real_loss)
            else:
                raise ValueError(f"support 'abs' and 'relative' loss checking mode, but got {self.loss_mode}.")

            print(f"loss check mode: {self.loss_mode}.")
            assert loss_diff < self.loss_error, \
                f"The error between loss: {loss} and loss_list_std: {real_loss} is larger than {self.loss_error}"

    def on_train_end(self, run_context):
        _ = run_context
        self.total_time.sort()
        self.total_time = self.total_time[:-self.skip_time_num]
        avg_step_time = sum(self.total_time) / len(self.total_time)
        self.time_recoder.append(avg_step_time)

        if self.avg_step_time_std is not None and not self.experiment_mode:
            assert (avg_step_time - self.avg_step_time_std) / self.avg_step_time_std < self.time_error_ratio, \
                f"The error ratio between avg_step_time: {avg_step_time} and " \
                f"avg_step_time_std: {self.avg_step_time_std} is larger than {self.time_error_ratio}"

    def get_experiment_results(self):
        print("\nexperiment loss: ")
        for i in range(0, len(self.loss_recoder), 5):
            loss = [f"{item:.6f}" for item in self.loss_recoder[i:i + 5]]
            print(', '.join(loss) + ',')
        print(f"\nexperiment time: {int(self.time_recoder[0])}")
