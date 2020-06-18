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
"""SummaryStep Callback class."""

from ._callback import Callback


class SummaryStep(Callback):
    """
    The summary callback class.

    Args:
        summary (Object): Summary recode object.
        flush_step (int): Number of interval steps to execute. Default: 10.
    """

    def __init__(self, summary, flush_step=10):
        super(SummaryStep, self).__init__()
        if not isinstance(flush_step, int) or isinstance(flush_step, bool) or flush_step <= 0:
            raise ValueError("`flush_step` should be int and greater than 0")
        self._summary = summary
        self._flush_step = flush_step

    def __enter__(self):
        self._summary.__enter__()
        return self

    def __exit__(self, *err):
        return self._summary.__exit__(*err)

    def step_end(self, run_context):
        """
        Save summary.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self._flush_step == 0:
            self._summary.record(cb_params.cur_step_num, cb_params.train_network)

    @property
    def summary_file_name(self):
        return self._summary.full_file_name
