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
"""TimeMonitor Callback class."""

import time

from ._callback import Callback


class TimeMonitor(Callback):
    """Time Monitor."""

    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size
        print("epoch time: {0}, per step time: {1}".format(epoch_mseconds, per_step_mseconds), flush=True)
