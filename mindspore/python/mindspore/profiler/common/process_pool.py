# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""Analyze profiling data asynchronously by asynchronous process"""
import atexit
from typing import List
from multiprocessing import Process

from mindspore.profiler.common.singleton import Singleton


@Singleton
class MultiProcessPool:
    """A ProcessPool to run task asynchronously"""

    def __init__(self) -> None:
        self.porcess_list: List[Process] = []
        atexit.register(self.wait_all_job_finished)

    def add_async_job(self, func):
        """Add job and run in subprocess"""
        process = Process(target=func)
        process.start()
        self.porcess_list.append(process)

    def wait_all_job_finished(self):
        """Wait all subprocess finished"""
        for process in self.porcess_list:
            process.join()
        self.porcess_list = []
