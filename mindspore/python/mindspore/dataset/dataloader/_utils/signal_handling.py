# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Signal handling module."""

import signal
import threading

import mindspore._c_dataengine as cde
from mindspore import log as logger

_SIGCHLD_HANDLER_SET = False


def set_sigchld_handler():
    """
    Set signal handler for SIGCHLD.
    """

    if threading.current_thread() != threading.main_thread():
        logger.warning("Cannot set signal handler in child threads.")
        return
    global _SIGCHLD_HANDLER_SET
    if _SIGCHLD_HANDLER_SET:
        return
    last_sigchld_handler = signal.getsignal(signal.SIGCHLD)

    def sigchld_handler(signum, frame):
        worker_status = cde.check_if_worker_exit()
        if worker_status != "":
            raise RuntimeError(worker_status)
        if callable(last_sigchld_handler):
            last_sigchld_handler(signum, frame)

    signal.signal(signal.SIGCHLD, sigchld_handler)
    _SIGCHLD_HANDLER_SET = True
