# Copyright 2023 Huawei Technologies Co., Ltd
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
mindspore.multiprocessing is a wrapper around the native `multiprocessing` module.
Some methods are overrode to support fork-based multiprocess.
"""
import types
import signal
import multiprocessing as mp
from multiprocessing import *
from multiprocessing import pool as mp_pool
from mindspore._c_expression import fork_utils

__all__ = []
__all__ += mp.__all__


class Process(mp.Process): # pylint: disable=function-redefined
    """
    Trigger fork callbacks by overriding native multiprocessing methods.
    """
    _child_at_fork_func = None

    def run(self):
        """
        Trigger child_at_fork callback function after fork by overriding
        multiprocessing.run method.
        """
        if mp.get_start_method() == "fork":
            fork_utils.prctl_set_pdeathsig(signal.SIGINT)
            fork_utils.child_at_fork()
            if Process._child_at_fork_func and callable(Process._child_at_fork_func):
                Process._child_at_fork_func() # pylint: disable=not-callable
        super().run()

    def start(self):
        """
        Trigger prepare_before_fork and parent_at_fork callback functions
        by overriding multiprocessing.start method.
        """
        if mp.get_start_method() == "fork":
            fork_utils.prepare_before_fork()
            super().start()
            fork_utils.parent_at_fork()
        else:
            super().start()

_MsProcess = Process


class Pool(mp_pool.Pool):  # pylint: disable=function-redefined, abstract-method
    """
    Trigger fork callbacks by overriding native multiprocessing methods.
    """
    def Process(self, *args, **kwds):
        if self._ctx.get_start_method() == "fork":
            # Process() becomes a staticmethod function of Pool with first argument 'ctx' in python 3.8.0 and later
            if isinstance(super().Process, types.FunctionType):
                args = args[1:]
            return _MsProcess(*args, **kwds)
        return super().Process(*args, **kwds)
