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
# ==============================================================================
"""
This module defines the class for minddata pipeline debugger.
class DebugHook is not exposed to users as an external API.
"""

from abc import ABC, abstractmethod
from mindspore import log as logger


class DebugHook(ABC):
    """
    The base class for Dataset Pipeline Python Debugger hook. All user defined hook behaviors
    must inherit this base class.

    To debug the input and output data of map operation in dataset pipeline, users can add
    breakpoint to or single stepping in this class. They can also see the type and shape of
    the data from the log being printed.

    Args:
        prev_op_name (str, optional): name of the operation before current debugging point.
    """
    def __init__(self, prev_op_name=None):
        self.prev_op_name = prev_op_name

    def __call__(self, *args):
        # log op name
        if self.prev_op_name:
            log_message = "Debugging the output of the operation [{}].".format(self.prev_op_name)
        else:
            log_message = "Debugging the input of the first operation."
        logger.info(log_message)

        ######################## NOTE ########################
        # Add a breakpoint to the following line to inspect
        # input and output of each transform.
        ######################################################
        self.compute(args)
        return args

    @abstractmethod
    def compute(self, *args):
        """
        Defines the debug behaviour to be performed. This method must be overridden by all subclasses.
        """
        raise RuntimeError("compute() is not overridden in subclass of class DebugHook.")

    def set_previous_op_name(self, prev_op_name):
        self.prev_op_name = prev_op_name
