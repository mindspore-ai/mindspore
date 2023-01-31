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
class DebugWrapper is not exposed to users as an external API.
"""

import collections
import numpy as np
from PIL import Image
from mindspore import log as logger


class DebugWrapper:
    """
    A class for Minddata Python Debugger.

    To debug the input and output data of map operation in dataset pipeline, users can add
    breakpoint to or single stepping in this class. They can also see the type and shape of
    the data from the log being printed.

    Args:
        prev_op_name (str, optional): name of the operation before current debugging point.
    """
    def __init__(self, prev_op_name=None):
        self.prev_op_name = prev_op_name

    def __call__(self, x):
        # log op name
        if self.prev_op_name:
            log_message = "Debugging the output of the operation [{}].".format(self.prev_op_name)
        else:
            log_message = "Debugging the input of the first operation."

        # log type
        log_message += " The type is [{}].".format(type(x))

        # log shape/size
        if isinstance(x, np.ndarray):
            log_message += " The shape is [{}].".format(x.shape)
        elif isinstance(x, Image.Image):
            log_message += " The shape is [{}].".format(x.size)
        elif isinstance(x, collections.abc.Sized):
            log_message += " The size is [{}].".format(len(x))

        ######################## NOTE ########################
        # Add a breakpoint to the following line to inspect
        # input and output of each transform.
        ######################################################
        logger.info(log_message)
        return x
