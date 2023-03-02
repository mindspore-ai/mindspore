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
This module defines the subclass of DebugHook for minddata pipeline debugger.
All these class are pre-defined for users for basic debugging purpose.
"""

import collections
import numpy as np
from PIL import Image
from mindspore import log as logger
from mindspore.dataset.debug.debug_hook import DebugHook


class PrintMetaDataHook(DebugHook):
    """
    Debug hook used for MindData debug mode to print type and shape of data.
    """
    def __init__(self):
        super().__init__()

    def compute(self, *args):
        for col_idx, col in enumerate(*args):
            log_message = "Column {}. ".format(col_idx)
            # log type
            log_message += "The type is [{}].".format(type(col))

            # log shape/size
            if isinstance(col, np.ndarray):
                log_message += " The shape is [{}].".format(col.shape)
            elif isinstance(col, Image.Image):
                log_message += " The shape is [{}].".format(col.size)
            elif isinstance(col, collections.abc.Sized):
                log_message += " The size is [{}].".format(len(col))
            logger.info(log_message)
        return args


class PrintDataHook(DebugHook):
    """
    Debug hook used for MindData debug mode to print data.
    """
    def __init__(self):
        super().__init__()

    def compute(self, *args):
        for col_idx, col in enumerate(*args):
            log_message = "Column {}. ".format(col_idx)
            if isinstance(col, Image.Image):
                data = np.asarray(col)
                log_message += "The data is [{}].".format(data)
            else:
                log_message += "The data is [{}].".format(col)
            logger.info(log_message)
        return args
