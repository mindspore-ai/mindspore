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
# ==============================================================================
"""
General py_transforms_utils functions.
"""
from __future__ import absolute_import

import sys
import traceback
import numpy as np


def is_numpy(img):
    """
    Check if the input image is Numpy format.

    Args:
        img: Image to be checked.

    Returns:
        Bool, True if input is Numpy image.
    """
    return isinstance(img, np.ndarray)


class KeyErrorParse(str):
    """re-implement repr method, which returns itself in repr"""
    def __repr__(self):
        return self


class ExceptionHandler:
    """Wraps an exception with traceback to be raised in main thread/process"""
    def __init__(self, except_info=None, where="in python function"):
        # catch system exception info, when error raised.
        if except_info is None:
            except_info = sys.exc_info()
        self.where = where
        self.except_type = except_info[0]
        self.except_msg = "".join(traceback.format_exception(*except_info))

    def reraise(self):
        """Reraise the caught exception in the main thread/process"""
        # Find the last traceback which is more useful to user.
        index = [i for i in range(len(self.except_msg)) if self.except_msg.startswith('Traceback', i)]
        err_msg = "{}".format(self.except_msg[index[-1]:]).strip()

        if self.except_type == KeyError:
            # As KeyError will call its repr() function automatically, which makes stack info hard to read.
            err_msg = KeyErrorParse(err_msg)
        elif hasattr(self.except_type, "message"):
            raise self.except_type(message=err_msg)
        raise self.except_type(err_msg)
