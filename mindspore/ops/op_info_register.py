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

"""Operators info register."""

import os
import inspect
from mindspore._c_expression import Oplib
from mindspore._checkparam import ParamValidator as validator

# path of built-in op info register.
BUILT_IN_OPS_REGISTER_PATH = "mindspore/ops/_op_impl"


def op_info_register(op_info):
    """
    A decorator used as register of operator implementation.

    Note:
        'op_info' must be a str of json format represent the op info, the op info will be added into oplib.

    Args:
        op_info (str): op info of json format.

    Returns:
        Function, returns a decorator for op info register.
    """
    def register_decorator(func):
        validator.check_type("op_info", op_info, [str])
        op_lib = Oplib()
        file_path = os.path.realpath(inspect.getfile(func))
        # keep the path custom ops implementation.
        imply_path = "" if BUILT_IN_OPS_REGISTER_PATH in file_path else file_path
        if not op_lib.reg_op(op_info, imply_path):
            raise ValueError('Invalid op info {}:\n{}\n'.format(file_path, op_info))

        def wrapped_function(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped_function
    return register_decorator
