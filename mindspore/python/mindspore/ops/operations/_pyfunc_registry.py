# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Register pyfunc for py_func_cpu_kernel"""

from __future__ import absolute_import
from mindspore.ops._register_for_op import PyFuncRegistry


class CustomPyFuncRegistry:
    """
    Registry class for custom pyfunc function.
    Key: func id
    Value : pyfunc
    """

    def __init__(self):
        self._func_dict = PyFuncRegistry()

    @classmethod
    def instance(cls):
        """
        Get singleton of CustomPyFuncRegistry.

        Returns:
            An instance of CustomPyFuncRegistry.
        """
        if not hasattr(CustomPyFuncRegistry, "_instance"):
            CustomPyFuncRegistry._instance = CustomPyFuncRegistry()
        return CustomPyFuncRegistry._instance

    def register(self, fn_id, fn):
        """register id, pyfunc to dict"""
        self._func_dict.register(fn_id, fn)

    def get(self, fn_id):
        """get pyfunc function by id"""
        return self._func_dict.get(fn_id)


def add_pyfunc(fn_id, fn):
    CustomPyFuncRegistry.instance().register(fn_id, fn)


def get_pyfunc(fn_id):
    return CustomPyFuncRegistry.instance().get(fn_id)
