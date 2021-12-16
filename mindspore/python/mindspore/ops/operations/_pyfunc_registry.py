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

from mindspore.ops._register_for_op import PyFuncRegistry


registered_py_id = PyFuncRegistry()


def add_pyfunc(fn_id, fn):
    registered_py_id.register(fn_id, fn)


def get_pyfunc(fn_id):
    return registered_py_id.get(fn_id)
