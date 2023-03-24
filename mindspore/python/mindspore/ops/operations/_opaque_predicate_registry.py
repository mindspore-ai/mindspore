# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Register pyfunc for opaque_func_cpu_kernel"""

from mindspore.ops._register_for_op import OpaquePredicateRegistry


registered_func_name = OpaquePredicateRegistry()


def add_opaque_predicate(fn_name, func):
    """restore opaque predicate functions"""
    registered_func_name.register(fn_name, func)


def get_opaque_predicate(fn_name):
    """get opaque predicate function by their name"""
    return registered_func_name.get(fn_name)


def get_func_names():
    """get function names"""
    return registered_func_name.func_names


def clean_funcs():
    """clean restored functions"""
    registered_func_name.func_names = []
