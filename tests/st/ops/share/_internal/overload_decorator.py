# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Decorator for ops with different parameters
   with such decorator, code size could be ioptimized
   eg: mint.ops(input) callable: ops(mint.ops), mint.ops(input,dtype=None) callable: ops_dtype(mint.ops)
"""
from typing import Callable

def wrap_op(func:Callable) -> Callable:
    """
       Decorator of operators with parameter (input)
    """
    def wrapper(op_input):
        return func(op_input)
    return wrapper

def wrap_op_with_dtype(func:Callable) -> Callable:
    """
       Decorator of operators with parameter (input, dtype = None)
    """
    def wrapper(op_input, dtype=None):
        return func(op_input, dtype=dtype)
    return wrapper

def wrap_reduction_op_overload(func:Callable, dim_key_arg=False) -> Callable:
    """
       Decorator of operators with parameter (input, dim, keepdim=False)
    """
    def wrapper(op_input, dim=None, keepdim=False):
        if dim_key_arg:
            return func(op_input, dim=dim, keepdim=keepdim)
        return func(op_input, dim, keepdim=keepdim)
    return wrapper

def wrap_reduction_op_overload_with_dtype(func:Callable, dim_key_arg=False) -> Callable:
    """
       Decorator of operators with parameter (input, dim, keepdim=False, *, dtype = None)
    """
    def wrapper(op_input, dim=None, keepdim=False, dtype=None):
        if dim_key_arg:
            return func(op_input, dim=dim, keepdim=keepdim, dtype=dtype)
        return func(op_input, dim, keepdim=keepdim, dtype=dtype)
    return wrapper
