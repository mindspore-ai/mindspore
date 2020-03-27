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

"""
Composite operators.

Pre-defined combination of operators.
"""


from .base import GradOperation, HyperMap, MultitypeFuncGraph, add_flags, \
                  grad, grad_all, grad_all_with_sens, grad_by_list, grad_by_list_with_sens, grad_with_sens, \
                  core, env_get, tail, zip_operation
from .clip_ops import clip_by_value
from .multitype_ops.add_impl import hyper_add
from .multitype_ops.ones_like_impl import ones_like
from .multitype_ops.zeros_like_impl import zeros_like


__all__ = [
    'grad',
    'grad_by_list_with_sens',
    'grad_all',
    'grad_by_list',
    'grad_all_with_sens',
    'grad_with_sens',
    'env_get',
    'core',
    'add_flags',
    'tail',
    'MultitypeFuncGraph',
    'GradOperation',
    'HyperMap',
    'hyper_add',
    'zeros_like',
    'ones_like',
    'zip_operation',
    'clip_by_value']
