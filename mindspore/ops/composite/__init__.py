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


from .base import GradOperation, HyperMap, Map, MultitypeFuncGraph, add_flags, \
                  core, env_get, tail, zip_operation
from .clip_ops import clip_by_value, clip_by_global_norm
from .multitype_ops.add_impl import hyper_add
from .multitype_ops.ones_like_impl import ones_like
from .multitype_ops.zeros_like_impl import zeros_like
from .random_ops import normal, laplace, uniform, gamma, poisson, multinomial
from .math_ops import count_nonzero, tensor_dot, dot, batch_dot, matmul
from .array_ops import repeat_elements, sequence_mask


__all__ = [
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
    'normal',
    'laplace',
    'uniform',
    'gamma',
    'poisson',
    'multinomial',
    'clip_by_value',
    'clip_by_global_norm',
    'count_nonzero',
    'tensor_dot',
    'dot',
    'batch_dot',
    'repeat_elements',
    'sequence_mask',
    'matmul']
