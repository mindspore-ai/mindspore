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

from __future__ import absolute_import
from mindspore.ops.composite.base import GradOperation, _Grad, HyperMap, Map, MultitypeFuncGraph, add_flags, \
    tail, zip_operation, _Vmap, _TaylorOperation
from mindspore.ops.composite.env_ops import env_get
from mindspore.ops.function.clip_func import clip_by_global_norm
from mindspore.ops.composite.multitype_ops.add_impl import hyper_add
from mindspore.ops.composite.multitype_ops.ones_like_impl import ones_like
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.function.random_func import normal, laplace, uniform, gamma, poisson, multinomial
from mindspore.ops.composite.math_ops import matmul, cummin, mm
from mindspore.ops.function.math_func import count_nonzero, tensor_dot, vecdot, dot, batch_dot
from mindspore.ops.function.array_func import repeat_interleave, repeat_elements, sequence_mask
from mindspore.ops.function.vmap_func import _VmapGeneralPreprocess, _VmapGeneralRule
from mindspore.ops.function.clip_func import clip_by_value


__all__ = [
    'env_get',
    'add_flags',
    'tail',
    'Map',
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
    'cummin',
    'repeat_elements',
    'repeat_interleave',
    'sequence_mask',
    'matmul',
    'mm',
    'count_nonzero',
    'tensor_dot',
    'vecdot',
    'dot',
    'batch_dot',
    '_Grad',
    '_Vmap']
