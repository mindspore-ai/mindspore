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

"""vmap impl."""
from __future__ import absolute_import

from . import vmap_base, vmap_array_ops, vmap_grad_nn_ops, vmap_debug_ops, vmap_math_ops, vmap_nn_ops,\
    vmap_image_ops, vmap_other_ops, vmap_sparse_ops, vmap_random_ops, vmap_convolution_ops, vmap_grad_math_ops
from .vmap_base import get_vmap_rule, vmap_monad_rule, _broadcast_by_axis, vmap_bind_all_none,\
    vmap_unstack, vmap_stack, vmap_general_output_process

__all__ = ['get_vmap_rule', 'vmap_monad_rule', '_broadcast_by_axis', 'vmap_bind_all_none',
           'vmap_unstack', 'vmap_stack', 'vmap_general_output_process']
