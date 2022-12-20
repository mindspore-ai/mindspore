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

"""grad impl."""
from __future__ import absolute_import
from mindspore.ops._grad.grad_base import get_bprop_fn, get_taylor_fprop_fn
from . import grad_array_ops, grad_comm_ops, grad_debug_ops, grad_implementations, grad_clip_ops, \
              grad_math_ops, grad_nn_ops, grad_other_ops, grad_quant_ops, grad_sparse,            \
              grad_inner_ops, taylor_rule, grad_sequence_ops

__all__ = ['get_bprop_fn', 'get_taylor_fprop_fn']
