# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""grad experimental impl."""
from __future__ import absolute_import
from mindspore.ops._grad.grad_base import get_bprop_fn
from mindspore.ops._grad_experimental import grad_array_ops
from mindspore.ops._grad_experimental import grad_image_ops
from mindspore.ops._grad_experimental import grad_inner_ops
from mindspore.ops._grad_experimental import grad_nn_ops
from mindspore.ops._grad_experimental import grad_math_ops
from mindspore.ops._grad_experimental import grad_linalg_ops
from mindspore.ops._grad_experimental import grad_sparse
from mindspore.ops._grad_experimental import grad_sparse_ops
from mindspore.ops._grad_experimental import grad_scalar_ops

__all__ = ['get_bprop_fn']
