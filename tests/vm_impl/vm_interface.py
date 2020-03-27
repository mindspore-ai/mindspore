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
"""vm interface."""
# pylint: disable=wildcard-import,unused-wildcard-import
from .vm_me import *


class Vm:
    pass


vm = Vm()
setattr(vm, 'avg_pooling', avg_pooling)
setattr(vm, 'avg_pool_grad', avg_pool_grad)
setattr(vm, 'batch_norm', batch_norm)
setattr(vm, 'batch_norm_grad', batch_norm_grad)
setattr(vm, 'col2im', col2im)
setattr(vm, 'convolve', convolve)
setattr(vm, 'conv2d', conv2d)
setattr(vm, 'conv2d_backprop_filter', conv2d_backprop_filter)
setattr(vm, 'conv2d_backprop_input', conv2d_backprop_input)
setattr(vm, 'flatten', flatten)
setattr(vm, 'flatten2', flatten2)
setattr(vm, 'flatten_batch', flatten_batch)
setattr(vm, 'flatten_grad', flatten_grad)
setattr(vm, 'im2col', im2col)
setattr(vm, 'matmul', matmul)
setattr(vm, 'max_pooling', max_pooling)
setattr(vm, 'max_pool_grad', max_pool_grad)
setattr(vm, 'max_pool_grad_with_argmax', max_pool_grad_with_argmax)
setattr(vm, 'max_pool_with_argmax', max_pool_with_argmax)
setattr(vm, 'relu', relu)
setattr(vm, 'relu_grad', relu_grad)
setattr(vm, 'softmax', softmax)
setattr(vm, 'softmax_cross_entropy_with_logits', softmax_cross_entropy_with_logits)
setattr(vm, 'expand_dims', expand_dims)
setattr(vm, 'squeeze', squeeze)
setattr(vm, 'reshape', reshape)
setattr(vm, 'shape', shape)
setattr(vm, 'rank', rank)
setattr(vm, 'logsoftmax', logsoftmax)
setattr(vm, 'transpose', transpose)
setattr(vm, 'invert_permutation', invert_permutation)
setattr(vm, 'select', select)
setattr(vm, 'sum', sum_by_axis)
setattr(vm, 'equal', equal)
setattr(vm, 'not_equal', not_equal)
setattr(vm, 'greater', greater)
setattr(vm, 'less', less)
setattr(vm, 'logical_not', logical_not)
setattr(vm, 'sqrt', sqrt)
setattr(vm, 'power', power)
setattr(vm, "exp", exp)
setattr(vm, "tanh", tanh)
setattr(vm, "sigmoid", sigmoid)
setattr(vm, 'maximum', maximum)
setattr(vm, 'minimum', minimum)
