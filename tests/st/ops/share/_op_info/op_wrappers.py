# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
# pylint: skip-file
"""API wrapper functions for operation testing.

This module provides wrapper functions that adapt MindSpore and PyTorch
tensor/nn methods to a uniform call signature used by the OpInfo framework.
"""
import mindspore as ms
from mindspore import nn, mint
import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.experimental import optim
import torch
from torch.nn import functional as torch_F

# =====================================wrap tensor/nn method====================================

# wrap tensor method for astype
def tensor_astype_ms(op_input, dtype=ms.float32, copy=False):
    return op_input.astype(dtype=dtype, copy=copy)

def tensor_astype_torch(op_input,  dtype=torch.float32, copy=False):
    return op_input.type_as(torch.tensor(1.0, dtype=dtype))

# wrap tensor method for byte
def tensor_byte_ms(op_input):
    return op_input.byte()

def tensor_byte_torch(op_input):
    return op_input.byte()

# wrap tensor method for clone
def tensor_clone_ms(op_input):
    return op_input.clone()

def tensor_clone_torch(op_input):
    return op_input.clone()

# wrap tensor method for contiguous
def tensor_contiguous_ms(op_input):
    return op_input.contiguous()

def tensor_contiguous_torch(op_input):
    return op_input.contiguous()

# wrap tensor method for is_contiguous
def tensor_is_contiguous_ms(op_input):
    return op_input.is_contiguous()

def tensor_is_contiguous_torch(op_input):
    return op_input.is_contiguous()

# wrap tensor method for matmul
def tensor_matmul_ms(op_input, x):
    return op_input.matmul(x)

def tensor_matmul_torch(op_input, x):
    return op_input.matmul(x)

# wrap tensor method for numpy
def tensor_numpy_ms(op_input):
    return op_input.numpy()

def tensor_numpy_torch(op_input):
    return op_input.numpy()


def mint_one_hot_torch(op_input, *args, **kwargs):
    return torch_F.one_hot(op_input.to(torch.long), *args, **kwargs)

# wrap tensor method for tanh
def tensor_tanh_ms(op_input):
    return op_input.tanh()

def tensor_ceil_ms(op_input):
    return op_input.ceil()

def tensor_exp_ms(op_input):
    return op_input.exp()

def tensor_log_ms(op_input):
    return op_input.log()

def tensor_flatten_ms(op_input, start_dim=0, end_dim=-1):
    return op_input.flatten(start_dim=start_dim, end_dim=end_dim)

def tensor_flatten_torch(op_input, start_dim=0, end_dim=-1):
    return op_input.flatten(start_dim=start_dim, end_dim=end_dim)

def tensor_neg_ms(op_input):
    return op_input.neg()

def tensor_sigmoid_ms(op_input):
    return op_input.sigmoid()

def tensor_sqrt_ms(op_input):
    return op_input.sqrt()

def tensor_square_ms(op_input):
    return op_input.square()

def tensor_select_ms(op_input, dim, index):
    return op_input.select(dim, index)

def tensor_floor_ms(op_input):
    return op_input.floor()

def tensor_abs_ms(op_input):
    return op_input.abs()

def tensor_floor_divide_ms(op_input, other):
    return op_input.floor_divide(other)

def tensor_tanh_torch(op_input):
    return op_input.tanh()

def tensor_repeat_interleave_ms(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim, output_size=output_size)

def tensor_mint_var_mean_ms(op_input, dim=None, keepdim=False, correction=1):
    return mint.var_mean(op_input, dim=dim,correction = correction,keepdim=keepdim)

def tensor_repeat_interleave_torch(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim=dim, output_size=output_size)

def tensor_repeat_interleave_func_grad(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim, output_size=output_size)

def nn_functional_selu_ms(op_input):
    return mint.nn.functional.selu(op_input)

def nn_functional_selu_torch(op_input):
    return torch.nn.functional.selu(op_input)

def tensor_add__ms(op_input, other, alpha=1):
    """Wrapper for Tensor.add_ (in-place operation)."""
    #TODO:The is_inplace_op item of OpInfo still has problem. Should be fixed in the future.
    cloned_input = op_input.clone()
    cloned_input.add_(other, alpha=alpha)
    return cloned_input

def tensor_add__torch(op_input, other, alpha=1):
    """Wrapper for torch.Tensor.add_ (in-place operation)."""
    cloned_input_pt = op_input.clone()
    cloned_input_pt.add_(other, alpha=alpha)
    return cloned_input_pt

def tensor_masked_scatter_ms(op_input, mask, source):
    return op_input.masked_scatter(mask, source)

def tensor_masked_scatter_torch(op_input, mask, source):
    return op_input.masked_scatter(mask, source)

def tensor_masked_scatter__ms(op_input, mask, source):
    """Wrapper for Tensor.masked_scatter_ (in-place operation)."""
    # For in-place operations, we need to clone the input to avoid modifying the original
    # in the test framework, but the actual operation modifies in-place
    cloned_input = op_input.clone()
    result = cloned_input.masked_scatter_(mask, source)
    return result

def tensor_masked_scatter__torch(op_input, mask, source):
    """Wrapper for torch.Tensor.masked_scatter_ (in-place operation)."""
    cloned_input = op_input.clone()
    result = cloned_input.masked_scatter_(mask, source)
    return result

def tensor_eq_ms(input, other):
    return input.eq(other)

def tensor_eq_torch(input, other):
    return input.eq(other)

def tensor_operator_eq_ms(input, other):
    return input == other

def tensor_operator_eq_torch(input, other):
    return input == other

def tensor_repeat_ms(op_input, *sizes):
    return op_input.repeat(*sizes)

def tensor_repeat_torch(op_input, *sizes):
    return op_input.repeat(*sizes)

def tensor_repeat_ms_overload1(op_input, repeats):
    return op_input.repeat(repeats)

def tensor_repeat_torch_overload1(op_input, repeats):
    return op_input.repeat(repeats)

def tensor_reshape_ms(op_input, *sizes):
    return op_input.reshape(*sizes)

def tensor_reshape_torch(op_input, *sizes):
    return op_input.reshape(*sizes)

def tensor_greater_equal_ms(op_input, x):
    return op_input.greater_equal(x)

def tensor_greater_equal_torch(op_input, x):
    return op_input.greater_equal(x)

def tensor_greater_ms(op_input, x):
    return op_input.greater(x)

def tensor_greater_torch(op_input, x):
    return op_input.greater(x)

def tensor_less_equal_ms(op_input, x):
    return op_input.less_equal(x)

def tensor_less_equal_torch(op_input, x):
    return op_input.less_equal(x)

def tensor_less_ms(op_input, x):
    return op_input.less(x)

def less_operator_ms(op_input, x):
    return op_input < x

def less_operator_torch(op_input, x):
    return op_input < x

def tensor_less_torch(op_input, x):
    return op_input.less(x)

def tensor_ne_ms(op_input, x):
    return op_input.ne(x)

def tensor_ne_torch(op_input, x):
    return op_input.ne(x)

def tensor_atan2_ms(op_input, x):
    return op_input.atan2(x)

def tensor_atan2_torch(op_input, x):
    return op_input.atan2(x)

def tensor_gt_ms(op_input, x):
    return op_input.gt(x)

def tensor_gt_torch(op_input, x):
    return op_input.gt(x)

def tensor_le_ms(op_input, x):
    return op_input.le(x)

def tensor_le_torch(op_input, x):
    return op_input.le(x)

def tensor_lt_ms(op_input, x):
    return op_input.lt(x)

def tensor_lt_torch(op_input, x):
    return op_input.lt(x)

def tensor_maximum_ms(op_input, other):
    return op_input.maximum(other)

def tensor_maximum_torch(op_input, other):
    return op_input.maximum(other)

def tensor_minimum_ms(op_input, other):
    return op_input.minimum(other)

def tensor_minimum_torch(op_input, other):
    return op_input.minimum(other)

def tensor_mul_ms(op_input, other):
    return op_input.mul(other)

def tensor_mul_torch(op_input, other):
    return op_input.mul(other)

def tensor_ceil_torch(op_input):
    return op_input.ceil()

def tensor_exp_torch(op_input):
    return op_input.exp()

def tensor_log_torch(op_input):
    return op_input.log()

def tensor_neg_torch(op_input):
    return op_input.neg()

def tensor_select_torch(op_input, dim, index):
    return op_input.select(dim, index)

def tensor_sigmoid_torch(op_input):
    return op_input.sigmoid()

def tensor_sqrt_torch(op_input):
    return op_input.sqrt()

def tensor_square_torch(op_input):
    return op_input.square()

def tensor_floor_torch(op_input):
    return op_input.floor()

def tensor_abs_torch(op_input):
    return op_input.abs()

def tensor_floor_divide_torch(op_input, other):
    return op_input.floor_divide(other)

def tensor_to_ms(op_input, dtype=ms.float32):
    return op_input.to(dtype=dtype)

# tensor.to has keyword parameter
def tensor_to_torch(op_input, dtype=torch.float32):
    return op_input.to(dtype=dtype)

def tensor_to_ms_overload1(op_input, dtype=None, non_blocking=False, copy=False):
    return op_input.to(dtype=dtype, non_blocking=non_blocking, copy=copy)

def tensor_to_torch_overload1(op_input, dtype=None, non_blocking=False, copy=False):
    return op_input.to(dtype=dtype, non_blocking=non_blocking, copy=copy)

def tensor_to_ms_overload2(op_input, device=None, dtype=None, non_blocking=False, copy=False):
    return op_input.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

def tensor_to_torch_overload2(op_input, device=None, dtype=None, non_blocking=False, copy=False):
    return op_input.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

def tensor_div_ms(op_input, other, rounding_mode=None):
    return mint.div(op_input, other, rounding_mode=rounding_mode)

# wrap method for empty
def empty_ms(op_input, dtype=None, device=None, pin_memory=False):
    return mint.empty(op_input.shape, dtype=dtype, device=device, pin_memory=pin_memory).shape

def empty_torch(op_input, dtype=None, device=None, pin_memory=False):
    return torch.empty(op_input.shape, dtype=dtype).shape

# wrap method for empty_like
def empty_like_ms(op_input, dtype=None, device=None, pin_memory=False):
    return mint.empty_like(op_input, dtype=dtype, device=device, pin_memory=pin_memory).shape

def empty_like_torch(op_input, dtype=None, device=None, pin_memory=False):
    return torch.empty_like(op_input, dtype=dtype).shape

def mint_special_log_softmax_ms(op_input, dim=0, dtype=None):
    return mint.special.log_softmax(op_input, dim=dim, dtype=dtype)

def mint_special_log_softmax_torch(op_input, dim=0, dtype=None):
    return torch.special.log_softmax(op_input, dim=dim, dtype=dtype)

# wrap nn method for batchnorm1d
def nn_batchnorm1d_ms(op_input, num_features, eps, momentum, affine, track_running_stats, dtype=None):
    op = mint.nn.BatchNorm1d(
        num_features=num_features, eps=eps, momentum=momentum, affine=affine,
        track_running_stats=track_running_stats, dtype=dtype
    )
    op.set_train()
    return op(op_input)

def nn_batchnorm1d_torch(op_input, num_features=9, eps=2.0, momentum=-2.0, affine=True,
                         track_running_stats=False, dtype=None):
    return torch.nn.BatchNorm1d(
        num_features=num_features, eps=eps, momentum=momentum, affine=affine,
        track_running_stats=track_running_stats, dtype=dtype
    )(op_input)

# wrap nn method for batchnorm2d
def nn_batchnorm2d_ms(op_input, num_features, eps, momentum, affine, track_running_stats, dtype=None):
    op = mint.nn.BatchNorm2d(
        num_features=num_features, eps=eps, momentum=momentum, affine=affine,
        track_running_stats=track_running_stats, dtype=dtype
    )
    op.set_train()
    return op(op_input)

def nn_batchnorm2d_torch(op_input, num_features=3, eps=565.73012560178, momentum=-2.0, affine=True,
                         track_running_stats=False, dtype=None):
    return torch.nn.BatchNorm2d(
        num_features=num_features, eps=eps, momentum=momentum, affine=affine,
        track_running_stats=track_running_stats, dtype=dtype
    )(op_input)

# wrap nn method for batchnorm3d
def nn_batchnorm3d_ms(op_input, num_features, eps, momentum, affine, track_running_stats, dtype=None):
    op = mint.nn.BatchNorm3d(
        num_features=num_features, eps=eps, momentum=momentum, affine=affine,
        track_running_stats=track_running_stats, dtype=dtype
    )
    op.set_train()
    return op(op_input)

def nn_batchnorm3d_torch(op_input, num_features=3, eps=121.30250066286163, momentum=-2.0,
                         affine=True, track_running_stats=False, dtype=None):
    return torch.nn.BatchNorm3d(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
        dtype=dtype,
    )(op_input)

# wrap nn method for gelu
def nn_gelu_ms(op_input, approximate='none'):
    return mint.nn.GELU(approximate=approximate)(op_input)

def nn_gelu_torch(op_input, approximate='none'):
    return torch.nn.GELU(approximate=approximate)(op_input)

def nn_functional_gelu_ms(op_input, approximate='none'):
    return mint.nn.functional.gelu(op_input, approximate=approximate)

def nn_functional_gelu_torch(op_input, approximate='none'):
    return torch_F.gelu(op_input, approximate=approximate)

def nn_functional_leaky_relu_ms(op_input, negative_slope=0.01):
    return mint.nn.functional.leaky_relu(op_input, negative_slope=negative_slope)

def nn_functional_leaky_relu_torch(op_input, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(op_input, negative_slope=negative_slope)

# wrap nn method for glu
def nn_glu_ms(op_input, *args):
    return mint.nn.GLU(*args)(op_input)

def nn_glu_torch(op_input, *args):
    return torch.nn.GLU(*args)(op_input)

# wrap nn method for identity
def nn_identity_ms(op_input):
    return mint.nn.Identity()(op_input)

def nn_identity_torch(op_input):
    return torch.nn.Identity()(op_input)

# wrap nn method for logsigmoid
def nn_logsigmoid_ms(op_input):
    return mint.nn.LogSigmoid()(op_input)

def nn_logsigmoid_torch(op_input):
    return torch.nn.LogSigmoid()(op_input)

# wrap nn method for prelu
def nn_prelu_ms(op_input, num_parameters=1, init=0.25, dtype=None):
    return mint.nn.PReLU(num_parameters=num_parameters, init=init, dtype=dtype)(op_input)

def nn_prelu_torch(op_input, num_parameters=1, init=0.25, dtype=None):
    return torch.nn.PReLU(num_parameters=num_parameters, init=init, dtype=dtype)(op_input)

# wrap nn method for relu
def nn_relu_ms(op_input):
    return mint.nn.ReLU()(op_input)

def nn_relu_torch(op_input):
    return torch.nn.ReLU()(op_input)

# wrap nn method for relu6
def nn_relu6_ms(op_input):
    return mint.nn.ReLU6()(op_input)

def nn_relu6_torch(op_input):
    return torch.nn.ReLU6()(op_input)

def tensor_tile_ms(op_input, dims):
    return op_input.tile(dims)

def tensor_tile_torch(op_input, dims):
    return op_input.tile(dims)

def tensor_reciprocal_ms(op_input):
    return op_input.reciprocal()

def tensor_reciprocal_torch(op_input):
    return op_input.reciprocal()

def tensor_unsqueeze_ms(op_input, dim):
    return op_input.unsqueeze(dim)

def tensor_unsqueeze_torch(op_input, dim):
    return op_input.unsqueeze(dim)

def tensor_topk_ms(op_input, *args, **kwargs):
    return op_input.topk(*args, **kwargs)

def tensor_topk_torch(op_input, *args, **kwargs):
    return op_input.topk(*args, **kwargs)

def tensor_transpose_ms(op_input, *args, **kwargs):
    return op_input.transpose(*args, **kwargs)

def tensor_transpose_torch(op_input, *args, **kwargs):
    return op_input.transpose(*args, **kwargs)

# wrap nn method for tanh
def nn_tanh_ms(op_input):
    return mint.nn.Tanh()(op_input)

def nn_tanh_torch(op_input):
    return torch.nn.Tanh()(op_input)

# wrap tensor method for expand_as
def tensor_expand_as_ms(op_input, other):
    return op_input.expand_as(other)

def tensor_expand_as_torch(op_input, other):
    return op_input.expand_as(other)

# wrap tensor method for sin
def tensor_sin_ms(op_input):
    return op_input.sin()

def tensor_sin_torch(op_input):
    return op_input.sin()

# wrap tensor method for add
def tensor_add_ms(op_input, other, alpha=1):
    cloned_input = op_input.clone()
    return cloned_input.add(other, alpha=alpha)

def tensor_add_torch(op_input, other, alpha=1):
    cloned_input_pt = op_input.clone()
    return cloned_input_pt.add(other, alpha=alpha)

# wrap tensor method for sub
def tensor_sub_ms(op_input, args):
    return op_input.sub(args)

def tensor_sub_torch(op_input, args):
    return op_input.sub(args)

# wrap tensor method for logical_and
def tensor_logical_and_ms(op_input, args):
    return op_input.logical_and(args)

def tensor_logical_and_torch(op_input, args):
    return op_input.logical_and(args)

# wrap tensor method for logical_not
def tensor_logical_not_ms(op_input):
    return op_input.logical_not()

def tensor_logical_not_torch(op_input):
    return op_input.logical_not()

# wrap tensor method for bfloat16
def tensor_bfloat16_ms(op_input):
    return op_input.bfloat16()

def tensor_bfloat16_torch(op_input):
    return op_input.bfloat16()

# wrap tensor method for bool
def tensor_bool_ms(op_input):
    return op_input.bool()

def tensor_bool_torch(op_input):
    return op_input.bool()

# wrap tensor method for double
def tensor_double_ms(op_input):
    return op_input.double()

def tensor_double_torch(op_input):
    return op_input.double()

# wrap tensor method for float
def tensor_float_ms(op_input):
    return op_input.float()

def tensor_float_torch(op_input):
    return op_input.float()

# wrap tensor method for half
def tensor_half_ms(op_input):
    return op_input.half()

def tensor_half_torch(op_input):
    return op_input.half()

# wrap tensor method for int
def tensor_int_ms(op_input):
    return op_input.int()

def tensor_int_torch(op_input):
    return op_input.int()

# wrap tensor method for long
def tensor_long_ms(op_input):
    return op_input.long()

def tensor_long_torch(op_input):
    return op_input.long()

# wrap tensor method for logical_or
def tensor_logical_or_ms(op_input, args):
    return op_input.logical_or(args)

def tensor_logical_or_torch(op_input, args):
    return op_input.logical_or(args)

# wrap tensor method for split
def tensor_split_ms(op_input, args, dim):
    return op_input.split(args, dim)

def tensor_split_torch(op_input, args, dim):
    return op_input.split(args, dim)

# wrap tensor method for index_select
def tensor_index_select_ms(op_input, dim, index):
    return op_input.index_select(dim, index)

def tensor_index_select_torch(op_input, dim, index):
    return op_input.index_select(dim, index)

# wrap tensor method for cos
def tensor_cos_ms(op_input):
    return op_input.cos()

def tensor_cos_torch(op_input):
    return op_input.cos()

def nn_group_norm_ms(op_input, num_groups, num_channels, eps=1e-5, affine=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = mint.nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    return module(op_input)

def nn_group_norm_torch(op_input, num_groups, num_channels, eps=1e-5, affine=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = torch.nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    return module(op_input)

def nn_layer_norm_ms(op_input, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = mint.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, dtype=dtype)
    return module(op_input)

def nn_layer_norm_torch(op_input, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    dtype = op_input.dtype if hasattr(op_input, 'dtype') else None
    module = torch.nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, dtype=dtype)
    return module(op_input)

# wrap nn method for AdaptiveAvgPool2d
def nn_adaptiveavgpool2d_ms(op_input, output_size):
    return mint.nn.AdaptiveAvgPool2d(output_size)(op_input)

def nn_adaptiveavgpool2d_torch(op_input, output_size):
    return torch.nn.AdaptiveAvgPool2d(output_size)(op_input)

# wrap nn method for AdaptiveAvgPool3d
def nn_adaptiveavgpool3d_ms(op_input, output_size):
    return mint.nn.AdaptiveAvgPool3d(output_size)(op_input)

def nn_adaptiveavgpool3d_torch(op_input, output_size):
    return torch.nn.AdaptiveAvgPool3d(output_size)(op_input)

# wrap tensor method for gather
def tensor_gather_ms(input, dim, index):
    return input.gather(dim, index)

def tensor_gather_overload_ms(input, input_indices, axis):
    return input.gather(input_indices, axis)

def tensor_gather_torch(input, dim, index):
    return input.gather(dim, index)

# wrap tensor method for unique
def tensor_unique_ms(op_input, *op_args, **op_kwargs):
    return op_input.unique(*op_args, **op_kwargs)

def tensor_unique_torch(op_input, *op_args, **op_kwargs):
    return op_input.unique(*op_args, **op_kwargs)

# wrap tensor method for clamp
def tensor_clamp_ms(op_input, *op_args, **op_kwargs):
    return op_input.clamp(*op_args, **op_kwargs)

def tensor_clamp_torch(op_input, *op_args, **op_kwargs):
    return op_input.clamp(*op_args, **op_kwargs)

def mint_clamp_ms(op_input, min=None, max=None):
    return mint.clamp(op_input, min=min, max=max)

def mint_clamp_torch(op_input, min=None, max=None):
    return torch.clamp(op_input, min=min, max=max)

def mint_flatten_ms(op_input, start_dim=0, end_dim=-1):
    return mint.flatten(op_input, start_dim=start_dim, end_dim=end_dim)

def mint_flatten_torch(op_input, start_dim=0, end_dim=-1):
    return torch.flatten(op_input, start_dim=start_dim, end_dim=end_dim)

def mint_eq_with_explicit_args(input, other):
    return mint.eq(input, other)

#  logsumexp'dim default no value,test will fail when kwargs={},so make dim=0
def tensor_mint_logsumexp_ms(op_input, dim=0, keepdim=False):
    return mint.logsumexp(op_input, dim, keepdim)

#  logsumexp'dim default no value,test will fail when kwargs={},so make dim=0
def tensor_torch_logsumexp(op_input, dim=0, keepdim=False):
    return torch.logsumexp(op_input, dim, keepdim)

#  nansum has keyword parameters dtype,so for matching cases make this fouctions
def tensor_mint_nansum(input, dim=None, keepdim=False, dtype=None):
    return mint.nansum(input, dim, keepdim, dtype=dtype)

#  nansum has keyword parameters dtype,so for matching cases make this fouctions
def tensor_torch_nansum(input, dim=None, keepdim=False, dtype=None):
    return torch.nansum(input, dim, keepdim, dtype=dtype)

def tensor_batch_norm_ms(op_input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5):
    output = mint.nn.functional.batch_norm(op_input, running_mean, running_var, weight, bias, training, momentum, eps)
    return (output, running_mean, running_var)

def tensor_batch_norm_torch(op_input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5):
    output = torch.nn.functional.batch_norm(op_input, running_mean, running_var, weight, bias, training, momentum, eps)
    return (output, running_mean, running_var)

def dropout_fallback_ms(op_input):
    if op_input.dtype.is_floating_point:
        return op_input * 0
    return mint.zeros_like(op_input)

def dropout_fallback_torch(op_input):
    if op_input.is_floating_point():
        return op_input * 0
    return torch.zeros_like(op_input)

def tensor_nn_functional_dropout_ms_decorator(op):
    def _dropout_func(op_input, p=0.5, training=False, inplace=False):
        if 0 < p < 1:
            if training:
                if op_input.dtype.is_floating_point and op_input.dtype != ms.float64 and op_input.size != 0:
                    repeat_count = 400
                    sizes = 0
                    m1 = None
                    input_clone = None
                    for _ in range(repeat_count):
                        input_clone = op_input.clone()
                        m1 = op(input_clone, p=p, training=training, inplace=inplace)
                        sizes = sizes + (m1 == 0).sum().item()
                    mask = (m1 == 0).astype(op_input.dtype)
                    modified_input = op_input * (1 - mask)
                    result = (1 / (1 - p) * modified_input - m1,
                              ms.Tensor(sizes / (repeat_count * op_input.size), dtype=op_input.dtype))
                    if inplace:
                        result = result + (m1 - input_clone,)
                    return result
                return dropout_fallback_ms(op_input)
        return op(op_input, p=p, training=training, inplace=inplace)
    return _dropout_func

def tensor_nn_functional_dropout_torch_decorator(op):
    def _dropout_func(op_input, p=0.5, training=False, inplace=False):
        if 0 < p < 1:
            if training:
                if op_input.dtype.is_floating_point and op_input.dtype != torch.float64 and op_input.numel() != 0:
                    repeat_count = 400
                    sizes = 0
                    t1 = None
                    input_clone = None
                    for _ in range(repeat_count):
                        input_clone = op_input.clone()
                        t1 = op(input_clone, p=p, training=training, inplace=inplace)
                        sizes = sizes + (t1 == 0).sum().item()
                    mask = (t1 == 0).to(op_input.dtype)
                    modified_input = op_input * (1 - mask)
                    result = (1 / (1 - p) * modified_input - t1,
                              torch.tensor(sizes / (repeat_count * op_input.numel()), dtype=op_input.dtype))
                    if inplace:
                        result = result + (t1 - input_clone,)
                    return result
                return dropout_fallback_torch(op_input)
        return op(op_input, p=p, training=training, inplace=inplace)
    return _dropout_func

def tensor_triangular_solve_ms(B, A, upper=True, transpose=False, unitriangular=False ):
    return mint.triangular_solve(B, A, upper=upper, transpose=transpose, unitriangular=unitriangular)[0]

def tensor_triangular_solve_torch(B, A, upper=True, transpose=False, unitriangular=False ):
    return torch.triangular_solve(B, A, upper=upper, transpose=transpose, unitriangular=unitriangular)[0]

def tensor_baddbmm_ms(input, batch1, batch2, beta=1, alpha=1 ):
    return mint.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)

def tensor_addmv_ms(input, mat1, mat2, beta=1, alpha=1 ):
    return mint.addmv(input, mat1, mat2, beta=beta, alpha=alpha)

def tensor_addmv_torch(input, mat1, mat2, beta=1, alpha=1 ):
    return torch.addmv(input, mat1, mat2, beta=beta, alpha=alpha)

def mint_addmm_ms(input, mat1, mat2, beta=1, alpha=1 ):
    return mint.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def mint_addmm_torch(input, mat1, mat2, beta=1, alpha=1 ):
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def tensor_addmm_ms(input, mat1, mat2, beta=1, alpha=1 ):
    return input.addmm(mat1, mat2, beta=beta, alpha=alpha)

def tensor_addmm_torch(input, mat1, mat2, beta=1, alpha=1 ):
    return input.addmm(mat1, mat2, beta=beta, alpha=alpha)

# NOTE: customize_op_func, used by gradient comparison only when there are kwargs in op!
def add_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.add(x, y, alpha=alpha)

def sub_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.sub(x, y, alpha=alpha)

def div_func_grad(op_input, other, rounding_mode=None):
    return mint.div(op_input, other, rounding_mode=rounding_mode)

def var_func_grad(op_input, dim=None, keepdim=False, correction=1):
    return mint.var(op_input, dim=dim, keepdim=keepdim, correction=correction)

def std_func_grad(op_input, dim=None, keepdim=False, correction=1):
    return mint.std(op_input, dim=dim, keepdim=keepdim, correction=correction)

def std_mean_func_grad(op_input, dim=None, keepdim=False, correction=1):
    return mint.std_mean(op_input, dim=dim, keepdim=keepdim, correction=correction)

def repeat_interleave_func_grad(input_x, repeats, dim=None, output_size=None):
    return mint.repeat_interleave(input_x, repeats, dim, output_size=output_size)

def mint_min_func_grad_without_kwargs(op_input, *args):
    return mint.min(op_input, *args)

def op_func_grad_without_kwargs(op):
    '''
    Generate arg_names for dynamic kwargs when using operators defined with ``*args, **kwargs``
    signature and decorated with ``enable_dynamic``.
    Args:
        op(callable): The operator function to process. The operator should be defined with
         ``*args, **kwargs`` signature.
    Returns:
        function: A wrapped operator function that handles dynamic keyword arguments.
    '''
    def op_func(x, y, *args):
        return op(x, y, *args)
    return op_func

def tensor_mint_linalg_matrix_norm(input, ord, dim, keepdim, dtype):
    return mint.linalg.matrix_norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def tensor_torch_linalg_matrix_norm(input, ord, dim, keepdim, dtype):
    return torch.linalg.matrix_norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def tensor_mint_linalg_vector_norm(input, ord, dim, keepdim, dtype):
    return mint.linalg.vector_norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def tensor_torch_linalg_vector_norm(input, ord, dim, keepdim, dtype):
    return torch.linalg.vector_norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def tensor_mint_linalg_norm(input, ord, dim, keepdim, dtype):
    return mint.linalg.norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def tensor_torch_linalg_norm(input, ord, dim, keepdim, dtype):
    return torch.linalg.norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def tensor_mint_allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    return ms.Tensor(mint.allclose(input, other, rtol, atol, equal_nan))

def tensor_torch_allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    return torch.tensor(torch.allclose(input, other, rtol, atol, equal_nan))

def tensor_mint_addbmm(input, batch1, batch2, beta=1, alpha=1):
    return mint.addbmm(input, batch1, batch2, beta=beta, alpha=alpha)

def tensor_torch_addbmm(input, batch1, batch2, beta=1, alpha=1):
    return torch.addbmm(input, batch1, batch2, beta=beta, alpha=alpha)

#tensor.sum has keyword parameter dtype
def tensor_sum_ms(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.sum(dim, keepdim, dtype=dtype)

#tensor.sum has keyword parameter dtype
def tensor_sum_torch(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.sum(dim, keepdim, dtype=dtype)

#tensor.mean has keyword parameter dtype
def tensor_mean_ms(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.mean(dim, keepdim, dtype=dtype)

#tensor.mean has keyword parameter dtype
def tensor_mean_torch(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.mean(dim, keepdim, dtype=dtype)

def tensor_argmax_ms(op_input, *op_args, **op_kwargs):
    return op_input.argmax(*op_args, **op_kwargs)

def tensor_argmax_torch(op_input, *op_args, **op_kwargs):
    return op_input.argmax(*op_args, **op_kwargs)

def tensor_argmin_ms(op_input, *op_args, **op_kwargs):
    return op_input.argmin(*op_args, **op_kwargs)

def tensor_argmin_torch(op_input, *op_args, **op_kwargs):
    return op_input.argmin(*op_args, **op_kwargs)

def tensor_max_ms(op_input, *op_args, **op_kwargs):
    return op_input.max(*op_args, **op_kwargs)

def tensor_max_torch(op_input, *op_args, **op_kwargs):
    return op_input.max(*op_args, **op_kwargs)

def tensor_view_dtype_ms(op_input, dtype):
    return op_input.view(dtype)

def tensor_sign_to_sgn(op_input, *op_args, **op_kwargs):
    if isinstance(op_input, torch.Tensor) and op_input.dtype.is_complex:
        return torch.sgn(op_input, *op_args, **op_kwargs)
    return torch.sign(op_input, *op_args, **op_kwargs)

# wrapper function for mindspore.mint.nn.Conv1d
def mint_nn_conv1d_ms(op_input, in_channels, out_channels, kernel_size, stride=1, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None,
                      weight_init=None, bias_init=None):
    """Wrapper for mindspore.mint.nn.Conv1d."""
    conv1d_layer = mint.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    conv1d_layer.weight = Parameter(weight_init, name='weight')
    if bias:
        conv1d_layer.bias = Parameter(bias_init, name='bias')
    return conv1d_layer(op_input)


def nn_conv1d_torch(op_input, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                         dtype=None, weight_init=None, bias_init=None):
    """Wrapper for torch.nn.Conv1d."""
    conv1d_layer = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    conv1d_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        conv1d_layer.bias = torch.nn.Parameter(bias_init)
    return conv1d_layer(op_input)


# wrapper function for mindspore.mint.nn.Conv2d
def mint_nn_conv2d_ms(op_input, in_channels, out_channels, kernel_size, stride=1, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None,
                      weight_init=None, bias_init=None):
    """Wrapper for mindspore.mint.nn.Conv2d."""
    conv2d_layer = mint.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    if weight_init is not None:
        conv2d_layer.weight = Parameter(weight_init, name='weight')
    if bias:
        conv2d_layer.bias = Parameter(bias_init, name='bias')
    return conv2d_layer(op_input)

def nn_conv2d_torch(op_input, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                         dtype=None, weight_init=None, bias_init=None):
    """Wrapper for torch.nn.Conv2d."""
    conv2d_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    if weight_init is not None:
        conv2d_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        conv2d_layer.bias = torch.nn.Parameter(bias_init)
    return conv2d_layer(op_input)

def mint_rand_ms(op_input, generator=None, dtype=None, device=None):
    return mint.rand(op_input.shape, generator=generator, dtype=dtype, device=device)

def _drop_torch_random_device(device):
    """PyTorch benchmark for random ops always runs on default CPU."""
    del device


def _drop_torch_random_device_kwarg(op_kwargs):
    op_kwargs = dict(op_kwargs)
    op_kwargs.pop('device', None)
    return op_kwargs

def mint_rand_torch(op_input, generator=None, dtype=None, device=None):
    _drop_torch_random_device(device)
    return torch.rand(op_input.shape, generator=generator, dtype=dtype)

def mint_randn_ms(op_input, generator=None, dtype=None, device=None):
    return mint.randn(op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_randn_torch(op_input, generator=None, dtype=None, device=None):
    _drop_torch_random_device(device)
    return torch.randn(op_input.shape, generator=generator, dtype=dtype)

def mint_randint_ms(op_input, low, high, generator=None, dtype=None, device=None):
    return mint.randint(low, high, op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_randint_torch(op_input, low, high, generator=None, dtype=None, device=None):
    _drop_torch_random_device(device)
    return torch.randint(low, high, op_input.shape, generator=generator, dtype=dtype)

def mint_rand_like_ms(op_input, dtype=None, **op_kwargs):
    return mint.rand_like(op_input, dtype=dtype, **op_kwargs)

def mint_rand_like_torch(op_input, dtype=None, **op_kwargs):
    op_kwargs = _drop_torch_random_device_kwarg(op_kwargs)
    return torch.rand_like(op_input, dtype=dtype, **op_kwargs)

def mint_randn_like_ms(op_input, dtype=None, **op_kwargs):
    return mint.randn_like(op_input, dtype=dtype, **op_kwargs)

def mint_randn_like_torch(op_input, dtype=None, **op_kwargs):
    op_kwargs = _drop_torch_random_device_kwarg(op_kwargs)
    return torch.randn_like(op_input, dtype=dtype, **op_kwargs)

def mint_randint_like_ms(op_input, low, high, dtype=None, **op_kwargs):
    return mint.randint_like(op_input, low, high, dtype=dtype, **op_kwargs)

def mint_randint_like_torch(op_input, low, high, dtype=None, **op_kwargs):
    op_kwargs = _drop_torch_random_device_kwarg(op_kwargs)
    return torch.randint_like(op_input, low=low, high=high, dtype=dtype, **op_kwargs)

def mint_nn_elu_ms(op_input, alpha=1.0, inplace=False):
    cloned_input = op_input.clone()
    result = mint.nn.ELU(alpha=alpha, inplace=inplace)(cloned_input)
    return result

def mint_nn_elu_torch(op_input, alpha=1.0, inplace=False):
    cloned_input = op_input.clone()
    result =  torch.nn.ELU(alpha=alpha, inplace=inplace)(cloned_input)
    return result

def mint_nn_hardshrink_ms(op_input, lambd=0.5):
    return mint.nn.Hardshrink(lambd=lambd)(op_input)

def mint_nn_hardshrink_torch(op_input, lambd=0.5):
    return torch.nn.Hardshrink(lambd=lambd)(op_input)

def mint_nn_hardsigmoid_ms(op_input):
    return mint.nn.Hardsigmoid()(op_input)

def mint_nn_hardsigmoid_torch(op_input):
    return torch.nn.Hardsigmoid()(op_input)

def mint_nn_hardswish_ms(op_input):
    return mint.nn.Hardswish()(op_input)

def mint_nn_hardswish_torch(op_input):
    return torch.nn.Hardswish()(op_input)

def mint_nn_logsoftmax_ms(op_input, dim=None):
    return mint.nn.LogSoftmax(dim=dim)(op_input)

def mint_nn_logsoftmax_torch(op_input, dim=None):
    return torch.nn.LogSoftmax(dim=dim)(op_input)

def mint_nn_mish_ms(op_input):
    return mint.nn.Mish()(op_input)

def mint_nn_mish_torch(op_input):
    return torch.nn.Mish()(op_input)

def mint_nn_selu_ms(op_input):
    return mint.nn.SELU()(op_input)

def mint_nn_selu_torch(op_input):
    return torch.nn.SELU()(op_input)

def mint_nn_sigmoid_ms(op_input):
    return mint.nn.Sigmoid()(op_input)

def mint_nn_sigmoid_torch(op_input):
    return torch.nn.Sigmoid()(op_input)

def mint_nn_silu_ms(op_input, inplace=False):
    cloned_input = op_input.clone()
    result = mint.nn.SiLU(inplace=inplace)(cloned_input)
    return result

def mint_nn_silu_torch(op_input, inplace=False):
    cloned_input = op_input.clone()
    result =  torch.nn.SiLU(inplace=inplace)(cloned_input)
    return result

def mint_nn_softmax_ms(op_input, dim=None):
    return mint.nn.Softmax(dim=dim)(op_input)

def mint_nn_softmax_torch(op_input, dim=None):
    return torch.nn.Softmax(dim=dim)(op_input)

def mint_searchsorted_ms(sorted_sequence, values, out_int32=False, right=False, side=None, sorter=None):
    return mint.searchsorted(sorted_sequence=sorted_sequence, values=values, out_int32=out_int32, right=right,
                             side=side, sorter=sorter)

def mint_searchsorted_torch(sorted_sequence, values, out_int32=False, right=False, side=None, sorter=None):
    if side is None:
        side = "left"
    kwargs_p = {
        "sorted_sequence": sorted_sequence,
        "input": values,
        "out_int32": out_int32,
        "right": right,
        "side": side,
    }
    if sorter is not None:
        kwargs_p["sorter"] = sorter
    return torch.searchsorted(**kwargs_p)

def mint_AdaptiveMaxPool1d_ms(input, output_size, return_indices=False):
    net = mint.nn.AdaptiveMaxPool1d(output_size, return_indices=return_indices)
    return net(input)

def mint_AdaptiveMaxPool1d_torch(input, output_size, return_indices=False):
    net = torch.nn.AdaptiveMaxPool1d(output_size, return_indices=return_indices)
    return net(input)

def mint_ReplicationPad1d_ms(input, padding):
    net = mint.nn.ReplicationPad1d(padding)
    return net(input)

def mint_ReplicationPad1d_torch(input, padding):
    net = torch.nn.ReplicationPad1d(padding)
    return net(input)

def mint_ReplicationPad2d_ms(input, padding):
    net = mint.nn.ReplicationPad2d(padding)
    return net(input)

def mint_ReplicationPad2d_torch(input, padding):
    net = torch.nn.ReplicationPad2d(padding)
    return net(input)

def mint_ReplicationPad3d_ms(input, padding):
    net = mint.nn.ReplicationPad3d(padding)
    return net(input)

def mint_ReplicationPad3d_torch(input, padding):
    net = torch.nn.ReplicationPad3d(padding)
    return net(input)

def mint_ReflectionPad1d_ms(input, padding):
    net = mint.nn.ReflectionPad1d(padding)
    return net(input)

def mint_ReflectionPad1d_torch(input, padding):
    net = torch.nn.ReflectionPad1d(padding)
    return net(input)

def mint_ReflectionPad2d_ms(input, padding):
    net = mint.nn.ReflectionPad2d(padding)
    return net(input)

def mint_ReflectionPad2d_torch(input, padding):
    net = torch.nn.ReflectionPad2d(padding)
    return net(input)

def mint_ReflectionPad3d_ms(input, padding):
    net = mint.nn.ReflectionPad3d(padding)
    return net(input)

def mint_ReflectionPad3d_torch(input, padding):
    net = torch.nn.ReflectionPad3d(padding)
    return net(input)

def mint_ZeroPad1d_ms(input, padding):
    net = mint.nn.ZeroPad1d(padding)
    return net(input)

def mint_ZeroPad1d_torch(input, padding):
    net = torch.nn.ZeroPad1d(padding)
    return net(input)

def mint_ZeroPad2d_ms(input, padding):
    net = mint.nn.ZeroPad2d(padding)
    return net(input)

def mint_ZeroPad2d_torch(input, padding):
    net = torch.nn.ZeroPad2d(padding)
    return net(input)

def mint_ZeroPad3d_ms(input, padding):
    net = mint.nn.ZeroPad3d(padding)
    return net(input)

def mint_ZeroPad3d_torch(input, padding):
    net = torch.nn.ZeroPad3d(padding)
    return net(input)

def mint_ConstantPad2d_ms(input, padding, value):
    net = mint.nn.ConstantPad2d(padding, value)
    return net(input)

def mint_ConstantPad2d_torch(input, padding, value):
    net = torch.nn.ConstantPad2d(padding, value)
    return net(input)

def mint_ConstantPad3d_ms(input, padding, value):
    net = mint.nn.ConstantPad3d(padding, value)
    return net(input)

def mint_ConstantPad3d_torch(input, padding, value):
    net = torch.nn.ConstantPad3d(padding, value)
    return net(input)

def mint_MaxUnpool2d_ms(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    net = mint.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return net(input, indices, output_size)

def mint_MaxUnpool2d_torch(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    if output_size is not None and len(input.shape) == 3 and len(output_size) == 3:
        output_size = output_size[-2:]
    net = torch.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return net(input, indices, output_size)

def mint_AvgPool3d_ms(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                      divisor_override=None):
    net = mint.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad, divisor_override=divisor_override)
    return net(input)

def mint_AvgPool3d_torch(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                         divisor_override=None):
    net = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad, divisor_override=divisor_override)
    return net(input)

def mint_AvgPool2d_ms(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                      divisor_override=None):
    net = mint.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad, divisor_override=divisor_override)
    return net(input)

def mint_AvgPool2d_torch(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                         divisor_override=None):
    net = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad, divisor_override=divisor_override)
    return net(input)

# wrapper function for mindspore.mint.nn.Linear
def mint_nn_linear_ms(op_input, in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=None):
    """Wrapper for mindspore.mint.nn.Linear."""
    linear_layer = mint.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        weight_init=weight_init,
        bias_init=bias_init,
        dtype=dtype
    )
    return linear_layer(op_input)

def nn_linear_torch(op_input, in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=None):
    """Wrapper for torch.nn.Linear.
    Note: weight_init, bias_init, and dtype are kept for interface consistency but not used in PyTorch.
    """
    linear_layer = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype
    )
    linear_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        linear_layer.bias = torch.nn.Parameter(bias_init)
    return linear_layer(op_input)

def mint_nn_conv3d_ms(op_input, in_channels, out_channels, kernel_size, stride=1, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None,
                      weight_init=None, bias_init=None):
    """Wrapper for mindspore.mint.nn.Conv3d."""
    conv3d_layer = mint.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    if weight_init is not None:
        conv3d_layer.weight = Parameter(weight_init, name='weight')
    if bias:
        conv3d_layer.bias = Parameter(bias_init, name='bias')
    return conv3d_layer(op_input)

def nn_conv3d_torch(op_input, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                         dtype=None, weight_init=None, bias_init=None):
    """Wrapper for torch.nn.Conv3d."""
    conv3d_layer = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    if weight_init is not None:
        conv3d_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        conv3d_layer.bias = torch.nn.Parameter(bias_init)
    return conv3d_layer(op_input)

def mint_nn_conv_transpose2d_ms(op_input, in_channels, out_channels, kernel_size, stride=1, padding=0,
                      output_padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None,
                      weight_init=None, bias_init=None, output_size=None):
    """Wrapper for mindspore.mint.nn.ConvTranspose2d."""
    conv2d_layer = mint.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    if weight_init is not None:
        conv2d_layer.weight = Parameter(weight_init, name='weight')
    if bias:
        conv2d_layer.bias = Parameter(bias_init, name='bias')
    return conv2d_layer(op_input, output_size=output_size)

def nn_conv_transpose2d_torch(op_input, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, output_padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                         dtype=None, weight_init=None, bias_init=None, output_size=None):
    """Wrapper for torch.nn.ConvTranspose2d."""
    conv2d_layer = torch.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        dtype=dtype
    )
    if weight_init is not None:
        conv2d_layer.weight = torch.nn.Parameter(weight_init)
    if bias:
        conv2d_layer.bias = torch.nn.Parameter(bias_init)
    return conv2d_layer(op_input, output_size=output_size)

def mint_nn_fold(op_input, output_size, kernel_size, dilation, padding, stride):
    net=mint.nn.Fold(output_size, kernel_size, dilation, padding, stride)
    return net(op_input)

def torch_nn_fold(op_input, output_size, kernel_size, dilation, padding, stride):
    net=torch.nn.Fold(output_size, kernel_size, dilation, padding, stride)
    return net(op_input)

def mint_nn_unfold(op_input, kernel_size, dilation, padding, stride):
    net=mint.nn.Unfold(kernel_size, dilation, padding, stride)
    return net(op_input)

def torch_nn_unfold(op_input, kernel_size, dilation, padding, stride):
    net=torch.nn.Unfold(kernel_size, dilation, padding, stride)
    return net(op_input)

def ms_nn_dropout_decorator(op_class):
    def _dropout_func(op_input, p=0.5, training=True, inplace=False):
        net = op_class(p, inplace=inplace)
        if training:
            net.set_train()
            if 0 < p < 1:
                if op_input.dtype.is_floating_point and op_input.dtype != ms.float64 and op_input.size != 0:
                    repeat_count = 400
                    sizes = 0
                    m1 = None
                    input_clone = None
                    for _ in range(repeat_count):
                        input_clone = op_input.clone()
                        m1 = net(input_clone)
                        sizes = sizes + (m1 == 0).sum().item()
                    mask = (m1 == 0).astype(op_input.dtype)
                    modified_input = op_input * (1 - mask)
                    result = (1 / (1 - p) * modified_input - m1,
                              ms.Tensor(sizes / (repeat_count * op_input.size), dtype=op_input.dtype))
                    if inplace:
                        result = result + (m1 - input_clone,)
                    return result
                return dropout_fallback_ms(op_input)
        else:
            net.set_train(False)
        return net(op_input)
    return _dropout_func

def torch_nn_dropout_decorator(op_class):
    def _dropout_func(op_input, p=0.5, training=True, inplace=False):
        net = op_class(p, inplace=inplace)
        if training:
            if 0 < p < 1:
                if op_input.dtype.is_floating_point and op_input.dtype != torch.float64 and op_input.numel() != 0:
                    repeat_count = 400
                    sizes = 0
                    t1 = None
                    input_clone = None
                    for _ in range(repeat_count):
                        input_clone = op_input.clone()
                        t1 = net(input_clone)
                        sizes = sizes + (t1 == 0).sum().item()
                    mask = (t1 == 0).to(op_input.dtype)
                    modified_input = op_input * (1 - mask)
                    result = (1 / (1 - p) * modified_input - t1,
                              torch.tensor(sizes / (repeat_count * op_input.numel()), dtype=op_input.dtype))
                    if inplace:
                        result = result + (t1 - input_clone,)
                    return result
                return dropout_fallback_torch(op_input)
        else:
            net.eval()
        return net(op_input)
    return _dropout_func

def ms_nn_pixel_shuffle(input, upscale_factor):
    net = mint.nn.PixelShuffle(upscale_factor)
    return net(input)

def torch_nn_pixel_shuffle(input, upscale_factor):
    net = torch.nn.PixelShuffle(upscale_factor)
    return net(input)

def ms_nn_up_sample(input, size=None, scale_factor=None, mode="nearest",
                    align_corners=None, recompute_scale_factor=None):
    net = mint.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode,
                           align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    return net(input)

def torch_nn_up_sample(input, size=None, scale_factor=None, mode="nearest",
                    align_corners=None, recompute_scale_factor=None):
    net = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode,
                           align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    return net(input)

def ms_nn_bce_loss(input, target, weight=None, reduction=None):
    if weight is not None:
        loss = mint.nn.BCELoss(weight=weight, reduction=reduction)
    else:
        loss = mint.nn.BCELoss(reduction=reduction)
    output = loss(input, target)
    return output

def torch_nn_bce_loss(input, target, weight=None, reduction=None):
    if weight is not None:
        loss = torch.nn.BCELoss(weight=weight, reduction=reduction)
    else:
        loss = torch.nn.BCELoss(reduction=reduction)
    output = loss(input, target)
    return output

def mint_nn_softshrink_ms(op_input, lambd=0.5):
    softshrink = mint.nn.Softshrink(lambd=lambd)
    return softshrink(op_input)

def mint_nn_softshrink_torch(op_input, lambd=0.5):
    softshrink = torch.nn.Softshrink(lambd=lambd)
    return softshrink(op_input)

def mint_nn_threshold_ms(op_input, threshold, value, inplace=False):
    threshold = mint.nn.Threshold(threshold=threshold, value=value, inplace=inplace)
    return threshold(op_input.clone())

def mint_nn_threshold_torch(op_input, threshold, value, inplace=False):
    threshold = torch.nn.Threshold(threshold=threshold, value=value, inplace=inplace)
    return threshold(op_input.clone())

def mint_nn_cosineembeddingloss_ms(op_input, op_target, target, margin=0.0, reduction='mean'):
    cosineembeddingloss = mint.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    return cosineembeddingloss(op_input, op_target, target)

def mint_nn_cosineembeddingloss_torch(op_input, op_target, target, margin=0.0, reduction='mean'):
    cosineembeddingloss = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    return cosineembeddingloss(op_input, op_target, target)

def mint_nn_crossentropyloss_ms(op_input, op_target, weight=None, ignore_index=-100,
                                reduction='mean', label_smoothing=0.0):
    crossentropyloss = mint.nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing
    )
    return crossentropyloss(op_input, op_target)

def mint_nn_crossentropyloss_torch(op_input, op_target, weight=None, ignore_index=-100,
                                reduction='mean', label_smoothing=0.0):
    crossentropyloss = torch.nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing
    )
    return crossentropyloss(op_input, op_target)

def mint_nn_kldivloss_ms(op_input, op_target, reduction='mean', log_target=False):
    kldivloss = mint.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    return kldivloss(op_input, op_target)

def mint_nn_kldivloss_torch(op_input, op_target, reduction='mean', log_target=False):
    kldivloss = torch.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    return kldivloss(op_input, op_target)

def mint_nn_l1loss_ms(op_input, op_target, reduction='mean'):
    l1loss = mint.nn.L1Loss(reduction=reduction)
    return l1loss(op_input, op_target)

def mint_nn_l1loss_torch(op_input, op_target, reduction='mean'):
    l1loss = torch.nn.L1Loss(reduction=reduction)
    return l1loss(op_input, op_target)

def mint_nn_mseloss_ms(op_input, op_target, reduction='mean'):
    mseloss = mint.nn.MSELoss(reduction=reduction)
    return mseloss(op_input, op_target)

def mint_nn_mseloss_torch(op_input, op_target, reduction='mean'):
    mseloss = torch.nn.MSELoss(reduction=reduction)
    return mseloss(op_input, op_target)

def mint_nn_nllloss_ms(op_input, op_target, weight=None, ignore_index=-100, reduction='mean'):
    nllloss = mint.nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    return nllloss(op_input, op_target)

def mint_nn_nllloss_torch(op_input, op_target, weight=None, ignore_index=-100, reduction='mean'):
    nllloss = torch.nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    return nllloss(op_input, op_target)

def mint_nn_smoothl1loss_ms(op_input, op_target, reduction='mean', beta=1.0):
    smoothl1loss = mint.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    return smoothl1loss(op_input, op_target)

def mint_nn_smoothl1loss_torch(op_input, op_target, reduction='mean', beta=1.0):
    smoothl1loss = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    return smoothl1loss(op_input, op_target)

def mint_nn_functional_elu_ms(op_input, alpha=1.0, inplace=False):
    cloned_input = op_input.clone()
    result = mint.nn.functional.elu(cloned_input, alpha=alpha, inplace=inplace)
    return result

def mint_nn_functional_elu_torch(op_input, alpha=1.0, inplace=False):
    cloned_input = op_input.clone()
    result = torch.nn.functional.elu(cloned_input, alpha=alpha, inplace=inplace)
    return result

def mint_nn_functional_elu__ms(op_input, alpha=1.0):
    cloned_input = op_input.clone()
    result = mint.nn.functional.elu_(cloned_input, alpha=alpha)
    return result

def mint_nn_functional_elu__torch(op_input, alpha=1.0):
    cloned_input = op_input.clone()
    result = torch.nn.functional.elu_(cloned_input, alpha=alpha)
    return result

def mint_nn_functional_hardshrink_ms(op_input, lambd=0.5):
    cloned_input = op_input.clone()
    result = mint.nn.functional.hardshrink(cloned_input, lambd=lambd)
    return result

def mint_nn_functional_hardshrink_torch(op_input, lambd=0.5):
    cloned_input = op_input.clone()
    result = torch.nn.functional.hardshrink(cloned_input, lambd=lambd)
    return result

def mint_nn_functional_mish_ms(op_input):
    return mint.nn.functional.mish(op_input)

def mint_nn_functional_mish_torch(op_input):
    return torch.nn.functional.mish(op_input)

def mint_nn_functional_sigmoid_ms(op_input):
    return mint.nn.functional.sigmoid(op_input)

def mint_nn_functional_sigmoid_torch(op_input):
    return torch.nn.functional.sigmoid(op_input)

def mint_nn_functional_silu_ms(op_input, inplace=False):
    cloned_input = op_input.clone()
    result = mint.nn.functional.silu(cloned_input, inplace=inplace)
    return result

def mint_nn_functional_silu_torch(op_input, inplace=False):
    cloned_input = op_input.clone()
    result =  torch.nn.functional.silu(cloned_input, inplace=inplace)
    return result

def mint_nn_functional_group_norm_ms(op_input, num_groups, weight=None, bias=None, eps=1e-5):
    result = mint.nn.functional.group_norm(op_input, num_groups, weight=weight, bias=bias, eps=eps)
    return result

def mint_nn_functional_group_norm_torch(op_input, num_groups, weight=None, bias=None, eps=1e-5):
    result = torch.nn.functional.group_norm(op_input, num_groups, weight=weight, bias=bias, eps=eps)
    return result

def mint_nn_functional_normalize_ms(op_input, p=2, dim=1, eps=1e-12,):
    result = mint.nn.functional.normalize(op_input, p=p, dim=dim, eps=eps)
    return result

def mint_nn_functional_normalize_torch(op_input, p=2, dim=1, eps=1e-12,):
    result = torch.nn.functional.normalize(op_input, p=p, dim=dim, eps=eps)
    return result

def mint_nn_functional_prelu_ms(op_input, weight):
    return mint.nn.functional.prelu(op_input, weight=weight)

def mint_nn_functional_prelu_torch(op_input, weight):
    return torch.nn.functional.prelu(op_input, weight=weight)

def mint_nn_functional_relu__ms(op_input):
    cloned_input = op_input.clone()
    return mint.nn.functional.relu_(cloned_input)

def mint_nn_functional_relu__torch(op_input):
    cloned_input = op_input.clone()
    return torch.nn.functional.relu_(cloned_input)

def mint_nn_functional_softplus_ms(op_input, beta=1, threshold=20):
    return mint.nn.functional.softplus(op_input, beta=beta, threshold=threshold)

def mint_nn_functional_softplus_torch(op_input, beta=1, threshold=20):
    return torch.nn.functional.softplus(op_input, beta=beta, threshold=threshold)

def mint_nn_functional_softshrink_ms(op_input, lambd=0.5):
    return mint.nn.functional.softshrink(op_input, lambd=lambd)

def mint_nn_functional_softshrink_torch(op_input, lambd=0.5):
    return torch.nn.functional.softshrink(op_input, lambd=lambd)

def mint_nn_bceloss_ms(op_input, target, weight=None, reduction='mean'):
    loss = mint.nn.BCELoss(weight=weight, reduction=reduction)
    result = loss(op_input, target)
    return result

def mint_nn_bceloss_torch(op_input, target, weight=None, reduction='mean'):
    loss = torch.nn.BCELoss(weight=weight, reduction=reduction)
    result = loss(op_input, target)
    return result

def mint_MaxPool2d_ms(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    net = mint.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                            return_indices=return_indices, ceil_mode=ceil_mode)
    return net(input)

def mint_MaxPool2d_torch(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
                         return_indices=False):
    net = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                             return_indices=return_indices, ceil_mode=ceil_mode)
    return net(input)


def _cross_entropy_cast_torch_target(target):
    if torch.is_floating_point(target):
        return target
    return target.long()


def mint_nn_functional_cross_entropy_ms(input, target, weight=None, ignore_index=-100,
                                        reduction='mean', label_smoothing=0.0):
    return mint.nn.functional.cross_entropy(input, target, weight, ignore_index, reduction, label_smoothing)


def mint_nn_functional_cross_entropy_torch(input, target, weight=None, ignore_index=-100,
                                           reduction='mean', label_smoothing=0.0):
    target = _cross_entropy_cast_torch_target(target)
    return torch_F.cross_entropy(
        input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def mint_BCEWithLogitsLoss_ms(input, target, weight=None, reduction='mean', pos_weight=None):
    return mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight)


def mint_BCEWithLogitsLoss_torch(input, target, weight=None, reduction='mean', pos_weight=None):
    return torch_F.binary_cross_entropy_with_logits(input, target, weight=weight,
                                                    reduction=reduction, pos_weight=pos_weight)


def mint_BCEWithLogitsLoss_ms_grad(input, target, *args):
    weight = None
    reduction = 'mean'
    pos_weight = None
    for arg in args:
        if isinstance(arg, str):
            reduction = arg
        elif weight is None:
            weight = arg
        elif pos_weight is None:
            pos_weight = arg
        else:
            raise TypeError(f"Unexpected extra argument for mint_BCEWithLogitsLoss_ms_grad: {arg}")
    return mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight)


def mint_binary_cross_entropy_with_logits_ms(input, target, weight=None, reduction='mean', pos_weight=None):
    return mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight)

def mint_nn_embedding_ms(op_input, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                         scale_grad_by_freq=True, sparse=False, _weight=None, _freeze=False, dtype=None):
    embedding = mint.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx,
                                  max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse, _weight=_weight, _freeze=_freeze, dtype=dtype)
    return embedding(op_input)

def mint_nn_embedding_torch(op_input, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                         scale_grad_by_freq=True, sparse=False, _weight=None, _freeze=False, dtype=None):
    embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx,
                                  max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse, _weight=_weight, _freeze=_freeze, dtype=dtype)
    return embedding(op_input)

def mint_nn_functional_threshold_ms(op_input, threshold, value, inplace=False):
    return mint.nn.functional.threshold(op_input.clone(), threshold=threshold, value=value, inplace=inplace)

def mint_nn_functional_threshold_torch(op_input, threshold, value, inplace=False):
    return torch.nn.functional.threshold(op_input.clone(), threshold=threshold, value=value, inplace=inplace)

def mint_nn_functional_threshold__ms(op_input, threshold, value):
    return mint.nn.functional.threshold_(op_input.clone(), threshold=threshold, value=value)

def mint_nn_functional_threshold__torch(op_input, threshold, value):
    return torch.nn.functional.threshold_(op_input.clone(), threshold=threshold, value=value)

def ms_optim_sgd(test_input, weight, bias, train_input , target,
                 lr, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False):
    out_features, in_features = weight.shape
    net = mint.nn.Linear(in_features, out_features)
    net.weight = Parameter(weight)
    net.bias = Parameter(bias)
    loss_fn = mint.nn.MSELoss()
    optimizer = mint.optim.SGD(net.trainable_params(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay, dampening=dampening, nesterov=nesterov, maximize=maximize)
    def _forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    grad_fn = ms.value_and_grad(_forward_fn, None, optimizer.parameters, has_aux=True)
    for epoch in range(100):
        (loss, _), grads = grad_fn(train_input, target)
        optimizer(grads)
    return net(test_input)

def torch_optim_sgd(test_input, weight, bias, train_input, target,
                 lr, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False):
    out_features, in_features = weight.shape
    net = torch.nn.Linear(in_features, out_features)
    net.weight = torch.nn.Parameter(weight)
    net.bias = torch.nn.Parameter(bias)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay, dampening=dampening,nesterov=nesterov, maximize=maximize)
    for i in range(100):
        optimizer.zero_grad()
        output = net(train_input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    return net(test_input)

class Net(nn.Cell):
    def __init__(self, in_channel, ):
        super().__init__()
        self.conv = mint.nn.Conv2d(in_channel, 10, (2, 2),  bias=False)
        self.optimizer = None
    def construct(self, x):
        return ms.value_and_grad(self.conv, None, self.conv.weight)(x)[1]

def mint_optim_Adam_ms(input, net, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                       maximize=False,):

    net.optimizer(net.construct(input)) if net.optimizer is not None else mint.optim.Adam(net.trainable_params(),lr=lr,
            betas=betas, eps=eps,weight_decay=weight_decay,amsgrad=amsgrad, maximize=maximize)(net.construct(input))

    return  net.conv.weight.data


def mint_optim_Adam_torch(input, net, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                          maximize=False,):

    net_torch = torch.nn.Conv2d(input.shape[-3], 10, (2, 2), bias=False)

    #torch.nn.Conv2d not support float16,so convert to float32
    net_torch.weight.data = torch.from_numpy(net.conv.weight.data.to(ms.float32).asnumpy()).to(torch.bfloat16) if (net
                .conv.weight.dtype== ms.bfloat16) else torch.from_numpy(net.conv.weight.data.to(ms.float32).asnumpy())

    optimizer = torch.optim.Adam(net_torch.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad, maximize=maximize)
    loss=net_torch(input)

    loss.backward(torch.ones_like(loss))

    optimizer.step()

    return net_torch.weight.data

def mint_optim_AdamW_ms(input,  weight, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                        maximize=False,):

    net = mint.nn.Conv2d(input.shape[-3], 10, (2, 2),  bias=False)

    net.weight = ms.Parameter(weight)

    optimizer = mint.optim.AdamW(net.trainable_params(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad, maximize=maximize)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    for i in range(6):  #epoch=6
        optimizer((ms.value_and_grad(net, None, net.trainable_params())(input)[1]))
        scheduler.step()

    return net.weight.value()

def mint_optim_AdamW_torch(input, weight, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                           maximize=False,):

    net = torch.nn.Conv2d(input.shape[-3], 10, (2, 2), bias=False)


    net.weight = torch.nn.Parameter(weight)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad, maximize=maximize)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    for i in range(6):
        optimizer.zero_grad()
        loss = net(input)
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        scheduler.step()

    return net.weight.data

