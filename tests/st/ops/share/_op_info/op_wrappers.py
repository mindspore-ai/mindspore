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
"""API wrapper functions for operation testing.

This module provides wrapper functions that adapt MindSpore and PyTorch
tensor/nn methods to a uniform call signature used by the OpInfo framework.
"""
import mindspore as ms
from mindspore import mint
import torch

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
    result = cloned_input.add_(other, alpha=alpha)
    return result

def tensor_add__torch(op_input, other, alpha=1):
    """Wrapper for torch.Tensor.add_ (in-place operation)."""
    cloned_input_pt = op_input.clone()
    result = cloned_input_pt.add_(other, alpha=alpha)
    return result

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
    return torch.nn.functional.gelu(op_input, approximate=approximate)

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

def tensor_unsqueeze_ms(op_input, *args, **kwargs):
    return op_input.unsqueeze(*args, **kwargs)

def tensor_unsqueeze_torch(op_input, *args, **kwargs):
    return op_input.unsqueeze(*args, **kwargs)

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
def tensor_add_ms(op_input, args):
    return op_input.add(args)

def tensor_add_torch(op_input, args):
    return op_input.add(args)

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
def tensor_split_ms(op_input, args):
    return op_input.split(args)

def tensor_split_torch(op_input, args):
    return op_input.split(args)

# wrap tensor method for index_select
def tensor_index_select_ms(op_input, axis, index):
    return op_input.index_select(axis, index)

def tensor_index_select_torch(op_input, axis, index):
    return op_input.index_select(axis, index)

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
