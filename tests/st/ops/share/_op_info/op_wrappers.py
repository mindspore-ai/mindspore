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
from mindspore.common.parameter import Parameter
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

def tensor_nn_functional_dropout_ms(op_input, p=0.5, training=False):
    input_clone = op_input.clone()
    if  0 < p < 1:
        if training:
            if op_input.dtype.is_floating_point and op_input.dtype != ms.float64:
                sizes = 0
                m1 = None
                for _ in  range(100):
                    m1 = mint.nn.functional.dropout(op_input, p=p, training=training)
                    sizes = sizes + (m1 == 0).sum().item()
                input_clone[m1 == 0] = 0
                return (1 / (1-p) * input_clone -m1, ms.Tensor(sizes / (100 * op_input.size), dtype=op_input.dtype))
            #torch backward must hava revalance with input tensor, so return value is related to input
            return op_input*0
    return  mint.nn.functional.dropout(op_input, p=p, training=training)

def tensor_nn_functional_dropout_torch(op_input, p=0.5, training=False):
    if  0 < p < 1:
        if training:
            if op_input.dtype.is_floating_point and op_input.dtype != torch.float64:
                sizes = 0
                t1 = None
                input_clone = op_input.clone()
                for _ in range(100):
                    t1 = torch.nn.functional.dropout(op_input, p=p, training=training)
                    sizes = sizes + (t1 == 0).sum().item()
                input_clone[t1 == 0] = 0
                return (1 / (1 - p) * input_clone - t1, torch.tensor(sizes / (100 * op_input.numel()),
                                                                     dtype=op_input.dtype))
            return op_input*0
    return torch.nn.functional.dropout(op_input, p=p, training=training)

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
    """Wrapper for mindspore.mint.nn.Conv1d."""
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

def mint_rand_torch(op_input, generator=None, dtype=None, device=None):
    return torch.rand(op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_randn_ms(op_input, generator=None, dtype=None, device=None):
    return mint.randn(op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_randn_torch(op_input, generator=None, dtype=None, device=None):
    return torch.randn(op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_randint_ms(op_input, low, high, generator=None, dtype=None, device=None):
    return mint.randint(low, high, op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_randint_torch(op_input, low, high, generator=None, dtype=None, device=None):
    return torch.randint(low, high, op_input.shape, generator=generator, dtype=dtype, device=device)

def mint_rand_like_ms(op_input, dtype=None, **op_kwargs):
    return mint.rand_like(op_input, dtype=dtype, **op_kwargs)

def mint_randn_like_ms(op_input, dtype=None, **op_kwargs):
    return mint.randn_like(op_input, dtype=dtype, **op_kwargs)

def mint_randint_like_ms(op_input, low, high, dtype=None, **op_kwargs):
    return mint.randint_like(op_input, low=low, high=high, dtype=dtype, **op_kwargs)

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
