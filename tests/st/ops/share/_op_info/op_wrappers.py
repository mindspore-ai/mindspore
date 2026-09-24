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
import numpy as np
import mindspore as ms
import numpy as np
from mindspore import nn, mint
from mindspore.scipy.linalg import lstsq
from mindspore.common.parameter import Parameter
from mindspore.experimental import optim
import torch
from torch.nn import functional as torch_F


# =====================================wrap tensor/nn method====================================

# wrap tensor method for astype
def tensor_astype_ms(op_input, dtype, copy=True):
    return op_input.astype(dtype=dtype, copy=copy)

def tensor_astype_torch(op_input, dtype, copy=True):
    if isinstance(dtype, str):
        dtype = torch.float32
    return op_input.to(dtype=dtype, copy=copy)

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

def tensor_frac_ms(op_input):
    return op_input.frac()

def tensor_frac_torch(op_input):
    return op_input.frac()

# wrap tensor method for contiguous
def tensor_contiguous_ms(op_input):
    return op_input.contiguous()

def tensor_contiguous_torch(op_input):
    return op_input.contiguous()

def tensor_imag_ms(op_input):
    return op_input.imag()

def tensor_imag_torch(op_input):
    return op_input.imag

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

def tensor_expm1_ms(op_input):
    return op_input.expm1()

def tensor_expm1_torch(op_input):
    return op_input.expm1()

def tensor_fliplr_ms(op_input):
    return op_input.fliplr()

def tensor_fliplr_torch(op_input):
    return op_input.fliplr()

def tensor_flipud_ms(op_input):
    return op_input.flipud()

def tensor_flipud_torch(op_input):
    return op_input.flipud()

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

def mint_div_ms(op_input, other, rounding_mode=None):
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

def mint_narrow_ms(op_input, dim=0, start=0, length=0):
    return mint.narrow(op_input, dim=dim, start=start, length=length)

def mint_narrow_torch(op_input, dim=0, start=0, length=0):
    return torch.narrow(op_input, dim=dim, start=start, length=length)

def tensor_narrow_ms(op_input, dim=0, start=0, length=0):
    return op_input.narrow(dim=dim, start=start, length=length)

def tensor_narrow_torch(op_input, dim=0, start=0, length=0):
    return op_input.narrow(dim=dim, start=start, length=length)

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
    return op_input.add(other, alpha=alpha)

def tensor_add_torch(op_input, other, alpha=1):
    return op_input.add(other, alpha=alpha)

# wrap tensor method for sub
def tensor_sub_ms(op_input, args):
    return op_input.sub(args)

def tensor_sub_torch(op_input, args):
    return op_input.sub(args)

def tensor_inner_ms(op_input, other):
    return op_input.inner(other)

def tensor_inner_torch(op_input, other):
    return op_input.inner(other)

def tensor_hypot_ms(op_input, other):
    return op_input.hypot(other)

def tensor_hypot_torch(op_input, other):
    return op_input.hypot(other)

def tensor_igamma_ms(op_input, other):
    return op_input.igamma(other)

def tensor_igamma_torch(op_input, other):
    return op_input.igamma(other)

def tensor_igammac_ms(op_input, other):
    return op_input.igammac(other)

def tensor_igammac_torch(op_input, other):
    return op_input.igammac(other)

def tensor_logaddexp_ms(op_input, other):
    return op_input.logaddexp(other)

def tensor_logaddexp_torch(op_input, other):
    return op_input.logaddexp(other)

def tensor_histc_ms(op_input, bins=100, min=0, max=0):
    return op_input.histc(bins=bins, min=min, max=max)

def tensor_histc_torch(op_input, bins=100, min=0, max=0):
    return op_input.histc(bins=bins, min=min, max=max)

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

def tensor_adjoint_ms(op_input):
    return op_input.adjoint()

def tensor_adjoint_torch(op_input):
    return op_input.adjoint()

def tensor_addbmm_ms(op_input, batch1, batch2, beta=1, alpha=1):
    return op_input.addbmm(batch1, batch2, beta=beta, alpha=alpha)

def tensor_addbmm_torch(op_input, batch1, batch2, beta=1, alpha=1):
    return op_input.addbmm(batch1, batch2, beta=beta, alpha=alpha)

def tensor_addmv_ms(op_input, vec1, vec2, beta=1, alpha=1):
    return op_input.addmv(vec1, vec2, beta=beta, alpha=alpha)

def tensor_addmv_torch(op_input, vec1, vec2, beta=1, alpha=1):
    return op_input.addmv(vec1, vec2, beta=beta, alpha=alpha)

def tensor_addr_ms(op_input, vec1, vec2, beta=1, alpha=1):
    return op_input.addr(vec1, vec2, beta=beta, alpha=alpha)

def tensor_addr_torch(op_input, vec1, vec2, beta=1, alpha=1):
    return op_input.addr(vec1, vec2, beta=beta, alpha=alpha)

def tensor_addcdiv_ms(op_input, tensor1, tensor2, value=1):
    return op_input.addcdiv(tensor1, tensor2, value=value)

def tensor_addcdiv_torch(op_input, tensor1, tensor2, value=1):
    return op_input.addcdiv(tensor1, tensor2, value=value)

def tensor_addcmul_ms(op_input, tensor1, tensor2, value=1):
    return op_input.addcmul(tensor1, tensor2, value=value)

def tensor_addcmul_torch(op_input, tensor1, tensor2, value=1):
    return op_input.addcmul(tensor1, tensor2, value=value)

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

def tensor_hsplit_ms(op_input, indices_or_sections):
    return op_input.hsplit(indices_or_sections)

def tensor_hsplit_torch(op_input, indices_or_sections):
    return op_input.hsplit(indices_or_sections)

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

def tensor_gather_nd_ms(op_input, indices):
    return op_input.gather_nd(indices)

def tensor_gather_elements_ms(op_input, dim, index):
    return op_input.gather_elements(dim, index)

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
def tensor_mint_nansum(op_input, dim=None, keepdim=False, dtype=None):
    return mint.nansum(op_input, dim, keepdim, dtype=dtype)

#  nansum has keyword parameters dtype,so for matching cases make this fouctions
def tensor_torch_nansum(op_input, dim=None, keepdim=False, dtype=None):
    return torch.nansum(op_input, dim, keepdim, dtype=dtype)

def tensor_nansum_ms(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.nansum(dim, keepdim, dtype=dtype)

def tensor_nansum_torch(op_input, dim=None, keepdim=False, dtype=None):
    return op_input.nansum(dim, keepdim, dtype=dtype)

def tensor_mint_where(condition, op_input, other):
    return mint.where(condition=condition, input=op_input, other=other)

def tensor_torch_where(condition, op_input, other):
    return torch.where(condition=condition, input=op_input, other=other)

def tensor_mint_index_add(op_input, dim, index, source, alpha=1):
    return mint.index_add(op_input, dim, index, source, alpha=alpha)

def tensor_torch_index_add(op_input, dim, index, source, alpha=1):
    return torch.index_add(op_input, dim, index, source, alpha=alpha)

def tensor_mint_reshape(op_input, shape):
    return mint.reshape(op_input, shape)

def tensor_torch_reshape(op_input, shape):
    return torch.reshape(op_input, shape)

def tensor_mint_squeeze(op_input, dim):
    return mint.squeeze(op_input, dim)

def tensor_torch_squeeze(op_input, dim):
    return torch.squeeze(op_input, dim)

def tensor_mint_diag(op_input, diagonal=0):
    return mint.diag(op_input, diagonal=diagonal)

def tensor_torch_diag(op_input, diagonal=0):
    return torch.diag(op_input, diagonal=diagonal)

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

def mint_baddbmm_ms(input, batch1, batch2, beta=1, alpha=1):
    return mint.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)

def mint_addmv_ms(input, mat1, mat2, beta=1, alpha=1):
    return mint.addmv(input, mat1, mat2, beta=beta, alpha=alpha)

def mint_addmv_torch(input, mat1, mat2, beta=1, alpha=1):
    return torch.addmv(input, mat1, mat2, beta=beta, alpha=alpha)

def mint_addmm_ms(input, mat1, mat2, beta=1, alpha=1):
    return mint.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def mint_addmm_torch(input, mat1, mat2, beta=1, alpha=1):
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def tensor_addmm_ms(op_input, mat1, mat2, beta=1, alpha=1):
    return op_input.addmm(mat1, mat2, beta=beta, alpha=alpha)

def tensor_addmm_torch(op_input, mat1, mat2, beta=1, alpha=1):
    return op_input.addmm(mat1, mat2, beta=beta, alpha=alpha)

def tensor_addmm__ms(op_input, mat1, mat2, beta=1, alpha=1):
    cloned_input = op_input.clone()
    cloned_input.addmm_(mat1, mat2, beta=beta, alpha=alpha)
    return cloned_input

def tensor_addmm__torch(op_input, mat1, mat2, beta=1, alpha=1):
    cloned_input_pt = op_input.clone()
    cloned_input_pt.addmm_(mat1, mat2, beta=beta, alpha=alpha)
    return cloned_input_pt

def tensor_exponential__ms(op_input, lambd=1, generator=None):
    cloned_input = op_input.clone()
    cloned_input.exponential_(lambd=lambd, generator=generator)
    return cloned_input

def tensor_exponential__torch(op_input, lambd=1, generator=None):
    cloned_input_pt = op_input.clone()
    if generator is not None and not isinstance(generator, torch.Generator):
        import mindspore as ms
        if isinstance(generator, ms.Generator):
            torch_gen = torch.Generator()
            torch_gen.manual_seed(42)
            generator = torch_gen
    cloned_input_pt.exponential_(lambd=lambd, generator=generator)
    return cloned_input_pt

def tensor_index_put_ms(op_input, indices, values, accumulate=False):
    return op_input.index_put(indices, values, accumulate=accumulate)

def tensor_index_put_torch(op_input, indices, values, accumulate=False):
    return op_input.index_put(indices, values, accumulate=accumulate)

def tensor_index_put__ms(op_input, indices, values, accumulate=False):
    cloned_input = op_input.clone()
    cloned_input.index_put_(indices, values, accumulate=accumulate)
    return cloned_input

def tensor_index_put__torch(op_input, indices, values, accumulate=False):
    cloned_input_pt = op_input.clone()
    cloned_input_pt.index_put_(indices, values, accumulate=accumulate)
    return cloned_input_pt

def tensor_index_copy__ms(op_input, dim, index, tensor):
    cloned_input = op_input.clone()
    cloned_input.index_copy_(dim, index, tensor)
    return cloned_input

def tensor_index_copy__torch(op_input, dim, index, tensor):
    cloned_input_pt = op_input.clone()
    cloned_input_pt.index_copy_(dim, index, tensor)
    return cloned_input_pt

def tensor_index_fill__ms(op_input, dim, index, value):
    cloned_input = op_input.clone()
    cloned_input.index_fill_(dim, index, value)
    return cloned_input

def tensor_index_fill__torch(op_input, dim, index, value):
    cloned_input_pt = op_input.clone()
    if index.dtype != torch.int64:
        index = index.to(torch.int64)
    cloned_input_pt.index_fill_(dim, index, value)
    return cloned_input_pt

def tensor_index_fill_ms(op_input, dim, index, value):
    return op_input.index_fill(dim, index, value)

def tensor_index_fill_torch(op_input, dim, index, value):
    if index.dtype != torch.int64:
        index = index.to(torch.int64)
    return op_input.index_fill(dim, index, value)

def tensor_index_add_ms(op_input, dim, index, source, alpha=1):
    return op_input.index_add(dim, index, source, alpha=alpha)

def tensor_index_add_torch(op_input, dim, index, source, alpha=1):
    return op_input.index_add(dim, index, source, alpha=alpha)

def tensor_index_add__ms(op_input, dim, index, source, alpha=1):
    cloned_input = op_input.clone()
    cloned_input.index_add_(dim, index, source, alpha=alpha)
    return cloned_input

def tensor_index_add__torch(op_input, dim, index, source, alpha=1):
    cloned_input_pt = op_input.clone()
    cloned_input_pt.index_add_(dim, index, source, alpha=alpha)
    return cloned_input_pt

def tensor_fill__ms(op_input, value):
    cloned_input = op_input.clone()
    cloned_input.fill_(value)
    return cloned_input

def tensor_fill__torch(op_input, value):
    cloned_input_pt = op_input.clone()
    #only support scalar
    if isinstance(value, torch.Tensor):
        if value.dim() == 1:
            value = value.item()
        elif value.dim() == 0:
            value = value.item()
    cloned_input_pt.fill_(value)
    return cloned_input_pt

def tensor_fill_diagonal__ms(op_input, fill_value, wrap=False):
    cloned_input = op_input.clone()
    cloned_input.fill_diagonal_(fill_value, wrap=wrap)
    return cloned_input

def tensor_fill_diagonal__torch(op_input, fill_value, wrap=False):
    cloned_input_pt = op_input.clone()
    cloned_input_pt.fill_diagonal_(fill_value, wrap=wrap)
    return cloned_input_pt

def tensor_clamp__ms(op_input, min=None, max=None):
    cloned_input = op_input.clone()
    cloned_input.clamp_(min=min, max=max)
    return cloned_input

def tensor_clamp__torch(op_input, min=None, max=None):
    cloned_input = op_input.clone()
    cloned_input.clamp_(min=min, max=max)
    return cloned_input

def tensor_choose_ms(op_input, choices, mode="clip"):
    cloned_input = op_input.clone()
    return cloned_input.choose(choices, mode=mode)

def tensor_expand_dims_ms(op_input, axis=0):
    cloned_input = op_input.clone()
    return cloned_input.expand_dims(axis=axis)

def tensor_fill_diagonal_ms(op_input, fill_value, wrap=False):
    cloned_input = op_input.clone()
    return cloned_input.fill_diagonal(fill_value, wrap=wrap)

def tensor_cholesky_solve_ms(op_input, input2, upper=False):
    return op_input.cholesky_solve(input2, upper=upper)

def tensor_cholesky_solve_torch(op_input, input2, upper=False):
    return op_input.cholesky_solve(input2, upper=upper)

def tensor_cholesky_inverse_ms(op_input, upper=False):
    cloned_input = op_input.clone()
    return cloned_input.cholesky_inverse(upper=upper)

def tensor_cholesky_inverse_torch(op_input, upper=False):
    cloned_input_pt = op_input.clone()
    return cloned_input_pt.cholesky_inverse(upper=upper)

def tensor_chunk_ms(op_input, chunks, dim=0):
    cloned_input = op_input.clone()
    return cloned_input.chunk(chunks, dim=dim)

def tensor_chunk_torch(op_input, chunks, dim=0):
    cloned_input_pt = op_input.clone()
    return cloned_input_pt.chunk(chunks, dim=dim)

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

def tensor_fold_ms(op_input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return op_input.fold(output_size=output_size, kernel_size=kernel_size, dilation=dilation,
                         padding=padding, stride=stride)

# def tensor_fold_torch(op_input, output_size, kernel_size, dilation=1, padding=0, stride=1):
#     return op_input.fold(output_size=output_size, kernel_size=kernel_size, dilation=dilation,
#                          padding=padding, stride=stride)

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
    if isinstance(scale_factor, list):
        scale_factor = tuple(scale_factor)
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

    #  only error_sample_inputs net.optimizer is None
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

def mint_eye_ms(op_input, m=None, dtype=None):
    if op_input is None:
        n = -1
    elif not isinstance(op_input, ms.Tensor):
        n = 1.0
    elif len(op_input.shape) == 0:
        n = 0
    else:
        n = op_input.shape[0]
    return mint.eye(n, m=m, dtype=dtype)

def mint_eye_torch(op_input, m=None, dtype=None):
    if len(op_input.shape) == 0:
        n = 0
    else:
        n = op_input.shape[0]
    if m is None:
        m = n
    # for torch 2.1.0 does not implemented for bfloat16
    if dtype is torch.bfloat16:
        dtype = torch.float16
    return torch.eye(n, m=m, dtype=dtype)

def mint_full_ms(op_input, fill_value, dtype=None):
    if op_input is None:
        size = [1, -1]
    elif not isinstance(op_input, ms.Tensor):
        if op_input == 2.2:
            # size is not tuple[int] / list[int]
            size = (2.2, 3)
        else:
            # size is not list or tuple
            size = 1
    else:
        size = op_input.shape
    return mint.full(size, fill_value=fill_value, dtype=dtype)

def mint_full_torch(op_input, fill_value, dtype=None):
    size = op_input.shape
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.float().cpu().detach().numpy().item() \
            if dtype == torch.bfloat16 else fill_value.cpu().detach().numpy().item()
    return torch.full(size, fill_value=fill_value, dtype=dtype)

def mint_full_like_torch(op_input, fill_value, dtype=None):
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.float().cpu().detach().numpy().item() \
            if dtype == torch.bfloat16 else fill_value.cpu().detach().numpy().item()
    return torch.full_like(op_input, fill_value=fill_value, dtype=dtype)

def mint_linspace_ms(op_input, start, end, dtype=None):
    if op_input is None:
        steps = -1
    elif not isinstance(op_input, ms.Tensor):
        if op_input == 1.0:
            # steps is float
            steps = 1.0
        else:
            # steps is 0
            steps = 0
    elif len(op_input.shape) == 0:
        steps = 1
    else:
        steps = op_input.shape[0]
    return mint.linspace(start, end, steps, dtype=dtype)

def mint_linspace_torch(op_input, start, end, dtype=None):
    if len(op_input.shape) == 0:
        steps = 1
    else:
        steps = op_input.shape[0]
    return torch.linspace(start, end, steps, dtype=dtype)

def mint_stack_func_with_flatten_inputs(*args):
    """A kwargs-free wrapper for mint.stack used by ms.grad in tests.

    Note:
        OpSampleInput.convert_to_args() expands op_input lists/tuples into multiple
        positional arguments. For mint.stack, the first argument should be a single
        sequence of tensors, so this wrapper repacks positional tensor args into
        a list and treats an optional trailing int as dim.
    """
    if not args:
        raise ValueError("mint_stack_func_with_flatten_inputs expects at least one argument.")
    dim = 0
    tensors = args
    # In our tests, dim is passed as a Python int when present.
    if isinstance(args[-1], int):
        dim = args[-1]
        tensors = args[:-1]
    return mint.stack(list(tensors), dim=dim)


def mint_cat_func_with_flatten_inputs(*args):
    """A kwargs-free wrapper for mint.cat used by ms.grad in tests.

    Note:
        OpSampleInput.convert_to_args() expands op_input tuples/lists into multiple
        positional arguments. For mint.cat, the first argument should be a single
        sequence of tensors, so this wrapper repacks positional tensor args into
        a tuple and treats an optional trailing int as dim.
    """
    if not args:
        raise ValueError("mint_cat_func_with_flatten_inputs expects at least one argument.")
    dim = 0
    tensors = args
    # In our tests, dim is passed as a Python int when present.
    if isinstance(args[-1], int):
        dim = args[-1]
        tensors = args[:-1]
    return mint.cat(tuple(tensors), dim=dim)

def mint_nn_adaptiveavgpool1d(op_input, output_size):
    return mint.nn.AdaptiveAvgPool1d(output_size)(op_input)

def torch_nn_adaptiveavgpool1d(op_input, output_size):
    return torch.nn.AdaptiveAvgPool1d(output_size)(op_input)

def mint_nn_constantpad1d(input,padding,value):
    return mint.nn.ConstantPad1d(padding, value)(input)

def torch_nn_constantpad1d(input,padding,value):
    return torch.nn.ConstantPad1d(padding, value)(input)

def ms_tensor_scatter(input, dim, index, src):
    return input.scatter(dim, index, src)

def torch_tensor_scatter(input, dim, index, src):
    return input.scatter(dim, index, src)

def ms_tensor_flip(op_input, dim):
    return op_input.flip(dim)

def torch_tensor_flip(op_input, dim):
    return op_input.flip(dim)

def tensor_moveaxis_ms(op_input, source, destination):
    return op_input.moveaxis(source=source, destination=destination)

def tensor_moveaxis_torch(op_input, source, destination):
    return op_input.moveaxis(source=source, destination=destination)

def mint_bernoulli(input, generator=None):
    return mint.bernoulli(input, generator=generator)

def torch_bernoulli(input, generator=None):
    return torch.bernoulli(input, generator=generator)

def tensor_mint_remainder(input, other):
    return mint.remainder(input, other)

def tensor_torch_remainder(input, other):
    return torch.remainder(input, other)

def tensor_mint_less(input, other):
    return mint.less(input, other)

def tensor_torch_less(input, other):
    return torch.less(input, other)

def tensor_mint_lt(input, other):
    return mint.lt(input, other)

def tensor_torch_lt(input, other):
    return torch.lt(input, other)

def tensor_mm_ms(op_input, mat2):
    return op_input.mm(mat2)

def tensor_mm_torch(op_input, mat2):
    return op_input.mm(mat2)

def tensor_mint_floor_divide(input, other):
    return mint.floor_divide(input, other)

def tensor_torch_floor_divide(input, other):
    return torch.floor_divide(input, other)

def tensor_h_ms(op_input):
    return op_input.H

def tensor_h_torch(op_input):
    return op_input.H

def tensor_t_ms(op_input):
    return op_input.T

def tensor_t_torch(op_input):
    return op_input.T

# wrap tensor method for Tensor.absolute
def tensor_absolute_ms(op_input):
    return op_input.absolute()

def tensor_absolute_torch(op_input):
    return op_input.absolute()

def tensor_acos_ms(op_input):
    return op_input.acos()

def tensor_acosh_ms(op_input):
    return op_input.acosh()

def tensor_median_unary_ms(op_input):
    return op_input.median()

def tensor_median_unary_torch(op_input):
    return op_input.median()

def tensor_min_unary_ms(op_input):
    return op_input.min()

def tensor_min_unary_torch(op_input):
    return op_input.min()

# wrap tensor method for add
def tensor_add_ms_overload(op_input, other):
    return op_input.add(other)

def tensor_add_torch_overload(op_input, other):
    return op_input.add(other)

def tensor_arccos_ms(op_input):
    return op_input.arccos()

def tensor_arccos_torch(op_input):
    return op_input.arccos()

def tensor_any_ms(op_input, dim=None, keepdim=False):
    if dim==None:
        return op_input.any()
    return op_input.any(dim=dim, keepdim=keepdim)

def tensor_any_torch(op_input, dim=None, keepdim=False):
    if dim==None:
        return op_input.any()
    return op_input.any(dim=dim, keepdim=keepdim)

def tensor_median_ms(op_input, dim=None, keepdim=False):
    if dim==None:
        return op_input.median()
    return op_input.median(dim=dim, keepdim=keepdim)

def tensor_median_torch(op_input, dim=None, keepdim=False):
    if dim==None:
        return op_input.median()
    return op_input.median(dim=dim, keepdim=keepdim)

def tensor_min_ms(op_input, dim, keepdim=False):
    return op_input.min(dim=dim, keepdim=keepdim)

def tensor_min_torch(op_input, dim, keepdim=False):
    return op_input.min(dim=dim, keepdim=keepdim)

def tensor_mvlgamma_ms(op_input, p):
    return op_input.mvlgamma(p)

def tensor_mvlgamma_torch(op_input, p):
    return op_input.mvlgamma(p)

def tensor_nan_to_num_ms(op_input, nan=None, posinf=None, neginf=None):
    return op_input.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

def tensor_nan_to_num_torch(op_input, nan=None, posinf=None, neginf=None):
    return op_input.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

def tensor_masked_select_ms(op_input, mask):
    return op_input.masked_select(mask)

def tensor_masked_select_torch(op_input, mask):
    return op_input.masked_select(mask)

def tensor_masked_fill__ms(op_input, mask, value):
    cloned_input = op_input.clone()
    cloned_input.masked_fill_(mask, value)
    return cloned_input

def tensor_masked_fill__torch(op_input, mask, value):
    cloned_input_pt = op_input.clone()
    cloned_input_pt.masked_fill_(mask, value)
    return cloned_input_pt

def tensor_nanmedian_ms(op_input, axis=-1, keepdims=False):
    return op_input.nanmedian(axis=axis, keepdims=keepdims)

def tensor_nanmedian_torch(op_input, axis=-1, keepdims=False):
    return op_input.nanmedian(axis=axis, keepdims=keepdims)

def tensor_nanmean_ms(op_input, axis=None, keepdims=False, dtype=None):
    return op_input.nanmean(axis=axis, keepdims=keepdims, dtype=dtype)

def tensor_nanmean_torch(op_input, axis=None, keepdims=False, dtype=None):
    return op_input.nanmean(axis=axis, keepdims=keepdims, dtype=dtype)

def tensor_min_override_ms(op_input, axis=None, keepdims=False, initial=None, where=True, return_indices=False):
    return op_input.min(axis=axis, keepdims=keepdims, initial=initial, where=where, return_indices=return_indices)

def tensor_any_ms_override(op_input, axis=None, keep_dims=False):
    if axis==None:
        return op_input.any()
    return op_input.any(axis=axis, keep_dims=keep_dims)

def tensor_any_torch_override(op_input, axis=None, keep_dims=False):
    if axis==None:
        return op_input.any()
    return op_input.any(axis=axis, keepdim=keep_dims)

def tensor_angle_ms(op_input):
    return op_input.angle()

def tensor_angle_torch(op_input):
    return op_input.angle()

def tensor_aminmax_ms(op_input, axis=0, keepdims=False):
    return op_input.aminmax(axis=axis, keepdims=keepdims)

def tensor_aminmax_torch(op_input, axis=0, keepdims=False):
    return op_input.aminmax(axis=axis, keepdims=keepdims)

def tensor_amin_ms(op_input, is_kbk=False, axis=None, keepdims=False):
    if is_kbk:
        return op_input.amin(axis=axis, keep_dims=keepdims)
    return op_input.amin(axis=axis, keepdims=keepdims)

def tensor_amin_torch(op_input, is_kbk, axis=None, keepdims=False):
    return op_input.amin(axis=axis, keepdims=keepdims)

def tensor_amax_ms(op_input, is_kbk=False, axis=None, keepdims=False):
    if is_kbk:
        return op_input.amax(axis=axis, keep_dims=keepdims)
    return op_input.amax(axis=axis, keepdims=keepdims)

def tensor_amax_torch(op_input, is_kbk, axis=None, keepdims=False):
    return op_input.amax(axis=axis, keepdims=keepdims)

def tensor_allclose_ms(op_input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return ms.Tensor(op_input.allclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))

def tensor_allclose_torch(op_input, other, rtol=1e-05, atol=1e-08, equal_nan=False ):
    return torch.tensor(op_input.allclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))

#torch not support dim=None
def tensor_all_ms(op_input, dim=None, keepdim=False):
    if dim == None:
        return op_input.all()
    return op_input.all(dim=dim, keepdim=keepdim)

def tensor_all_torch(op_input, dim=None, keepdim=False):
    if dim == None:
        return op_input.all()
    return op_input.all(dim=dim, keepdim=keepdim)

def tensor_all_ms_override(op_input, axis=None, keep_dims=False):
    if axis == None:
        return op_input.all()
    return op_input.all(axis=axis, keep_dims=keep_dims)

def tensor_all_torch_override(op_input, axis=None, keep_dims=False):
    if axis == None:
        return op_input.all()
    return op_input.all(axis=axis, keepdims=keep_dims)

def mint_remainder_ms(input, other):
    return mint.remainder(input, other)

def mint_lt_ms(input, other):
    return mint.lt(input, other)

def mint_less_ms(input, other):
    return mint.less(input, other)

def mint_floor_divide_ms(input, other):
    return mint.floor_divide(input, other)

def tensor_bitwise_and_ms(op_input, other):
    return op_input.bitwise_and(other)

def tensor_bitwise_and_torch(op_input, other):
    return op_input.bitwise_and(other)

def tensor_bitwise_left_shift_ms(op_input, other):
    return op_input.bitwise_left_shift(other)

def tensor_bitwise_left_shift_torch(op_input, other):
    return op_input.bitwise_left_shift(other)

def tensor_bitwise_not_ms(op_input):
    return op_input.bitwise_not()

def tensor_bitwise_not_torch(op_input):
    return op_input.bitwise_not()

def tensor_bitwise_or_ms(op_input, other):
    return op_input.bitwise_or(other)

def tensor_bitwise_or_torch(op_input, other):
    return op_input.bitwise_or(other)

def tensor_bitwise_right_shift_ms(op_input, other):
    return op_input.bitwise_right_shift(other)

def tensor_bitwise_right_shift_torch(op_input, other):
    return op_input.bitwise_right_shift(other)

def tensor_bitwise_xor_ms(op_input, other):
    return op_input.bitwise_xor(other)

def tensor_bitwise_xor_torch(op_input, other):
    return op_input.bitwise_xor(other)

def tensor_bmm_ms(op_input, mat2):
    return op_input.bmm(mat2)

def tensor_bmm_torch(op_input, mat2):
    return op_input.bmm(mat2)

def tensor_broadcast_to_ms(op_input, shape):
    return op_input.broadcast_to(shape)

def tensor_broadcast_to_torch(op_input, shape):
    return op_input.broadcast_to(shape)

def tensor_arccosh_ms(input):
    return input.arccosh()

def tensor_arccosh_torch(input):
    return input.arccosh()

def tensor_arcsin_ms(input):
    return input.arcsin()

def tensor_arcsin_torch(input):
    return input.arcsin()

def tensor_arcsinh_ms(input):
    return input.arcsinh()

def tensor_arcsinh_torch(input):
    return input.arcsinh()

def tensor_arctanh_ms(input):
    return input.arctanh()

def tensor_arctanh_torch(input):
    return input.arctanh()

def tensor_argwhere_ms(input):
    return input.argwhere()

def tensor_argwhere_torch(input):
    return input.argwhere()

def tensor_arctan_ms(input):
    return input.arctan()

def tensor_arctan_torch(input):
    return input.arctan()

def tensor_arctan2_ms(input, other):
    return input.arctan2(other)

def tensor_arctan2_torch(input, other):
    return input.arctan2(other)

def tensor_argsort_ms(input, dim=-1, descending=False, stable=False):
    return input.argsort(dim=dim, descending=descending, stable=stable)

def tensor_argsort_torch(input, dim=-1, descending=False, stable=False):
    return input.argsort(dim=dim, descending=descending, stable=stable)

def tensor_argmin_with_value_ms(input, axis=0, keep_dims=False):
    return input.argmin_with_value(axis=axis, keep_dims=keep_dims)

def tensor_argmin_with_value_torch(input, axis=0, keep_dims=False):
    #argmin not support bool
    if axis == None:
        return (input.min(), (input.to(torch.int32) if input.dtype ==torch.bool else input).argmin(axis=axis, keepdims=keep_dims))
    return tuple(input.min(axis=axis, keepdims=keep_dims))

def tensor_argmax_with_value_ms(input, axis=0, keep_dims=False):
    return input.argmax_with_value(axis=axis, keep_dims=keep_dims)

def tensor_argmax_with_value_torch(input, axis=0, keep_dims=False):
    # argmin not support bool
    if axis == None:
        return (input.max(),
                (input.to(torch.int32) if input.dtype == torch.bool else input).argmax(axis=axis, keepdims=keep_dims))
    return tuple(input.max(axis=axis, keepdims=keep_dims))

def tensor_argmin_ms_overload(input, axis=None, keepdims=False):
    return input.argmin(axis=axis, keepdims=keepdims)

def tensor_argmin_torch_overload(input, axis=None, keepdims=False):
    return input.argmin(axis=axis, keepdims=keepdims)

def tensor_asinh_ms(op_input):
    return op_input.asinh()

def tensor_asinh_torch(op_input):
    return op_input.asinh()

def tensor_asin_ms(op_input):
    return op_input.asin()

def tensor_asin_torch(op_input):
    return op_input.asin()

def tensor_asnumpy_ms(op_input):
    # for torch.bfloat16, to ensure the output dtypes are the same during result comparison.
    if op_input.dtype == ms.bfloat16:
        return op_input.asnumpy().astype(np.float32)
    else:
        return op_input.asnumpy()

def tensor_asnumpy_torch(op_input):
    # unsupported bfloat16
    if op_input.dtype == torch.bfloat16:
        op_input = op_input.float()
    return op_input.numpy()

def tensor_atan_ms(op_input):
    return op_input.atan()

def tensor_atan_torch(op_input):
    return op_input.atan()

def tensor_atanh_ms(op_input):
    return op_input.atanh()

def tensor_atanh_torch(op_input):
    return op_input.atanh()

def tensor_baddbmm_ms(op_input, batch1, batch2, beta=1, alpha=1):
    return op_input.baddbmm(batch1, batch2, beta=beta, alpha=alpha)

def tensor_baddbmm_torch(op_input, batch1, batch2, beta=1, alpha=1):
    return op_input.baddbmm(batch1, batch2, beta=beta, alpha=alpha)

def tensor_bernoulli_ms(op_input, generator=None):
    return op_input.bernoulli(generator=generator)

def tensor_bernoulli_torch(op_input, generator=None):
    return op_input.bernoulli(generator=generator)

def tensor_bernoulli__ms(op_input, p=0.5, generator=None):
    cloned_input = op_input.clone()
    return cloned_input.bernoulli_(p=p, generator=generator)

def tensor_bernoulli__torch(op_input, p=0.5, generator=None):
    cloned_input = op_input.clone()
    return cloned_input.bernoulli_(p=p, generator=generator)

def tensor_bincount_ms(op_input, weights=None, minlength=0):
    return op_input.bincount(weights=weights, minlength=minlength)

def tensor_bincount_torch(op_input, weights=None, minlength=0):
    return op_input.bincount(weights=weights, minlength=minlength)

def tensor_clip_ms(op_input, min=None, max=None):
    return op_input.clip(min=min, max=max)

def tensor_clip_torch(op_input, min=None, max=None):
    return op_input.clip(min=min, max=max)

def tensor_conj_ms(op_input):
    return op_input.conj()

def tensor_conj_torch(op_input):
    return op_input.conj()

def tensor_copysign_ms(x, other):
    return x.copysign(other)

def tensor_copysign_torch(x, other):
    return x.copysign(other)

def tensor_cosh_ms(op_input):
    return op_input.cosh()

def tensor_cosh_torch(op_input):
    return op_input.cosh()

def tensor_copy__ms(op_input, src, non_blocking=False):
    return op_input.copy_(src, non_blocking=non_blocking)

def tensor_copy__torch(op_input, src, non_blocking=False):
    return op_input.copy_(src, non_blocking=non_blocking)

def tensor_count_nonzero_ms(op_input, dim):
    return op_input.count_nonzero(dim)

def tensor_count_nonzero_torch(op_input, dim):
    return op_input.count_nonzero(dim)

def tensor_cross_ms(op_input, other, dim=None):
    return op_input.cross(other, dim=dim)

def tensor_cross_torch(op_input, other, dim=None):
    return op_input.cross(other, dim=dim)

def tensor_ms_from_numpy(op_input):
    return ms.Tensor.from_numpy(op_input)

def tensor_torch_from_numpy(op_input):
    return torch.from_numpy(op_input)

def tensor_ms_item(input):
    return input.item()

def tensor_torch_item(input):
    return input.item()

def tensor_ms_nbytes(op_input):
    return op_input.nbytes

def tensor_torch_nbytes(op_input):
    return op_input.nbytes

def tensor_ms_itemsize(input):
    return input.itemsize

def tensor_torch_itemsize(input):
    return input.itemsize

def tensor_ms_lcm(input, other):
    return input.lcm(other)

def tensor_torch_lcm(input, other):
    return input.lcm(other)

def tensor_ms_ldexp(input, other):
    return input.ldexp(other)

def tensor_torch_ldexp(input, other):
    return input.ldexp(other)

def tensor_ms_lerp(input, other, weight):
    return input.lerp(other, weight)

def tensor_torch_lerp(input, other, weight):
    return input.lerp(other, weight)

def tensor_divide_ms(input, other, rounding_mode=None):
    return input.divide(other, rounding_mode=rounding_mode)

def tensor_divide_torch(input, other, rounding_mode=None):
    return input.divide(other, rounding_mode=rounding_mode)

def tensor_dot_ms(input, other):
    return input.dot(other)

def tensor_dot_torch(input, other):
    return input.dot(other)

def tensor_dsplit_ms(input, indices_or_sections):
    return input.dsplit(indices_or_sections)

def tensor_dsplit_torch(input, indices_or_sections):
    return input.dsplit(indices_or_sections)

def tensor_dtype_ms(input):
    return input.dtype

def tensor_dtype_torch(input):
    return input.dtype

def tensor_expand_ms(input, size):
    return input.expand(size)

def tensor_expand_torch(input, size):
    if isinstance(size, tuple):
        return input.expand(size)
    return input.expand(size.numpy().tolist())

def tensor_exp__ms(input):
    input = input + 0
    return input.exp_()

def tensor_exp__torch(input):
    input = input+0
    return input.exp_()

def tensor_eigvals_ms(input):
    result = ms.ops.eigvals(input).numpy()
    return ms.Tensor(np.take_along_axis(result, np.argsort(np.round(result.real, 0) + complex(0, 1)* result.imag, axis=-1),
                                        axis=-1),dtype=ms.complex64)

def tensor_eigvals_torch(input):
    result = torch.linalg.eigvals(input).numpy()
    return torch.tensor(
        np.take_along_axis(result, np.argsort(np.round(result.real, 0) +  complex(0, 1) * result.imag, axis=-1), axis=-1),
        dtype=torch.complex64)

def tensor_erf_ms(input):
    return input.erf()

def tensor_erf_torch(input):
    return input.erf()

def tensor_erfc_ms(input):
    return input.erfc()

def tensor_erfc_torch(input):
    return input.erfc()

def tensor_erfinv_ms(input):
    return input.erfinv()

def tensor_erfinv_torch(input):
    return input.erfinv()

def tensor_equal_ms(input, other):
    return input.equal(other)

def tensor_equal_torch(input, other):
    return input.eq(other)

def tensor_cummax_ms(op_input, axis):
    return op_input.cummax(axis=axis)

def tensor_cummax_torch(op_input, axis):
    return op_input.cummax(dim=axis)

def tensor_cummin_ms(op_input, axis):
    return op_input.cummin(axis=axis)

def tensor_cummin_torch(op_input, axis):
    return op_input.cummin(dim=axis)

def tensor_cumprod_ms(op_input, dim, dtype=None):
    return op_input.cumprod(dim, dtype=dtype)

def tensor_cumprod_torch(op_input, dim, dtype=None):
    return op_input.cumprod(dim, dtype=dtype)

def tensor_cumsum_ms(op_input, dim, dtype=None):
    return op_input.cumsum(dim=dim, dtype=dtype)

def tensor_cumsum_torch(op_input, dim, dtype=None):
    return op_input.cumsum(dim=dim, dtype=dtype)

def tensor_cumsum_ms_overload(op_input, axis, dtype=None):
    return op_input.cumsum(axis=axis, dtype=dtype)

def tensor_cumsum_torch_overload(op_input, axis, dtype=None):
    return op_input.cumsum(axis=axis, dtype=dtype)

def tensor_data_ms(op_input):
    return op_input.data

def tensor_data_torch(op_input):
    return op_input.data

def tensor_deg2rad_ms(op_input):
    return op_input.deg2rad()

def tensor_deg2rad_torch(op_input):
    return op_input.deg2rad()

def tensor_diag_ms(op_input, diagonal=0):
    return op_input.diag(diagonal=diagonal)

def tensor_diag_torch(op_input, diagonal=0):
    return op_input.diag(diagonal=diagonal)

def tensor_diagflat_ms(op_input, offset=0):
    return op_input.diagflat(offset=offset)

def tensor_diagflat_torch(op_input, offset=0):
    return op_input.diagflat(offset=offset)

def tensor_diagonal_ms(op_input, offset=0, axis1=0, axis2=1):
    return op_input.diagonal(offset=offset, axis1=axis1, axis2=axis2)

def tensor_diagonal_torch(op_input, offset=0, axis1=0, axis2=1):
    return op_input.diagonal(offset=offset, dim1=axis1, dim2=axis2)

def tensor_diagonal_scatter_ms(op_input, src, offset=0, dim1=0, dim2=1):
    return op_input.diagonal_scatter(src, offset=offset, dim1=dim1, dim2=dim2)

def tensor_diagonal_scatter_torch(op_input, src, offset=0, dim1=0, dim2=1):
    return op_input.diagonal_scatter(src, offset=offset, dim1=dim1, dim2=dim2)

def tensor_diff_ms(op_input, n=1, axis=-1, prepend=None, append=None):
    return op_input.diff(n=n, axis=axis, prepend=prepend, append=append)

def tensor_diff_torch(op_input, n=1, axis=-1, prepend=None, append=None):
    return op_input.diff(n=n, dim=axis, prepend=prepend, append=append)

def tensor_digamma_ms(op_input):
    return op_input.digamma()

def tensor_digamma_torch(op_input):
    return op_input.digamma()

def tensor_div_ms(op_input, other, rounding_mode=None):
    return op_input.div(other, rounding_mode=rounding_mode)

def tensor_div_torch(op_input, other, rounding_mode=None):
    return op_input.div(other, rounding_mode=rounding_mode)

def tensor_div__ms(op_input, other, rounding_mode=None):
    cloned_input = op_input.clone()
    cloned_input.div_(other, rounding_mode=rounding_mode)
    return cloned_input

def tensor_div__torch(op_input, other, rounding_mode=None):
    cloned_input = op_input.clone()
    cloned_input.div_(other, rounding_mode=rounding_mode)
    return cloned_input

def tensor_gcd_ms(op_input, other):
    return op_input.gcd(other)

def tensor_gcd_torch(op_input, other):
    return op_input.gcd(other)

def tensor_ge_ms(op_input, other):
    return op_input.ge(other)

def tensor_ge_torch(op_input, other):
    return op_input.ge(other)

def tensor_geqrf_ms(op_input):
    return op_input.geqrf()

def tensor_geqrf_torch(op_input):
    return op_input.geqrf()

def tensor_ger_ms(op_input, vec2):
    return op_input.ger(vec2)

def tensor_ger_torch(op_input, vec2):
    return op_input.ger(vec2)

def tensor_hardshrink_ms(op_input, lambd=0.5):
    return op_input.hardshrink(lambd=lambd)

def tensor_hardshrink_torch(op_input, lambd=0.5):
    return op_input.hardshrink(lambd=lambd)

def tensor_heaviside_ms(op_input, values):
    return op_input.heaviside(values=values)

def tensor_heaviside_torch(op_input, values):
    return op_input.heaviside(values=values)

def tensor_inv_ms(input):
    return input.inv()

def tensor_inv_torch(input):
    return input.reciprocal()

def tensor_inverse_ms(input):
    return input.inverse()

def tensor_inverse_torch(input):
    return input.inverse()

def tensor_isreal_ms(input):
    return input.isreal()

def tensor_isreal_torch(input):
    return input.isreal()

def tensor_isposinf_ms(input):
    return input.isposinf()

def tensor_isposinf_torch(input):
    return input.isposinf()

def tensor_isneginf_ms(input):
    return input.isneginf()

def tensor_isneginf_torch(input):
    return input.isneginf()

def tensor_isnan_ms(input):
    return input.isnan()

def tensor_isnan_torch(input):
    return input.isnan()

def tensor_isinf_ms(input):
    return input.isinf()

def tensor_isinf_torch(input):
    return input.isinf()

def tensor_isfinite_ms(input):
    return input.isfinite()

def tensor_isfinite_torch(input):
    return input.isfinite()

def tensor_isclose_ms(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return input.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan)

def tensor_isclose_torch(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return input.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan)

def tensor_is_signed_ms(input):
    return input.is_signed()

def tensor_is_signed_torch(input):
    return input.is_signed()

def tensor_is_meta_ms(input):
    return input.is_meta

def tensor_is_meta_torch(input):
    return input.is_meta

def tensor_is_pinned_ms(input):
    return input.is_pinned()

def tensor_is_pinned_torch(input):
    return input.is_pinned()

def tensor_is_shared_ms(input):
    return input.is_shared()

def tensor_is_shared_torch(input):
    return input.is_shared()

def tensor_is_floating_point_ms(input):
    return input.is_floating_point()

def tensor_is_floating_point_torch(input):
    return input.is_floating_point()

def tensor_is_complex_ms(input):
    return input.is_complex()

def tensor_is_complex_torch(input):
    return input.is_complex()

def tensor_float_power_ms(op_input, exponent):
    return op_input.float_power(exponent)

def tensor_float_power_torch(op_input, exponent):
    return op_input.float_power(exponent)

def tensor_floor__ms(op_input):
    cloned_input = op_input.clone()
    cloned_input.floor_()
    return cloned_input

def tensor_floor__torch(op_input):
    cloned_input = op_input.clone()
    cloned_input.floor_()
    return cloned_input

def tensor_floor_divide__ms(op_input, other):
    cloned_input = op_input.clone()
    return cloned_input.floor_divide_(other)

def tensor_floor_divide__torch(op_input, other):
    cloned_input = op_input.clone()
    return cloned_input.floor_divide_(other)

def tensor_fmod_ms(op_input, other):
    return op_input.fmod(other)

def tensor_fmod_torch(op_input, other):
    return op_input.fmod(other)

def tensor_masked_fill_ms(input, mask, value):
    return input.masked_fill(mask, value)

def tensor_masked_fill_torch(input, mask, value):
    return input.masked_fill(mask, value)

def tensor_mT_ms(input,):
    return input.mT

def tensor_mT_torch(input,):
    return input.mT

def tensor_mH_ms(input,):
    return input.mH

def tensor_mH_torch(input,):
    return input.mH

def tensor_logsumexp_ms(input,dim=None, keepdim=False):
    return input.logsumexp(dim=dim, keepdim=keepdim)

def tensor_logsumexp_torch(input, dim=None, keepdim=False):
    return input.logsumexp(dim=dim, keepdim=keepdim)

def tensor_logical_xor_ms(input, other):
    return input.logical_xor(other)

def tensor_logical_xor_torch(input, other):
    return input.logical_xor(other)

def tensor_logdet_ms(input):
    return input.logdet()

def tensor_logdet_torch(input):
    return input.logdet()

def tensor_logcumsumexp_ms(input, axis):
    return input.logcumsumexp(axis)

def tensor_logcumsumexp_torch(input, axis):
    return input.logcumsumexp(axis)

def tensor_movedim_ms(input, source, destination):
    return input.movedim(source, destination)

def tensor_movedim_torch(input, source, destination):
    return input.movedim(source, destination)

def tensor_msort_ms(input):
    return input.msort()

def tensor_msort_torch(input):
    return input.msort()

def tensor_mul__ms(input, other):
    return input.mul_(other)

def tensor_mul__torch(input, other):
    return input.mul_(other)

def tensor_multinomial_ms(input, num_sumples, replacement=False):
    if num_sumples == 1000:

        result= input.multinomial(num_sumples, replacement=replacement).numpy()
        empty_np=np.array([])
        for i in range(input.numpy().size):
            empty_np=np.append(empty_np, np.isclose((result==i).sum().item()/num_sumples, input[i]/input.sum(), atol=0.1))
        return ms.Tensor(empty_np, dtype=ms.bool_)

    return input.multinomial(num_sumples, replacement=replacement)

def tensor_multinomial_torch(input, num_sumples, replacement=False):
    if num_sumples == 1000:

        result = input.multinomial(num_sumples, replacement=replacement).numpy()
        empty_np=np.array([])
        for i in range(input.numpy().size):
            empty_np=np.append(empty_np, np.isclose((result == i).sum().item() / num_sumples, input[i] / input.sum(), atol=0.1))
        return torch.tensor(empty_np, dtype=torch.bool)

    return input.multinomial(num_sumples, replacement=replacement)

def tensor_multiply_ms(input, other):
    return input.multiply(other)

def tensor_multiply_torch(input, other):
    return input.multiply(other)

def tensor_lu_solve_ms(input, lu_data, lu_pivots):
    return input.lu_solve(lu_data, lu_pivots)

def tensor_lu_solve_torch(input, lu_data, lu_pivots):
    return input.lu_solve(lu_data, lu_pivots)

def tensor_logit_ms(input, eps=None):
    return input.logit(eps=eps)

def tensor_logit_torch(input, eps=None):
    return input.logit(eps=eps)

def tensor_logaddexp2_ms(input, other):
    return input.logaddexp2(other)

def tensor_logaddexp2_torch(input, other):
    return input.logaddexp2(other)

def tensor_lstsq_ms(input, other, rcond=None, driver=None):
    return lstsq(input, other, rcond=rcond, driver=driver)

def tensor_lstsq_torch(input, other, rcond=None, driver=None):
    return torch.linalg.lstsq(input, other, rcond=rcond, driver=driver)
