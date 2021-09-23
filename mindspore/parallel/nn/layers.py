# Copyright 2021 Huawei Technologies Co., Ltd
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
The basic layer of the Transformer Networks. This is an experimental interface that is subject to
change and/or deletion.
"""
from functools import wraps, partial
import inspect
import math
import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
import mindspore.nn as nn
from mindspore.nn.layer.activation import get_activation
from mindspore.ops import functional as F
from mindspore._checkparam import Validator
from mindspore.ops.primitive import constexpr, Primitive
from .op_parallel_config import default_dpmp_config, OpParallelConfig

__all__ = [
    "FixedSparseAttention"
]


def _args_type_validator_check(*type_args, **type_kwargs):
    """Check whether input data type is correct."""

    def type_check(func):
        sig = inspect.signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal bound_types
            bound_values = sig.bind(*args, **kwargs)

            argument_dict = bound_values.arguments
            if "kwargs" in bound_types:
                bound_types = bound_types["kwargs"]
            if "kwargs" in argument_dict:
                argument_dict = argument_dict["kwargs"]
            for name, value in argument_dict.items():
                if name in bound_types:
                    bound_types[name](value, name)
            return func(*args, **kwargs)

        return wrapper

    return type_check


def _valid_type_checks(types, class_name):
    # types should be a list of types, this function check if the type is in the valid dtypes
    def validator_check_func(value, name):
        # The args of Validator.check_type_name is (arg_name, arg_type, valid_types, prim_name)
        # as the input of _args_type_validator_check is fixed, so we need to manually change the input order
        partial_check = partial(Validator.check_type_name, valid_types=types, prim_name=class_name)
        return partial_check(name, type(value))

    return validator_check_func


def _valid_value_checks(types, class_name):
    # the value should be a list of types, this function check if the value is in the valid dtypes
    def validator_check_func(value, name):
        # The args of Validator.check_type_name is (arg_name, arg_type, valid_types, prim_name)
        # as the input of _args_type_validator_check is fixed, so we need to manually change the input order
        partial_check = partial(Validator.check_type_name, valid_types=types, prim_name=class_name)
        return partial_check(name, value)

    return validator_check_func


@constexpr
def _check_input_shape(input_shape, param_name, func_name, target_len):
    if len(input_shape) != target_len:
        raise ValueError(f"{func_name} {param_name} should be {target_len}d, but got shape {input_shape}")
    return True


@constexpr
def _check_past_none_input_none(use_past, param_name, func_name, input_tensor, default_value=None):
    """ If the past is True, check whether the inputs is None"""
    if not use_past and input_tensor is not default_value:
        raise ValueError(f"{func_name} {param_name} should be {default_value}, if use_past is False.")
    if use_past and input_tensor is default_value:
        raise ValueError(f"{func_name} {param_name} should not be {default_value}, if use_past is True.")
    return True


@constexpr
def _check_shape_equal(input_shape, param_name, func_name, target_shape):
    if len(input_shape) != len(target_shape):
        raise ValueError(f"{func_name} {param_name} shape should be {target_shape},"
                         f"but got {input_shape}")
    for i in range(len(input_shape)):
        if input_shape[i] != target_shape[i]:
            raise ValueError(f"{func_name} {param_name} shape should be {target_shape},"
                             f"but got {input_shape}")
    return True


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    Validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


@constexpr
def _check_input_shape_value(input_shape, dim, param_name, cls_name, target_value):
    if input_shape[dim] != target_value:
        raise ValueError(f"{cls_name} {param_name} at {dim} shape should be {target_value},"
                         f"but got {input_shape[dim]}")


class _LayerNorm(Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(_LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16]:
            raise TypeError(f"param type should in [float32, float16], but found type {type(param_init_type)}")
        if normalized_shape[0] <= 1024:
            self.layer_norm = P.LayerNorm(begin_norm_axis=-1,
                                          begin_params_axis=-1,
                                          epsilon=eps)
        self.is_self_defined = normalized_shape[0] > 1024
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.add = P.Add()
        self.eps = eps
        self.mul = P.Mul()
        self.add2 = P.Add()
        self.real_div = P.RealDiv()

    def construct(self, x):
        r"""
          x : batch x seq_length x hidden_size
        """
        if self.is_self_defined:
            mean = self.mean(x, -1)
            diff = self.sub1(x, mean)
            variance = self.mean(self.square(diff), -1)
            variance_eps = self.sqrt(self.add(variance, self.eps))
            output = self.real_div(diff, variance_eps)
            output = self.add2(self.mul(output, self.gamma), self.beta)
        else:
            output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return output

    def shard(self, strategy):
        r"""
        Set the shard for the layer norm. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy (tuple): The strategy for the dropout. Should be the same shape as the inputs.
        Examples:
            >>> net = mindspore.parallel.nn.transformer.LayerNorm(normalized_shape=(1024, 10))
            >>> net.shard(((10, 2, 1),))
        """
        if self.is_self_defined:
            self.mean.shard(strategy)
            self.square.shard(strategy)
            self.sqrt.shard(strategy)
            self.sub1.shard((strategy[0], strategy[0]))
            self.sub2.shard((strategy[0], strategy[0]))
            self.add.shard((strategy[0], ()))
            self.mul.shard((strategy[0], (1,)))
            self.add2.shard((strategy[0], (1,)))
            self.real_div.shard((strategy[0], strategy[0]))
        else:
            self.layer_norm.shard((strategy[0], (1,), (1,)))

        return self


class _Linear(Cell):
    r"""
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    3-D tensor.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    where :math:`X` is the input tensors, :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: None.
        expert_num (int): The number of experts used in this Linear. Here, for the case expert_num > 1, BatchMatMul is
            used and the first dimension in BatchMatMul indicate expert_num. Default: 1.
        compute_dtype (mstype): The computation type. Default: mstype.float16
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    @cell_attr_register(attrs=['has_bias', 'in_channels', 'out_channels', 'shard_output', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(_Linear, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        if param_init_type not in [mstype.float32, mstype.float16]:
            raise TypeError(f"param type should in [float32, float16], but found type {type(param_init_type)}")
        if activation and not isinstance(activation, str):
            raise ValueError("Activation can only be str, but found type {}".format(activation))
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        if transpose_b:
            weight_shape = [out_channels, in_channels]
        else:
            weight_shape = [in_channels, out_channels]
        self.expert_num = expert_num
        if self.expert_num > 1:
            self.expert_flag = True
            self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                    name="weight")
            self.matmul = P.BatchMatMul(transpose_b=transpose_b)
        else:
            self.expert_flag = False
            self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
            self.matmul = P.MatMul(transpose_b=transpose_b)
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = P.Add()
        self.act_name = activation
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError("The activation must be str or Cell or Primitive,"" but got {}.".format(activation))
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = P.Cast()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        if self.expert_flag is True:
            x = P.Reshape()(x, (self.expert_num, -1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        if self.activation_flag:
            output = self.activation(output)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, strategy_activation=None):
        r"""
         Set the shard for the linear. the strategy size should be equal to the inputs.

         Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

         Args:
             strategy_matmul (tuple): The strategy for the matmul. Should be the same shape as the inputs.
             strategy_bias (tuple): The strategy for the bias_add. Should be the same shape as the inputs.
             strategy_activation (tuple): The strategy for the strategy_activation. Should be the same shape as
                the inputs.
         """
        self.matmul.shard(strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        if self.activation_flag:
            # some operations has many primitives, need to manually set the shard
            if self.act_name.lower() == "leakyrelu":
                self.activation.select_op.shard((strategy_activation[0], strategy_activation[0]))
            elif self.act_name.lower() == "logsigmoid":
                self.activation.mul.shard((strategy_activation[0], ()))
                self.activation.exp.shard(strategy_activation)
                self.activation.add.shard((strategy_activation[0], ()))
                self.activation.rec.shard(strategy_activation)
                self.activation.log.shard(strategy_activation)
            elif self.act_name.lower() == "logsoftmax":
                raise ValueError("logsoftmax is not supported.")
            else:
                getattr(self.activation, self.act_name).shard(strategy_activation)

        return self


class FixedSparseAttention(nn.Cell):
    """
    Fixed Sparse Attention Layer

    This function contains the sparse attention primitives used in Sparse Transformers (see paper).
    https://arxiv.org/abs/1904.10509
    Specifically, it includes the following:
    1. A faster implementation of normal attention (the upper triangle is not computed, and many operations are fused).
    2. An implementation of "strided" and "fixed" attention, as in the Sparse Transformers paper.

    Args:
        batch_size (int): Number of input batch size.
        num_heads (int): Number of attention heads.
        block_size (int): An integer determining the block size. Current implementation of sparse self-attention
                          is based on blocked sparse matrices. In which this parameter defines size of such blocks,
                          Block X Block. only supports 64 for now
        seq_length (int): length of input sequence, only supports 1024 for now
        num_different_global_patterns (int):An integer determining number of different global attentions layouts.
                                            While global attention can be fixed by which block/s are representative of
                                            any local window, since there are multi-heads, each head can use a
                                            different global representative, only supports 4 for now
        size_per_head (int): An integer determining embedding size of each attention head,
                             only supports 64, 128 for now

    Inputs:
        - **q** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, seq_length, hidden_size]): Sequence of
          queries to query the context.
        - **k** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, seq_length, hidden_size]): Sequence of
          queries to query the context.
        - **v** (Tensor) - Tensor value (:class:`mstype.fp16` [batch size, sequence length, Embedding Size]):
          Sequence of queries to query the context.
        - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp32`, :class:`mstype.fp16`
          [batch_size, seq_length, seq_length]): Lower triangular matrix to pass masked information.

    Outputs:
        A Tensor. The output of the attention with shape [batch_size, seq_length, hidden_size]

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> model = FixedSparseAttention(batch_size=2,
        ...                              num_heads=8,
        ...                              size_per_head=64,
        ...                              block_size=64)
        >>> q = Tensor(np.ones((2, 1024, 8*64)), mstype.float16)
        >>> k = Tensor(np.ones((2, 1024, 8*64)), mstype.float16)
        >>> v = Tensor(np.ones((2, 1024, 8*64)), mstype.float16)
        >>> attention_mask = Tensor(np.ones((2, 1024, 1024)), mstype.float32)
        >>> output = model(q, k, v, attention_mask)
        >>> print(output.shape)
        (2, 1024, 512)
    """

    @_args_type_validator_check(batch_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                size_per_head=Validator.check_positive_int,
                                block_size=Validator.check_positive_int,
                                seq_length=Validator.check_positive_int,
                                num_different_global_patterns=Validator.check_positive_int,
                                parallel_config=_valid_type_checks([OpParallelConfig], "FixedSparseAttention"))
    def __init__(self,
                 batch_size,
                 num_heads,
                 size_per_head,
                 block_size,
                 seq_length=1024,
                 num_different_global_patterns=4,
                 parallel_config=default_dpmp_config):
        super(FixedSparseAttention, self).__init__()
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        if num_heads % mp != 0:
            raise ValueError(f"The number of heads {num_heads} must be a "
                             f"multiple of parallel_config.model_parallel {mp}.")
        if batch_size % dp != 0:
            raise ValueError(f"The batch_size {batch_size} must be a "
                             f"multiple of parallel_config.data_parallel {parallel_config.data_parallel}.")
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = size_per_head * num_heads
        self.num_heads = num_heads
        self.block_size = block_size
        self.block_num = seq_length // block_size
        self.size_per_head = size_per_head
        self.global_size = seq_length // 4
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, mp, 1),))
        self.batch_matmul = P.BatchMatMul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.multiply = P.Mul().shard(((dp, 1, 1, 1), (1, 1, 1)))
        self.multiply_data = Tensor([-10000.0], dtype=mstype.float32)
        self.parallel_config = parallel_config
        size_per_head_list = [64, 128]
        if self.seq_length != 1024:
            raise ValueError("seq_length only supports 1024 for now.")
        if self.block_size != 64:
            raise ValueError("block_size only supports 64 for now.")
        if num_different_global_patterns != 4:
            raise ValueError("num_different_global_patterns only supports 4 for now.")
        if self.size_per_head not in size_per_head_list:
            raise ValueError(f"size_per_head only supports {size_per_head_list} for now, "
                             f"but found {self.size_per_head}")
        local_ones = np.ones((self.block_size, self.block_size),
                             dtype=np.float16)
        global_mask_original = np.ones((self.seq_length, self.global_size), dtype=np.float16)
        for i in range(self.seq_length):
            for j in range(self.global_size):
                if i // 16 >= (j // 16 + 1) * 4:
                    global_mask_original[i, j] = 0.0

        global_mask_original = -10000 * global_mask_original
        global_mask_fx = global_mask_original.reshape((self.seq_length // 16, 16, self.global_size // 16, 16))
        global_mask = np.transpose(global_mask_fx, (2, 0, 1, 3))
        global_mask = np.repeat(global_mask[np.newaxis, :, :, :, :], self.batch_size, axis=0)
        global_mask = global_mask.reshape((self.batch_size * self.global_size // 16, self.seq_length // 16, 16, 16))
        self.global_mask = Tensor(global_mask, mstype.float32)
        self.local_mask_triangle = Tensor(np.tril(local_ones), mstype.float32)
        self.scale_factor = Tensor((math.sqrt(self.size_per_head)))
        self.matmul_dds = P.MatmulDDS(self.batch_size, self.num_heads).shard(((mp, dp, 1, 1),
                                                                              (mp, dp, 1, 1),
                                                                              (1, dp, 1, 1),
                                                                              (dp, 1, 1, 1)))
        self.matmul_dsd = P.DSDMatmul().shard(((dp, mp, 1, 1, 1, 1, 1), (dp, mp, 1, 1, 1, 1, 1), (dp, mp, 1, 1)))
        self.sub1 = P.Sub().shard(((1,), (dp, 1, 1, 1)))
        self.mul1 = P.Mul().shard(((dp, 1, 1, 1), (1,)))
        self.transpose1 = P.Transpose().shard(((dp, 1, 1, 1),))
        self.transpose2 = P.Transpose().shard(((dp, 1, 1, 1),))
        self.transpose3 = P.Transpose().shard(((dp, mp, 1, 1, 1, 1),))
        self.transpose4 = P.Transpose().shard(((dp, mp, 1, 1),))
        self.div = P.RealDiv().shard(((mp, dp, 1, 1), ()))
        self.slice1 = P.StridedSlice().shard(((dp, 1, 1),))

    def _transpose_inputs(self, q, k, v):
        """
        do reshape and transpose to inputs
        """
        q = self.transpose(
            self.reshape(
                q,
                (-1, 16, self.num_heads * self.size_per_head // 16, 16)),
            (2, 0, 1, 3))
        k = self.transpose(
            self.reshape(
                k, (-1, 16, self.num_heads * self.size_per_head // 16, 16)),
            (2, 0, 1, 3))
        v = self.transpose(
            self.reshape(
                v,
                (-1, 16, self.num_heads * self.size_per_head // 16, 16)),
            (0, 2, 3, 1))

        return q, k, v

    def _generate_attention_mask(self, attention_mask):
        """
        generate global attention mask and local attention mask from origin attention mask
        """
        attention_mask = self.reshape(attention_mask, (-1, self.seq_length, self.seq_length))
        input_mask = self.slice1(attention_mask, (0, self.seq_length - 1, 0),
                                 (self.batch_size, self.seq_length, self.seq_length), (1, 1, 1))
        input_mask = self.reshape(input_mask, (-1, self.seq_length))
        input_shape = P.Shape()(input_mask)  # bs, seq_length
        # bs, block_num, 1, block_size
        local_shape_right = (input_shape[0], self.block_num, 1, self.block_size)
        # bs, block_num, block_size, 1
        local_shape_left = (input_shape[0], self.block_num, self.block_size, 1)
        local_mask_left = self.reshape(input_mask, local_shape_left)
        local_mask_right = self.reshape(input_mask, local_shape_right)
        # bs, block_num, block_size, block_size
        local_attention_mask = self.batch_matmul(local_mask_left, local_mask_right)
        lower_triangle = P.ExpandDims()(self.local_mask_triangle, 0)
        local_attention_mask = self.multiply(local_attention_mask, lower_triangle)
        local_multiplied_out = self.sub1(P.Cast()(F.tuple_to_array((1.0,)), mstype.float32),
                                         P.Cast()(local_attention_mask, mstype.float32))
        local_adder = self.mul1(local_multiplied_out, self.multiply_data)
        local_mask_original = self.transpose1(local_adder, (0, 2, 1, 3))
        local_mask_original = self.reshape(
            local_mask_original,
            (self.batch_size * self.block_size, self.block_num * self.block_size))
        local_mask_fx = self.reshape(
            local_mask_original,
            (self.batch_size * self.block_size // 16, 16,
             self.block_num * self.block_size // 16, 16))
        local_mask = self.transpose2(local_mask_fx, (2, 0, 1, 3))
        global_mask = self.global_mask

        return local_mask, global_mask

    def construct(self, q, k, v, attention_mask):
        _check_shape_equal(F.shape(q), "q", self.cls_name,
                           [self.batch_size, self.seq_length, self.hidden_size])
        _check_input_dtype(F.dtype(q), "q", [mstype.float16], self.cls_name)
        _check_shape_equal(F.shape(k), "k", self.cls_name,
                           [self.batch_size, self.seq_length, self.hidden_size])
        _check_input_dtype(F.dtype(k), "k", [mstype.float16], self.cls_name)
        _check_shape_equal(F.shape(v), "v", self.cls_name,
                           [self.batch_size, self.seq_length, self.hidden_size])
        _check_input_dtype(F.dtype(v), "v", [mstype.float16], self.cls_name)
        _check_shape_equal(F.shape(attention_mask), "attention_mask", self.cls_name,
                           [self.batch_size, self.seq_length, self.seq_length])
        _check_input_dtype(F.dtype(attention_mask), "attention_mask", [mstype.float32, mstype.float16], self.cls_name)

        q, k, v = self._transpose_inputs(q, k, v)
        local_mask, global_mask = self._generate_attention_mask(attention_mask)
        q = self.div(q, F.cast(self.scale_factor, F.dtype(q)))
        k = self.div(k, F.cast(self.scale_factor, F.dtype(k)))
        local_prob, global_prob = self.matmul_dds(q, k, local_mask, global_mask)
        attention = self.matmul_dsd(local_prob, global_prob, v)
        attention_merge = self.transpose3(attention, (0, 1, 3, 4, 2, 5))
        attention_merge = F.reshape(
            attention_merge,
            (-1, self.num_heads, self.seq_length, self.size_per_head))
        attention_merge = self.transpose4(attention_merge, (0, 2, 1, 3))
        attention_merge = F.reshape(
            attention_merge,
            (-1, self.seq_length, self.size_per_head * self.num_heads))

        return attention_merge
