# Copyright 2024 Huawei Technologies Co., Ltd
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
change or deletion.
"""
from __future__ import absolute_import

from functools import wraps, partial
import inspect
import math

from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
import mindspore.common.dtype as mstype
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
import logging
from mindformers.modules.transformer.op_parallel_config import OpParallelConfig, MoEParallelConfig

logger = logging.getLogger()

__all__ = ["Linear"]


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


class _LayerInputCheck:
    """
       A input check class for the inputs of the transformer model.
    """

    @staticmethod
    def check_shape_length(input_shape, param_name, func_name, target_len):
        """
        Check the input shape's length is equal to the expected shape
        :param input_shape(list): a list of the tensor shapes.
        :param param_name(str): the name of the checked parameter.
        :param func_name(str): the name of the function.
        :param target_len: the expected length of the shape.
        :return:
        """
        if not isinstance(target_len, list):
            target_len = [target_len]
        matched = False
        for item in target_len:
            if len(input_shape) == item:
                matched = True
        if not matched:
            raise ValueError(f"{func_name} {param_name} shape length must be one of {target_len} dimension, "
                             f"but got shape {input_shape}")
        return True

    @staticmethod
    def check_shape_equal(input_shape, param_name, func_name, target_shape):
        """
        Check the input shape's is equal to the expected shape
        :param input_shape(list): a list of the tensor shapes.
        :param param_name(str): the name of the checked parameter.
        :param func_name(str): the name of the function.
        :param target_shape: the expected shape.
        :return:
        """
        if not isinstance(target_shape[0], list):
            target_shape = [target_shape]
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        _LayerInputCheck.check_shape_length(input_shape, param_name, func_name,
                                            [len(item) for item in target_shape])
        matched = False
        for item in target_shape:
            if item == input_shape:
                matched = True
                break

        if not matched:
            raise ValueError(f"{func_name} {param_name} shape must be one of {target_shape},"
                             f"but got {input_shape}")
        return True

    @staticmethod
    def check_shape_value_on_axis(input_shape, dim, param_name, cls_name, target_value):
        """ Check whether the input_shape[dim] is equal to target value"""
        if input_shape[dim] != target_value:
            raise ValueError(f"{cls_name} {param_name} at {dim} shape must be {target_value},"
                             f"but got {input_shape[dim]}")
        return True

    @staticmethod
    def check_shape_equal_without_batch(input_shape, param_name, func_name, target_shape):
        """
        Check the input shape's is equal to the expected shape, the value on 0-th is viewed as batch, and the
        batch size will not be checked.
        """
        target_shape = target_shape
        length, hidden = target_shape
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        _LayerInputCheck.check_shape_length(input_shape, param_name, func_name,
                                            [len(target_shape), len(target_shape) + 1])
        if input_shape[-1] != hidden:
            raise ValueError(f"For {func_name}, the last dimension of {param_name} shape must be {hidden},"
                             f"but got the last dimension {input_shape[-1]} in {input_shape}.")
        if input_shape[0] == 0:
            raise ValueError(f"For {func_name}, the first dimension of {param_name} shape greater than 0,"
                             f"but got the first dimension {input_shape[0]} in {input_shape}.")
        if len(input_shape) == 2 and input_shape[0] % length != 0:
            raise ValueError(f"For {func_name}, the first dimension of {param_name} shape should be divisible "
                             f"by {length}, "
                             f"but got the first dimension {input_shape[0]} in {input_shape}.")
        return True


@constexpr
def _check_past_none_input_none(use_past, param_name, func_name, default_value, is_tensor, is_default):
    """ If the use_past is True, check whether the inputs is None"""
    if not use_past:
        if is_tensor:
            raise TypeError(f"{func_name} {param_name} must be {default_value}, if use_past is False, but found "
                            f"a tensor")
        if not is_default:
            raise TypeError(f"{func_name} {param_name} must be {default_value}, if use_past is False.")
    else:
        if not is_tensor:
            raise TypeError(f"{func_name} {param_name} must be tensor, if use_past is True")
    return True


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    Validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


@constexpr
def _check_input_shape(input_shape, param_name, func_name, target_len):
    # check the input length
    _LayerInputCheck.check_shape_length(input_shape, param_name, func_name, target_len)


@constexpr
def _check_shape_equal(input_shape, param_name, func_name, target_shape):
    # check the input length
    _LayerInputCheck.check_shape_equal(input_shape, param_name, func_name, target_shape)


@constexpr
def _check_input_shape_value(input_shape, dim, param_name, cls_name, target_value):
    _LayerInputCheck.check_shape_value_on_axis(input_shape, dim, param_name, cls_name, target_value)


@constexpr
def _check_shape_equal_without_batch(input_shape, param_name, func_name, target_shape):
    _LayerInputCheck.check_shape_equal_without_batch(input_shape, param_name, func_name, target_shape)

class Linear(Cell):
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
            eg. 'ReLU'. Default: None.
        expert_num (int): The number of experts used in this Linear. Here, for the case expert_num > 1, BatchMatMul is
            used and the first dimension in BatchMatMul indicate expert_num. Default: 1.
        outer_batch (int): The replication number of experts. The replication is effective only when MoE is applied.
            Default: 1.
        expert_group_size (int): The number of tokens in each data parallel group. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: mstype.float16
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

    @cell_attr_register
    @_args_type_validator_check(in_channels=Validator.check_positive_int,
                                out_channels=Validator.check_positive_int,
                                has_bias=Validator.check_bool,
                                transpose_b=Validator.check_bool,
                                expert_num=Validator.check_positive_int,
                                outer_batch=Validator.check_positive_int,
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "Linear"),
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "Linear"))
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 skip_redistribution=False):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not (isinstance(activation, str) or activation is None or issubclass(activation, nn.Cell)):
            raise TypeError(f"For Linear cell, the activation should str type or nn.Cell type, but got {activation}.")
        if isinstance(weight_init, Tensor) and (weight_init.ndim != 2 or weight_init.shape[0] != out_channels or
                                                weight_init.shape[1] != in_channels):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.expert_group_size = expert_group_size
        self.transpose_b = transpose_b
        if self.expert_num > 1:
            self.expert_flag = True
            self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                    name="weight")
            self.matmul = P.BatchMatMul(transpose_b=transpose_b)
        else:
            self.expert_flag = False
            self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
            self.matmul = P.MatMul(transpose_b=transpose_b)
        self.use_expert_group_size = _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) \
                                     and not _is_sharding_propagation() and self.expert_flag is True
        if self.use_expert_group_size is True and self.expert_group_size is None:
            raise ValueError("'expert_group_size' should be configured as an integer in MoEConfig.")
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")
            if self.expert_flag:
                self.bias = Parameter(initializer(bias_init,
                                                  [1, self.expert_num, 1, out_channels], param_init_type), name="bias")
            else:
                self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = P.Add()
        self.act_name = activation
        if callable(activation):
            self.activation = activation()
        else:
            self.activation = activation
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        if skip_redistribution:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.shape = P.Shape()

    def construct(self, x):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(x)
        x = F.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, strategy_activation=None, out_strategy_matmul=None):
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
            out_strategy_matmul (tuple): The out strategy for the matmul. Should be the same shape as the inputs.
        """
        self.matmul.shard(in_strategy=strategy_matmul, out_strategy=out_strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        if self.activation_flag and isinstance(self.act_name, str):
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
                raise ValueError("The 'LogSoftmax' function is not supported in semi auto parallel "
                                 "or auto parallel mode.")
            else:
                getattr(self.activation, self.act_name).shard(strategy_activation)
        elif self.activation_flag and isinstance(self.activation, Cell):
            if hasattr(self.activation, 'activation_shard') and strategy_activation:
                shard_tuple = strategy_activation[0]
                if len(shard_tuple) == 2:
                    parallel_config = OpParallelConfig(data_parallel=shard_tuple[0],
                                                       model_parallel=shard_tuple[1])
                elif len(shard_tuple) == 4:
                    parallel_config = MoEParallelConfig(data_parallel=shard_tuple[0],
                                                        expert_parallel=shard_tuple[1],
                                                        model_parallel=shard_tuple[2])
                else:
                    raise ValueError("The user-defined activation function currently only supports the case where the "
                                     "input policy is 2 or 4, so that relevant policies can be extracted from it."
                                     "To avoid this error, you need to add the function of extracting "
                                     "'ParallelConfig' or 'OpParallelConfig' for the incoming strategy_activation ")
                self.activation.activation_shard(parallel_config)
            else:
                logger.warning("The user passed the custom defined activation function %s. "
                               "If the user want to enable shard for the activation cell, "
                               "the user should set the shard for each primitives in the cell.", self.activation_flag)
        return self

def _get_interleave(n):
    """calculate slopes of alibi tensor"""
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)

    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return _get_interleave_power_of_2(closest_power_of_2) + \
           _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
