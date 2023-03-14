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

"""inner_ops"""

import numbers
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common import dtype as mstype
from mindspore.common.dtype import tensor, dtype_to_pytype
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer, Primitive
from mindspore.ops import signature as sig


class ScalarCast(PrimitiveWithInfer):
    """
    Casts the input scalar to another type.

    Refer to :func:`mindspore.ops.scalar_cast` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> scalar_cast = ops.ScalarCast()
        >>> output = scalar_cast(255.0, mindspore.int32)
        >>> print(output)
        255
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, x, t):
        validator.check_equal_int(len(x['shape']), 0, 'x shape', self.name)
        value, to = x['value'], t['value']
        if value is not None:
            validator.check_value_type("value", value, [numbers.Number, bool], self.name)
            if isinstance(to, type(tensor)):
                to = to.element_type()
            np_type = dtype_to_pytype(to)
            value = np_type(value)
        out = {'shape': x['shape'],
               'dtype': t['value'],
               'value': value}
        return out


class Randperm(Primitive):
    """
    Generates n random samples from 0 to n-1 without repeating. If `max_length` > n,
    the last `max_length-n` elements will be filled with `pad`.

    Args:
        max_length (int): Number of items expected to get and the number must be greater than 0. Default: 1.
        pad (int): The pad value to be filled. Default: -1.
        dtype (mindspore.dtype): The type of output. Default: mindspore.int32.

    Inputs:
        - **n** (Tensor) - The input tensor with shape (1,) with and dtype int32 or int64.
          `n` must be in range [0, `max_length`].

    Outputs:
        - **output** (Tensor) - The output Tensor with shape: (`max_length`,) and type: `dtype`.

    Raises:
        TypeError: If neither `max_length` nor `pad` is an int.
        TypeError: If `n` is not a Tensor.
        TypeError: If `n` has non-Int elements.
        TypeError: If `n` has negative elements.
        TypeError: If `dtype` is not supported.
        ValueError: If `n` is out of range of `dtype`.
        ValueError: If `n` is larger than `max_length`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> # The result of every execution is different because this operator will generate n random samples.
        >>> randperm = ops.Randperm(max_length=30, pad=-1)
        >>> n = Tensor([20], dtype=mindspore.int32)
        >>> output = randperm(n)
        >>> print(output)
        [15 6 11 19 14 16 9 5 13 18 4 10 8 0 17 2 1 12 3 7
         -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
    """

    @prim_attr_register
    def __init__(self, max_length=1, pad=-1, dtype=mstype.int32):
        """Initialize Randperm"""
        validator.check_value_type("pad", pad, [int], self.name)
        validator.check_value_type("max_length", max_length, [int], self.name)
        validator.check_int(max_length, 1, Rel.GE, "max_length", self.name)
        valid_values = (mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16,
                        mstype.uint32, mstype.uint64, mstype.float16, mstype.float32, mstype.float64)
        validator.check_type_name("dtype", dtype, valid_values, self.name)
        self.dtype = dtype
        self.max_length = max_length
        self.init_prim_io_names(inputs=[], outputs=['output'])


class NoRepeatNGram(Primitive):
    """
    Updates the probability of occurrence of words with its corresponding n-grams.

    During beam search, if consecutive `ngram_size` words exist in the generated word sequence,
    the consecutive `ngram_size` words will be avoided during subsequent prediction.
    For example, when `ngram_size` is 3, the generated word sequence is [1, 2, 3, 2, 3],
    the next predicted word will not be 2 and the value of `log_probs` will be replaced with -FLOAT_MAX.
    Because 3 consecutive words [2, 3, 2] do not appear twice in the word sequence.

    Args:
        ngram_size (int): Size of n-grams, must be greater than 0. Default: 1.

    Inputs:
        - **state_seq** (Tensor) - n-gram word series, a 3-D tensor with shape: (batch_size, beam_width, m).
        - **log_probs** (Tensor) - Probability of occurrence of n-gram word series, a 3-D
          tensor with shape: (batch_size, beam_width, vocab_size).
          The value of log_probs will be replaced with -FLOAT_MAX when n-grams repeated.

    Outputs:
        - **log_probs** (Tensor) - The output Tensor with same shape and type as original `log_probs`.

    Raises:
        TypeError: If `ngram_size` is not an int.
        TypeError: If neither `state_seq` nor `log_probs` is a Tensor.
        TypeError: If the dtype of `state_seq` is not int.
        TypeError: If the dtype of `log_probs` is not float.
        ValueError: If `ngram_size` is less than zero.
        ValueError: If `ngram_size` is greater than m.
        ValueError: If neither `state_seq` nor `log_probs` is not a 3-D Tensor.
        ValueError: If the batch_size of `state_seq` and `log_probs` are not equal.
        ValueError: If the beam_width of `state_seq` and `log_probs` are not equal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> no_repeat_ngram = ops.NoRepeatNGram(ngram_size=3)
        >>> state_seq = Tensor([[[1, 2, 1, 2, 5, 1, 2],
        ...                      [9, 3, 9, 5, 4, 1, 5]],
        ...                     [[4, 8, 6, 4, 5, 6, 4],
        ...                      [4, 8, 8, 4, 3, 4, 8]]], dtype=mindspore.int32)
        >>> log_probs = Tensor([[[0.7, 0.8, 0.6, 0.9, 0.2, 0.8, 0.4, 0.6, 0.2, 0.7],
        ...                      [0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.9, 0.8, 0.7, 0.1]],
        ...                     [[0.9, 0.7, 0.6, 0.3, 0.5, 0.3, 0.5, 0.4, 0.8, 0.6],
        ...                      [0.5, 0.8, 0.8, 0.7, 0.7, 0.8, 0.2, 0.7, 0.9, 0.7]]], dtype=mindspore.float32)
        >>> output = no_repeat_ngram(state_seq, log_probs)
        >>> print(output)
        [[[ 6.9999999e-01 -3.4028235e+38  6.0000002e-01  8.9999998e-01
            2.0000000e-01 -3.4028235e+38  4.0000001e-01  6.0000002e-01
            2.0000000e-01  6.9999999e-01]
          [ 4.0000001e-01  5.0000000e-01  6.0000002e-01  6.9999999e-01
            8.0000001e-01  1.0000000e-01  8.9999998e-01  8.0000001e-01
            6.9999999e-01  1.0000000e-01]]
         [[ 8.9999998e-01  6.9999999e-01  6.0000002e-01  3.0000001e-01
            5.0000000e-01 -3.4028235e+38  5.0000000e-01  4.0000001e-01
            8.0000001e-01  6.0000002e-01]
          [ 5.0000000e-01  8.0000001e-01  8.0000001e-01  6.9999999e-01
            6.9999999e-01  8.0000001e-01  2.0000000e-01  6.9999999e-01
           -3.4028235e+38  6.9999999e-01]]]
    """

    @prim_attr_register
    def __init__(self, ngram_size=1):
        """NoRepeatNGram Randperm"""
        validator.check_value_type("ngram_size", ngram_size, [int], self.name)
        validator.check_int(ngram_size, 1, Rel.GE, "ngram_size", self.name)
        self.ngram_size = ngram_size
        self.init_prim_io_names(inputs=['state_seq', 'log_probs'], outputs=['log_probs'])


class LambApplyOptimizerAssign(PrimitiveWithInfer):
    r"""
    Updates gradients by LAMB optimizer algorithm. Get the compute ratio.

    The Lamb optimizer is proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    <https://arxiv.org/abs/1904.00962>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            m = \frac{m}{1 - \beta_1^t} \\
            v = \frac{v}{1 - \beta_2^t} \\
            r = \frac{m}{\sqrt{v} + \epsilon} \\
            w = w - l * \frac{\left \| w \right \|}{\left \| r \right \|} * (r + \lambda * w))
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents learning rate `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent `beta1_power` and
    `beta2_power`, :math:`\lambda` represents `weight_decay`, :math:`w` represents `var`, :math:`\epsilon` represents
    `epsilon`.

    Inputs:
        - **gradient** (Tensor) - Gradient of parameters, float32/float16.
        - **v** (Tensor) - the 2nd moment vector in the updating formula, has the same type as `gradient`.
        - **m** (Tensor) - The 1st moment vector in the updating formula, has the same type as `gradient`.
        - **var** (Tensor) - Weights to be updated, has the same type as `gradient`.
        - **beta1** (Tensor) - :math:`beta_1` in the updating formula, float32/float16.
        - **sub1** (Tensor) - :math:`1-beta_1` in the updating formula, has the same type as `beta1`.
        - **beta2** (Tensor) - :math:`beta_2` in the updating formula, has the same type as `beta1`.
        - **sub2** (Tensor) - :math:`1-beta_2` in the updating formula, has the same type as `beta1`.
        - **epsilon** (Tensor) - Term added to the denominator, has the same type as `beta1`.
        - **steps** (Tensor) - :math:`t` in the updating formula, global step, has the same type as `beta1`.
        - **lr** (Tensor) - :math:`l` in the updating formula, learning rate, has the same type as `beta1`.
        - **decay_flag** (Tensor) -Specify whether param update with weight decay, has the same type as `beta1`.
        - **weight_decay** (Tensor) - :math:`\lambda` in the updating formula, has the same type as `beta1`.

    Outputs:
        Tensor, the compute ratio r.
        - **update** (Tensor) - :math:`r + \lambda * w` in the updating formula. The same shape and data type as `m`.
        - **v** (Tensor) - the 2nd moment vector in the updating formula after updated inplace,
                           has the same type as `gradient`.
        - **m** (Tensor) - The 1st moment vector in the updating formula after updated inplace,
                           has the same type as `gradient`.

    Supported Platforms:
        ``Ascend``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LambApplyOptimizerAssign"""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, grad_shape, v_shape, m_shape, var_shape, beta1_shape, sub1_shape,
                    beta2_shape, sub2_shape, eps_shape, steps_shape, use_weight_shape, weight_decay_shape):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "grad_shape", grad_shape, Rel.EQ, self.name)
        return m_shape, v_shape, m_shape

    def infer_dtype(self, grad_dtype, v_dtype, m_dtype, var_dtype, beta1_dtype, sub1_dtype,
                    beta2_dtype, sub2_dtype, eps_dtype, steps_dtype, use_weight_dtype, weight_decay_dtype):
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)

        args = {"beta1": beta1_dtype, "sub1": sub1_dtype, "beta2": beta2_dtype, "sub2": sub2_dtype,
                "eps": eps_dtype, "steps": steps_dtype, "use_weight": use_weight_dtype,
                "weight_decay": weight_decay_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float16, mstype.float32], self.name, True)
        return m_dtype, v_dtype, v_dtype


class LambApplyWeightAssign(PrimitiveWithInfer):
    r"""
    Updates gradients by LAMB optimizer algorithm. The weight update part.

    The Lamb optimizer is proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    <https://arxiv.org/abs/1904.00962>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            m = \frac{m}{1 - \beta_1^t} \\
            v = \frac{v}{1 - \beta_2^t} \\
            r = \frac{m}{\sqrt{v} + \epsilon} \\
            w = w - l * \frac{\left \| w \right \|}{\left \| r \right \|} * (r + \lambda * w))
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents learning rate `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent `beta1_power` and
    `beta2_power`, :math:`\lambda` represents `weight_decay`, :math:`w` represents `var`, :math:`\epsilon` represents
    `epsilon`.

    Inputs:
        - **w_norm** (Tensor) - :math:`\left \| w \right \|` in the updating formula, float32/float16.
        - **g_norm** (Tensor) - :math:`\left \| r \right \|` in the updating formula, has the same type as `w_norm`.
        - **lr** (Tensor) - :math:`l` in the updating formula, the learning rate, float32/float16.
        - **update** (Tensor) - :math:`r + \lambda * w` in the updating formula, float32/float16.
        - **var** (Tensor) - Weights to be updated, the same shape and type as `update`.

    Outputs:
        - **var** (Tensor) - Weights to be updated in place, the same shape and type as `var` in inputs.

    Supported Platforms:
        ``Ascend``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LambApplyWeightAssign"""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, w_norm_shape, g_norm_shape, lr_shape, update_shape, var_shape):
        validator.check("var_shape", var_shape, "update_shape", update_shape, Rel.EQ, self.name)
        return var_shape

    def infer_dtype(self, w_norm_dtype, g_norm_dtype, lr_dtype, update_dtype, var_dtype):
        args = {"var": var_dtype, "update": update_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)

        args = {"w_norm": w_norm_dtype, "g_norm": g_norm_dtype, "lr": lr_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float16, mstype.float32], self.name, True)
        return var_dtype


class FusedWeightScaleApplyMomentum(PrimitiveWithInfer):
    """
    Optimizer that implements the Momentum algorithm with weight decay and loss scale.

    Refer to the paper `On the importance of initialization and momentum in deep
    learning <https://dl.acm.org/doi/10.5555/3042817.3043064>`_  for more details.

    Refer to :class:`mindspore.nn.Momentum` for more details about the formula and usage.

    Inputs of `variable`, `accumulation` and `gradient` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    relatively highest priority data type.
    Data type conversion of Parameter is not supported. RuntimeError exception will be thrown.

    Inputs:
        - **weight_decay** (Tensor) - The weight decay value, must be a scalar tensor with float data type.
          Default: 0.0.
        - **loss_scale** (Tensor) - The loss scale value, must be a scalar tensor with float data type.
          Default: 1.0.
        - **variable** (Parameter) - Weights to be updated. data type must be float.
        - **accumulation** (Parameter) - Accumulated gradient value by moment weight.
          Has the same data type with `variable`.
        - **learning_rate** (Union[Number, Tensor]) - The learning rate value, must be a float number or
          a scalar tensor with float data type.
        - **gradient** (Tensor) - Gradient, has the same data type as `variable`.
        - **momentum** (Union[Number, Tensor]) - Momentum, must be a float number or
          a scalar tensor with float data type.

    Outputs:
        Tensor, parameters to be updated.

    Supported Platforms:
        ``GPU``
    Examples:
        Please refer to the usage in :class:`mindspore.nn.Momentum`, and add weight_decay and loss_scale as inputs.
    """
    __mindspore_signature__ = (
        sig.make_sig('weight_decay', dtype=sig.sig_dtype.T3),
        sig.make_sig('loss_scale', dtype=sig.sig_dtype.T3),
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accumulation', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('learning_rate', dtype=sig.sig_dtype.T1),
        sig.make_sig('gradient', dtype=sig.sig_dtype.T),
        sig.make_sig('momentum', dtype=sig.sig_dtype.T2)
    )

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['weight_decay', 'loss_scale', 'variable', 'accumulation', 'learning_rate',
                                        'gradient', 'momentum'], outputs=['output'])

    def infer_shape(self, d_shape, s_shape, v_shape, a_shape, l_shape, g_shape, m_shape):
        return v_shape

    def infer_dtype(self, d_dtype, s_dtype, v_dtype, a_dtype, l_dtype, g_dtype, m_dtype):
        """infer dtype"""
        valid_dtypes = [mstype.float16, mstype.float32]
        if v_dtype != mstype.type_refkey and a_dtype != mstype.type_refkey:
            validator.check_tensor_dtype_valid("v", v_dtype, valid_dtypes, self.name)
            validator.check_tensor_dtype_valid("a", a_dtype, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l_dtype": l_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"g_dtype": g_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"m_dtype": m_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"d_dtype": d_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"s_dtype": s_dtype}, valid_dtypes, self.name)
        return v_dtype


class FusedCastAdamWeightDecay(PrimitiveWithInfer):
    r"""
    Updates gradients by the Adaptive Moment Estimation (AdamWeightDecay) algorithm with weight decay. This operator
    incorporates type conversion when parameters are initialized with dtype of float16.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.
    The AdamWeightDecay variant was proposed in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            update = \frac{m}{\sqrt{v} + \epsilon} \\
            update =
            \begin{cases}
                update + weight\_decay * w
                    & \text{ if } weight\_decay > 0 \\
                update
                    & \text{ otherwise }
            \end{cases} \\
            w  = w - lr * update
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`lr` represents `learning_rate`, :math:`w` represents `var`, :math:`decay` represents `weight_decay`,
    :math:`\epsilon` represents `epsilon`.

    Args:
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be updated with the type float16 or float32.
        - **m** (Tensor) - The 1st moment vector in the updating formula with the type float32.
        - **v** (Tensor) - the 2nd moment vector in the updating formula with the type float32.
        - **lr** (float) - :math:`lr` in the updating formula.
        - **beta1** (float) - The exponential decay rate for the 1st moment estimations.
        - **beta2** (float) - The exponential decay rate for the 2nd moment estimations.
        - **epsilon** (float) - Term added to the denominator to improve numerical stability.
        - **decay** (float) - The weight decay value, must be a scalar tensor with float data type.
        - **gradient** (Tensor) - Gradient, has the type float16.

    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **m** (Tensor) - The same shape and data type as `m`.
        - **v** (Tensor) - The same shape and data type as `v`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, Parameter
        >>> from mindspore import dtype as mstype
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.opt = ops.FusedCastAdamWeightDecay()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]), mstype.float16), name="var")
        ...         self.m = Parameter(Tensor(np.ones([2, 2]), mstype.float32), name="m")
        ...         self.v = Parameter(Tensor(np.ones([2, 2]), mstype.float32), name="v")
        ...     def construct(self, lr, beta1, beta2, epsilon, decay, grad, norm):
        ...         out = self.opt(self.var, self.m, self.v, lr, beta1, beta2, epsilon, decay, grad, norm)
        ...         return out
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
        >>> net = Net()
        >>> gradient = Tensor(np.ones([2, 2]), mstype.float16)
        >>> output = net(0.001, 0.9, 0.999, 1e-8, 0.0, gradient, 1.0)
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        self.add_prim_attr('side_effect_mem', True)
        validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def infer_shape(self, var_shape, m_shape, v_shape, lr_shape, beta1_shape, beta2_shape,
                    epsilon_shape, decay_shape, grad_shape, global_norm):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "grad_shape", grad_shape, Rel.EQ, self.name)
        return var_shape, m_shape, v_shape

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, lr_dtype, beta1_dtype, beta2_dtype,
                    epsilon_dtype, decay_dtype, grad_dtype, global_norm):
        """infer dtype"""
        args = {"m": m_dtype, "v": v_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        validator.check_scalar_or_tensor_types_same({"var": var_dtype}, [mstype.float16, mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"grad": grad_dtype}, [mstype.float16, mstype.float32], self.name)

        args = {"lr": lr_dtype, "beta1": beta1_dtype, "beta2": beta2_dtype, "epsilon": epsilon_dtype,
                "decay": decay_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float32], self.name, True)
        return var_dtype, m_dtype, v_dtype


class FusedAdaFactor(PrimitiveWithInfer):
    r"""
    Updates gradients by the Adaptive Learning Rates with Sublinear Memory Cost (Adafactor) algorithm.

    The Adafactor algorithm is proposed in `Adafactor: Adafactor: Adaptive Learning Rates with Sublinear Memory
    Cost <https://arxiv.org/abs/1804.04235>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Adafactor for weight vector are as follows,

    .. math::
        \begin{array}{l} \\
        \alpha_{t}=\max \left(\epsilon_{2}, \operatorname{RMS}\left(X_{t-1}\right)\right) \rho_{t} \\
        G_{t}=\nabla f_{t}\left(X_{t-1}\right) \\
        \hat{V}_{t}=\hat{\beta}_{2} \hat{V}_{t-1}+\left(1-\hat{\beta}_{2_{t}}\right)\left(G_{t}^{2}+ \\
        \epsilon_{1} 1_{n}\right) \\
        U_{t}=G_{t} / \sqrt{\hat{V}_{t}} \\
        \hat{U}_{t}=U_{t} / \max \left(1, \operatorname{RMS}\left(U_{t}\right) / d\right) \\
        X_{t}=X_{t-1}-\alpha_{t} \hat{U}_{t}
        \end{array}

    Adafactor for weight matrices are as follows,

    .. math::
        \begin{array}{l} \\
        \alpha_{t}=\max \left(\epsilon_{2}, \operatorname{RMS}\left(X_{t-1}\right)\right) \rho_{t} \\
        G_{t}=\nabla f_{t}\left(X_{t-1}\right) \\
        R_{t}=\hat{\beta}_{2 t} R_{t-1}+\left(1-\hat{\beta}_{2 t}\right)\left(G_{t}^{2}+ \\
        \epsilon_{1} 1_{n} 1_{m}^{\top}\right) 1_{m} \\
        C_{t}=\hat{\beta}_{2 t} C_{t-1}+\left(1-\hat{\beta}_{2 t}\right) 1_{n}^{\top}\left(G_{t}^{2}+ \\
        \epsilon_{1} 1_{n} 1_{m}^{\top}\right) \\
        \hat{V}_{t}=R_{t} C_{t} / 1_{n}^{\top} R_{t} \\
        U_{t}=G_{t} / \sqrt{\hat{V}_{t}} \\
        \hat{U}_{t}=U_{t} / \max \left(1, \operatorname{RMS}\left(U_{t}\right) / d\right) \\
        X_{t}=X_{t-1}-\alpha_{t} U_{t}
        \end{array}

    Where RMS is:

    .. math::
        \operatorname{RMS}\left(U_{t}\right)=\operatorname{RMS}_{x \in X}\left(u_{x t}\right)= \\
        \sqrt{\operatorname{Mean}_{x \in X}\left(\frac{\left(g_{x t}\right)^{2}}{\hat{v}_{x t}}\right)}

    :math:`x` is each individual parameter,
    :math:`t` is assumed to be the current number of steps,
    :math:`a_{t}` is the learning rate,
    :math:`f(X)` is the loss function,
    :math:`\epsilon1` and :math:`\epsilon2` is a small positive number to prevent errors,
    :math:`d` is the clipping threshold,
    :math:`\beta_{2}` is the moment decay,
    :math:`\rho` is the relative step size,
    :math:`R` is the running averages of the row sums of the squared gradient,
    :math:`C` is the running averages of the column sums of the squared gradient.

    Args:
        enable_weight_decay (bool): If True, enable weight decay. default: False
        enable_first_moment (bool): If True, enable first moment. default: False
        enable_scale_parameter (bool): If True, enable scale learning rate using parameter. default: False

    Inputs:
        - **epsilon** (Tensor) - input epsilon pair.
        - **clip_threshold** (float) - The threshold of root mean square of final gradient update.
        - **beta1** (float) - The exponential decay rate for the 1nd moment estimations.
        - **beta2** (float) - The exponential decay rate for the 2nd moment estimations.
        - **weight_decay** (float) - The weight decay value, must be a scalar tensor with float data type.
        - **learning_rate** (float) - The learning rate value.
        - **gradient** (Tensor) - Gradient.
        - **param** (Tensor) - Weights to be updated.
        - **exp_avg** (Tensor) - The exponential moving average of 1st moment optimizer state.
        - **exp_avg_sq_row** (Tensor) - The exponential moving average of square of gradient square row factor.
        - **exp_avg_sq_col** (Tensor) - The exponential moving average of square of gradient square col factor.
        - **exp_avg_sq** (Tensor) - The exponential moving average of square of gradient square.

    Outputs:
        - **dummy_param** (Tensor) - The same shape and data type as `param`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, Parameter
        >>> from mindspore import dtype as mstype
        >>> param_shape = [2, 3, 2]
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.opt = ops.FusedAdaFactor()
        ...         self.param = Parameter(Tensor(np.ones(param_shape), mstype.float32), name="param")
        ...         self.exp_avg = Parameter(Tensor(np.zeros(param_shape), mstype.float32), name="exp_avg")
        ...         self.exp_avg_sq = Parameter(Tensor(np.zeros(param_shape), mstype.float32), name="exp_avg_sq")
        ...         self.exp_avg_sq_row = Parameter(Tensor(np.zeros([2, 3]), mstype.float32), name="exp_avg_sq_row")
        ...         self.exp_avg_sq_col = Parameter(Tensor(np.zeros([2, 2]), mstype.float32), name="exp_avg_sq_col")
        ...
        ...     def construct(self, epsilon, clip_threshold, beta1, beta2, weight_decay, lr, grad):
        ...         out = self.opt(epsilon, clip_threshold, beta1, beta2, weight_decay, lr, grad, self.param,
        ...                        self.exp_avg, self.exp_avg_sq_row, self.exp_avg_sq_col, self.exp_avg_sq)
        ...         return out
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
        >>> net = Net()
        >>> gradient = Tensor(np.ones(param_shape), mstype.float32)
        >>> output = net((1e-30, 1e-3), 1.0, 0.9, 0.8, 1e-2, 0.03, gradient)
    """

    @prim_attr_register
    def __init__(self, enable_scale_parameter=False, enable_first_moment=False, enable_weight_decay=False):
        self.add_prim_attr('side_effect_mem', True)
        validator.check_value_type("enable_scale_parameter", enable_scale_parameter, [bool], self.name)
        validator.check_value_type("enable_first_moment", enable_first_moment, [bool], self.name)
        validator.check_value_type("enable_weight_decay", enable_weight_decay, [bool], self.name)

    def infer_shape(self, epsilon_shape, clip_threshold_shape, beta1_shape, beta2t_shape, weight_decay_shape,
                    learning_rate_shape, grad_shape, param_shape, exp_avg_shape, exp_avg_sq_row_shape,
                    exp_avg_sq_col_shape, exp_avg_sq_shape):
        validator.check("grad_shape", grad_shape, "param_shape", param_shape, Rel.EQ, self.name)
        return param_shape

    def infer_dtype(self, epsilon_type, clip_threshold_type, beta1_type, beta2t_type, weight_decay_type,
                    learning_rate_type, grad_type, param_type, exp_avg_type, exp_avg_sq_row_type,
                    exp_avg_sq_col_type, exp_avg_sq_type):
        return param_type


class FusedAdaFactorWithGlobalNorm(FusedAdaFactor):
    r"""
    Divide global norm for gradient in FusedAdaFactor, and refer to super class for FusedAdaFactor details
    """

    @prim_attr_register
    def __init__(self, enable_scale_parameter=False, enable_first_moment=False, enable_weight_decay=False):
        super(FusedAdaFactorWithGlobalNorm, self).__init__(enable_scale_parameter, enable_first_moment,
                                                           enable_weight_decay)

    def infer_shape(self, epsilon_shape, clip_threshold_shape, beta1_shape, beta2t_shape, weight_decay_shape,
                    learning_rate_shape, grad_shape, param_shape, exp_avg_shape, exp_avg_sq_row_shape,
                    exp_avg_sq_col_shape, exp_avg_sq_shape, global_norm_shape):
        validator.check("grad_shape", grad_shape, "param_shape", param_shape, Rel.EQ, self.name)
        return param_shape

    def infer_dtype(self, epsilon_type, clip_threshold_type, beta1_type, beta2t_type, weight_decay_type,
                    learning_rate_type, grad_type, param_type, exp_avg_type, exp_avg_sq_row_type,
                    exp_avg_sq_col_type, exp_avg_sq_type, global_norm_type):
        return param_type


class GenerateEodMask(Primitive):
    r"""
    Given the input `inputs_ids`, if found eod_token_id, the output position and attention mask matrix will be reset.
    This means the `position_id` will start counting from 0, and the corresponding mask matrix will be filled with 0.

    Args:
        eod_token_id (int) - In the NLP scenario, this value corresponds to the id of
            the symbol of 'EodOfDocument' in the vocabulary.

    Inputs:
      - **inputs_ids** (Tensor) - token id, a 2-D Tensor with shape :math:`(batch\_size, seq\_length)`.

    Outputs:
      - **position_id** (Tensor) - position id matrix with same shape and type as original `inputs_ids`.
      - **attention_mask** (Tensor) - attention mask matrix with type
            float16 and shape :math:`(batch\_size, seq\_length)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> op = ops.GenerateEodMask(eod_token_id=0)
        >>> position, mask = op(Tensor([[1, 0, 3], [1, 0, 0]], dtype=mindspore.int32))
        >>> print(position)
        [[0 1 0] [0 0 1]]
        >>> print(mask)
        [[[ 1. 0. 0.]
          [1. 1. 0.]
          [0. 0. 1.]]
         [[1. 0. 0.]
          [0. 1. 0.]
          [0. 1. 1.]]]

    Raises:
        - **TypeError** - If `eod_token_id` is not int.
        - **TypeError** - If `inputs_ids` is not int.
        - **ValueError** - If `inputs_ids` is not a 2-D Tensor.
    """
    @prim_attr_register
    def __init__(self, eod_token_id):
        """Initialize GenerateEodMask"""
        validator.check_value_type("eod_token_id", eod_token_id, [int], self.name)
        self.init_prim_io_names(inputs=['inputs_ids'],
                                outputs=['position_ids', 'attention_mask'])


class ScaleGrad(PrimitiveWithInfer):
    """
    Scale the input grad according to the loss scale.

    Inputs:
        - **gradients** (list of Tensor or tupe of Tensor).

    Outputs:
        tuple[Tensor], the shape of each output tensor is the same to the input gradient.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScaleGrad"""
