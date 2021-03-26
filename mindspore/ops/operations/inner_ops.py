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
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ...common.dtype import tensor, dtype_to_pytype
from ..primitive import prim_attr_register, Primitive, PrimitiveWithInfer


class ScalarCast(PrimitiveWithInfer):
    """
    Casts the input scalar to another type.

    Inputs:
        - **input_x** (scalar) - The input scalar. Only constant value is allowed.
        - **input_y** (mindspore.dtype) - The type to be cast. Only constant value is allowed.

    Outputs:
        Scalar. The type is the same as the python type corresponding to `input_y`.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a constant value.

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


class Randperm(PrimitiveWithInfer):
    """
    Generates n random samples from 0 to n-1 without repeating. If `max_length` > n,
    the last `max_length-n` elements will be filled with `pad`.

    Args:
        max_length (int): Number of items expected to get and the number must be greater than 0. Default: 1.
        pad (int): The pad value to be filled. Default: -1.
        dtype (mindspore.dtype): The type of output. Default: mindspore.int32.

    Inputs:
        - **n** (Tensor[int]) - The input tensor with shape: (1,) and the number must be in (0, `max_length`].
          Default: 1.

    Outputs:
        - **output** (Tensor) - The output Tensor with shape: (`max_length`,) and type: `dtype`.

    Raises:
        TypeError: If neither `max_length` nor `pad` is an int.
        TypeError: If `n` is not a Tensor.
        TypeError: If `n` has non-Int elements.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> randperm = ops.Randperm(max_length=30, pad=-1)
        >>> n = Tensor([20], dtype=mindspore.int32)
        >>> output = randperm(n)
        >>> print(output)
        [15 6 11 19 14 16 9 5 13 18 4 10 8 0 17 2 14 1 12 3 7
         -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
    """

    @prim_attr_register
    def __init__(self, max_length=1, pad=-1, dtype=mstype.int32):
        """Initialize Randperm"""
        validator.check_value_type("pad", pad, [int], self.name)
        validator.check_value_type("max_length", max_length, [int], self.name)
        validator.check_int(max_length, 1, Rel.GE, "1", self.name)
        self.dtype = dtype
        self.max_length = max_length
        self.init_prim_io_names(inputs=[], outputs=['output'])

    def infer_shape(self, n_shape):
        validator.check_int(len(n_shape), 1, Rel.EQ, "rank_of_n", self.name)
        validator.check_int(n_shape[0], 1, Rel.EQ, "length_of_n", self.name)
        return [self.max_length]

    def infer_dtype(self, n_type):
        validator.check_type_name("n_type", n_type, mstype.int32, self.name)

        valid_values = (mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                        mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64)
        validator.check_type_name("dtype", self.dtype, valid_values, self.name)
        return self.dtype


class NoRepeatNGram(PrimitiveWithInfer):
    """
    Updates log_probs with repeat n-grams.

    During beam search, if consecutive `ngram_size` words exist in the generated word sequence,
    the consecutive `ngram_size` words will be avoided during subsequent prediction.
    For example, when `ngram_size` is 3, the generated word sequence is [1, 2, 3, 2, 3],
    the next predicted word will not be 2 and the value of `log_probs` will be replaced with -FLOAT_MAX.
    Because 3 consecutive words [2, 3, 2] do not appear twice in the word sequence.

    Args:
        ngram_size (int): Size of n-grams, must be greater than 0. Default: 1.

    Inputs:
        - **state_seq** (Tensor) - A 3-D tensor with shape: (batch_size, beam_width, m).
        - **log_probs** (Tensor) - A 3-D tensor with shape: (batch_size, beam_width, vocab_size).
          The value of log_probs will be replaced with -FLOAT_MAX when n-grams repeated.

    Outputs:
        - **log_probs** (Tensor) - The output Tensor with same shape and type as original `log_probs`.

    Raises:
        TypeError: If `ngram_size` is not an int.
        TypeError: If neither `state_seq` nor `log_probs` is a Tensor.

    Supported Platforms:
        ``Ascend``

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

    def infer_shape(self, seq_shape, log_shape):
        validator.check_int(len(seq_shape), 3, Rel.EQ, "rank of state_seq", self.name)
        validator.check_int(len(log_shape), 3, Rel.EQ, "rank of log_probs", self.name)
        validator.check("state_seq shape[0]", seq_shape[0], "log_probs shape[0]", log_shape[0], Rel.EQ, self.name)
        validator.check("state_seq shape[1]", seq_shape[1], "log_probs shape[1]", log_shape[1], Rel.EQ, self.name)
        validator.check("ngram_size", self.ngram_size, "state_seq shape[2] + 1", seq_shape[2] + 1, Rel.LE, self.name)
        return log_shape

    def infer_dtype(self, seq_type, log_type):
        validator.check_type_name("seq_type", seq_type, mstype.int32, self.name)
        valid_values = (mstype.float16, mstype.float32, mstype.float64)
        validator.check_type_name("log_type", log_type, valid_values, self.name)
        return log_type


class LambApplyOptimizerAssign(PrimitiveWithInfer):
    r"""
    Updates gradients by LAMB optimizer algorithm. Get the compute ratio.

    The Lamb optimzier is proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
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

    The Lamb optimzier is proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
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
        - **update** (Tensor) -:math:`r + \lambda * w`in the updating formula, float32/float16.
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


class MakeRefKey(Primitive):
    """
    Makes a RefKey instance by string. RefKey stores the name of Parameter, can be passed through the functions,
    and used for Assign target.

    Args:
        tag (str): Parameter name to make the RefKey.

    Inputs:
        No inputs.

    Outputs:
        RefKeyType, made from the Parameter name.

    Raises:
        TypeError: If `tag` is not a str.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Parameter, Tensor
        >>> from mindspore import dtype as mstype
        >>> import mindspore.ops as ops
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.y = Parameter(Tensor(np.ones([2, 3]), mstype.int32), name="y")
        ...         self.make_ref_key = ops.MakeRefKey("y")
        ...
        ...     def construct(self, x):
        ...         key = self.make_ref_key()
        ...         ref = ops.make_ref(key, x, self.y)
        ...         return ref * x
        ...
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.int32)
        >>> net = Net()
        >>> output = net(x)
        >>> print(output)
        [[ 1  4  9]
         [16 25 36]]
    """

    @prim_attr_register
    def __init__(self, tag):
        validator.check_value_type('tag', tag, (str,), self.name)

    def __call__(self):
        pass
