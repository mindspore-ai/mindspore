# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Operators for random."""
from __future__ import absolute_import

from mindspore.common._decorator import deprecated
from mindspore._checkparam import Validator, Rel
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register, Primitive
from mindspore.ops._utils import get_broadcast_shape


class NonDeterministicInts(Primitive):
    r"""
    Generates some integers that match the given type.

    Returns the tensor with the given shape, the random numbers in it drawn from the data range
    that a given type can represent.

    .. warning::
        The value of "shape" must be greater than zero. The output length must be less than 1000000.

    Args:
        dtype (mindspore.dtype): The type of output. Its value must be one of the following types: mindspore.int32
            and mindspore.int64. Default: mindspore.int64.

    Inputs:
        - **shape** (Tensor) - The shape of random tensor to be generated. Its type must be one of the following types:
          mindspore.int32 and mindspore.int64.

    Outputs:
        Tensor. Its shape is spcified by the input `shape`. Its type is spcified by `dtype`.

    Raises:
        TypeError: If `shape` is not a Tensor.
        TypeError: If `dtype` and input tensor type are not allowed.
        ValueError: If `shape` has negative elements.
        ValueError: If `shape` has less than 2 elements.
        ValueError: If `shape` is not a 1-D tensor.
        ValueError: If the number of elements of output is more than 1000000.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> shape = Tensor(np.array([2,2]), mstype.int32)
        >>> ndints = ops.NonDeterministicInts(dtype=mstype.int32)
        >>> output = ndints(shape)
        >>> print(output)
        [[13031056   -141954883 ]
         [ 140364228  290834494 ]]
    """

    @prim_attr_register
    def __init__(self, dtype=mstype.int64):
        """Initialize NonDeterministicInts"""
        self.dtype = dtype
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=["shape"], outputs=["output"])
        valid_values = (mstype.int32, mstype.int64)
        Validator.check_type_name("dtype", dtype, valid_values, self.name)


class TruncatedNormal(Primitive):
    """
    Returns a tensor of the specified shape filled with truncated normal values.

    The generated values follow a normal distribution.

    .. warning::
        The value of "shape" must be greater than zero. The output length must be less than 1000000.

    Args:
        seed (int): An optional int. Defaults to 0. If either `seed` or `seed2` are set to be non-zero,
            the seed is set by the given seed. Otherwise, it is seeded by a random seed.
        seed2 (int): An optional int. Defaults to 0. A second seed to avoid seed collision.
        dtype (mindspore.dtype): Must be one of the following types: mindspore.float16, mindspore.float32 and
            mindspore.float64. Default: mindspore.float32.

    Inputs:
        - **shape** (Tensor) - The shape of random tensor to be generated. Its type must be one of the following types:
          mindspore.int32 and mindspore.int64.

    Outputs:
        Tensor. Its shape is spcified by the input `shape`. Its type is spcified by `dtype`.
        Its values are in [-2,2].

    Raises:
        TypeError: If `shape` is not a Tensor.
        TypeError: If `dtype` and input tensor type are not allowed.
        TypeError: If `Seed` is not an integer.
        ValueError: If `shape` elements are not positive.
        ValueError: If `shape` is not a 1-D tensor.
        ValueError: If the number of elements of output is more than 1000000.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = Tensor(np.array([2, 2]), mstype.int32)
        >>> seed = 0
        >>> seed2 = 0
        >>> truncated_normal = ops.TruncatedNormal(seed=seed, seed2=seed2)
        >>> output = truncated_normal(shape)
        >>> print(output)
        [[ -1.303105  0.641905 ]
         [ -0.917926  0.650655 ]]
    """

    @prim_attr_register
    def __init__(self, dtype=mstype.float32, seed=0, seed2=0):
        """Initialize TruncatedNormal"""
        self.dtype = dtype
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=["shape"], outputs=["output"])
        Validator.check_value_type('seed', seed, [int], self.name)
        Validator.check_value_type('seed2', seed2, [int], self.name)
        valid_values = (mstype.float16, mstype.float32, mstype.float64)
        Validator.check_type_name("dtype", dtype, valid_values, self.name)


class StandardNormal(Primitive):
    r"""
    Generates random numbers according to the standard Normal (or Gaussian) random number distribution.

    Refer to :func:`mindspore.ops.standard_normal` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> shape = (3, 4)
        >>> stdnormal = ops.StandardNormal(seed=2)
        >>> output = stdnormal(shape)
        >>> print(output)
        [[-1.3031056   0.64198005 -0.65207404 -1.767485  ]
         [-0.91792876  0.6508565  -0.9098478  -0.14092612]
         [ 0.7806437   1.1585592   1.9676613  -0.00440959]]
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize StandardNormal"""
        self.init_prim_io_names(inputs=['shape'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)


class StandardLaplace(Primitive):
    r"""
    Generates random numbers according to the Laplace random number distribution (mean=0, lambda=1).
    It is defined as:

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|),

    Args:
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Inputs:
        - **shape** (Union[tuple, Tensor]) - The shape of random tensor to be generated. Only constant value is allowed
          when the input type is tuple. And the operator supports dynamic shape only when the input type is Tensor.

    Outputs:
        Tensor. The shape that the input 'shape' denotes. The dtype is float32.

    Raises:
        TypeError: If seed or seed2 is not an int.
        TypeError: If shape is neither a tuple nor a Tensor.
        ValueError: If seed or seed2 is not a non-negative int.
        ValueError: If shape is a tuple containing non-positive items.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = (4, 16)
        >>> stdlaplace = ops.StandardLaplace(seed=2)
        >>> output = stdlaplace(shape)
        >>> result = output.shape
        >>> print(result)
        (4, 16)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize StandardLaplace"""
        self.init_prim_io_names(inputs=['shape'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)


class RandomGamma(Primitive):
    r"""
    Produces random positive floating-point values x, distributed according to probability density function:

    .. note::
        - Random seed: A set of regular random numbers can be obtained through some complex mathematical algorithms,
          and the random seed is the initial value of this random number. If the random seed is the same, the random
          number obtained will not change.
        - Global random seed and operator-level random seed are not set: Use the default value as the random seed.
        - Global random seed is set, but operator-level random seed is not set: A global random seed will splice
          with a randomly generated seed.
        - Global random seed is not set, operator-level random seed is set: The default global random seed is used,
          and splices with the operator-level random seed.
        - Both Global random and operator-level random seed are set: The global random seed will splice with the
          operator-level random seed.

    Args:
        seed (int): The operator-level random seed, used to generate random numbers, must be non-negative. Default: 0.
        seed2 (int): The global random seed and it will combile with the operator-level random seed to determine the
            final generated random number, must be non-negative. Default: 0.

    Inputs:
        - **shape** (Tensor) - The shape of random tensor to be generated.
        Must be one of the following types: int32, int64. 1-D integer tensor.
        - **alpha** (Tensor) - α is the shape parameter of RandomGamma distribution.
        It must be greater than 0. Must be one of the following types: half, float32, float64.

    Outputs:
        Tensor. The shape should be equal to the concat shape between the input `shape` and `alpha`.
        The dtype is the same type as alpha.

    Raises:
        TypeError: If data type of `seed` or `seed2` is not int.
        TypeError: If `shape` or `alpha` is not a Tensor.
        TypeError: If data type of `alpha` is not float32.
        ValueError: If `shape` is not a constant value.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> shape = Tensor(np.array([3, 1, 2]), mstype.int32)
        >>> alpha = Tensor(np.array([[3, 4], [5, 6]]), mstype.float32)
        >>> gamma = ops.RandomGamma(seed=3)
        >>> output = gamma(shape, alpha)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 2, 2, 2)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize Gamma"""
        self.init_prim_io_names(inputs=['shape', 'alpha'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)


class LogNormalReverse(Primitive):
    r"""
    Fills the elements of the input tensor with log normal values initialized by given mean and std:

    .. math::
        \text{f}(x;1.0,2.0)=\frac{1}{x\delta \sqrt[]{2\pi} }e^{-\frac{(\ln x-\mu )^2}{2\delta ^2} }

    Args:
        mean (float): the mean of normal distribution. With float data type. Default: 2.0.
        std (float): the std of normal distribution. With float data type. Default: 1.0.

    Inputs:
        - **input** (Tensor) - The tensor to be generated with log-normal distribution.
        Must be one of the following types: float16, float32.

    Outputs:
        Tensor. A Tensor with the same type and shape of input.

    Raises:
        TypeError: If `input` is not Tensor.
        ValueError: If `input` is NULL.

    Supported Platforms:
        ``Ascend`` ``CPU``


    Examples:
        >>> x = Tensor(np.array([2, 2]), mstype.float32)
        >>> mean = 2.0
        >>> std = 1.0
        >>> lognormalreverse = ops.LogNormalReverse(mean, std)
        >>> output = lognormalreverse(x)
        >>> print(output)
        (3, 1, 2)
    """

    @prim_attr_register
    def __init__(self, mean=2.0, std=1.0):
        """Initialize LogNormalReverse"""
        Validator.check_value_type("mean", mean, [float], self.name)
        Validator.check_value_type("std", std, [float], self.name)


class Gamma(PrimitiveWithInfer):
    r"""
    Produces random positive floating-point values x, distributed according to probability density function:

    .. math::
        \text{P}(x|α,β) = \frac{\exp(-x/β)}{{β^α}\cdot{\Gamma(α)}}\cdot{x^{α-1}}

    .. note::
        - Random seed: A set of regular random numbers can be obtained through some complex mathematical algorithms,
          and the random seed is the initial value of this random number. If the random seed is the same, the random
          number obtained will not change.
        - Global random seed and operator-level random seed are not set: Use the default value as the random seed.
        - Global random seed is set, but operator-level random seed is not set: A global random seed will splice
          with a randomly generated seed.
        - Global random seed is not set, operator-level random seed is set: The default global random seed is used,
          and splices with the operator-level random seed.
        - Both Global random and operator-level random seed are set: The global random seed will splice with the
          operator-level random seed.

    Args:
        seed (int): The operator-level random seed, used to generate random numbers, must be non-negative. Default: 0.
        seed2 (int): The global random seed and it will combile with the operator-level random seed to determine the
            final generated random number, must be non-negative. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **alpha** (Tensor) - α is the shape parameter of Gamma distribution, which mainly determines the shape of
          the curve. It must be greater than 0. The data type is float32.
        - **beta** (Tensor) - β is the inverse scale parameter of the Gamma distribution, which mainly determines how
          steep the curve is. It must be greater than 0. The data type is float32.

    Outputs:
        Tensor. The shape must be the broadcasted shape of Input "shape" and shapes of `alpha` and `beta`.
        The dtype is float32.

    Raises:
        TypeError: If data type of `seed` or `seed2` is not int.
        TypeError: If `alpha` or `beta` is not a Tensor.
        TypeError: If data type of `alpha` or `beta` is not float32.
        ValueError: If `shape` is not a constant value.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> shape = (3, 1, 2)
        >>> alpha = Tensor(np.array([[3, 4], [5, 6]]), mstype.float32)
        >>> beta = Tensor(np.array([1.0]), mstype.float32)
        >>> gamma = ops.Gamma(seed=3)
        >>> output = gamma(shape, alpha, beta)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize RandomGamma"""
        self.init_prim_io_names(inputs=['shape', 'alpha', 'beta'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)

    def __infer__(self, shape, alpha, beta):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For '{self.name}', the 'shape' cannot be None.")
        Validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            Validator.check_positive_int(shape_i, f'shape[{i}]', self.name)
        Validator.check_tensor_dtype_valid("alpha", alpha["dtype"], [mstype.float32], self.name)
        Validator.check_tensor_dtype_valid("beta", beta["dtype"], [mstype.float32], self.name)
        broadcast_shape = get_broadcast_shape(alpha['shape'], beta['shape'], self.name,
                                              arg_name1="alpha", arg_name2="beta")
        broadcast_shape = get_broadcast_shape(broadcast_shape, shape_v, self.name,
                                              arg_name1="broadcast_alpha_beta", arg_name2="shape")
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.float32,
            'value': None}
        return out


class ParameterizedTruncatedNormal(Primitive):
    """
    Returns a tensor of the specified shape filled with truncated normal values.

    When 'shape' is (batch_size, *), the shape of 'mean', 'stdevs', 'min', 'max' should be () or (batch_size, ).

    Note:
        The number in tensor minval must be strictly less than maxval at any position after broadcasting.

    Args:
        seed (int): An optional int. Defaults to 0. If either `seed` or `seed2` are set to be non-zero,
            the seed is set by the given seed. Otherwise, it is seeded by a random seed.
        seed2 (int): An optional int. Defaults to 0. A second seed to avoid seed collision.

    Inputs:
        - **shape** (Tensor) - The shape of random tensor to be generated. Its type must be one of the following types:
          int32 and int64.
        - **mean** (Tensor) - A Tensor. The parameter defines the mean of truncated normal distribution.
          Its type must be one of the following types:float16, float32, float64.
        - **stdevs** (Tensor) - A Tensor. The parameter defines the standard deviation for truncation of
          the normal distribution. It must be greater than 0 and have the same type as means.
        - **min** (Tensor) - The distribution parameter, a. The parameter defines the minimum of
          truncated normal distribution. It must have the same type as means.
        - **max** (Tensor) - The distribution parameter, b. The parameter defines the maximum of
          truncated normal distribution. It must have the same type as means.

    Outputs:
        Tensor. Its shape is spcified by the input `shape` and it must have the same type as means.

    Raises:
        TypeError: If `shape`, `mean`, `stdevs`, `min`, `max` and input tensor type are not allowed.
        TypeError: If `mean`, `stdevs`, `min`, `max` don't have the same type.
        TypeError: If `mean` or `stdevs` or `minval` or `maxval` is not a Tensor.
        ValueError: When 'shape' is (batch_size, *), if the shape of 'mean', 'stdevs', 'min', 'max'
                    is not () or (batch_size, ).
        ValueError: If `shape` elements are not positive.
        ValueError: If `stdevs` elements are not positive.
        ValueError: If `shape` has less than 2 elements.
        ValueError: If `shape` is not a 1-D tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> shape = Tensor(np.array([2, 3]), mstype.int32)
        >>> mean = Tensor(np.array([0], mstype.float32))
        >>> stdevs = Tensor(np.array([1], mstype.float32))
        >>> min = Tensor(np.array([-100], mstype.float32))
        >>> max = Tensor(np.array([100],  mstype.float32))
        >>> seed = 1
        >>> seed2 = 2
        >>> parameterized_truncated_normal = ops.ParameterizedTruncatedNormal(seed=seed, seed2=seed2)
        >>> output = parameterized_truncated_normal(shape, mean, stdevs, min, max)
        >>> print(output)
        [[-0.54974616 -1.4028727   1.5827523 ]
         [ 0.25759354 -1.9593946  -1.5078077 ]]
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize ParameterizedTruncatedNormal"""
        self.init_prim_io_names(inputs=['shape', 'mean', 'stdevs', 'min', 'max'], outputs=['y'])
        Validator.check_value_type('seed', seed, [int], self.name)
        Validator.check_value_type('seed2', seed2, [int], self.name)


class Poisson(PrimitiveWithInfer):
    r"""
    Produces random non-negative integer values i. Distributed according to discrete probability function:

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **mean** (Tensor) - μ parameter the distribution was constructed with. The parameter defines mean number
          of occurrences of the event. It must be greater than 0. With float32 data type.

    Outputs:
        Tensor. Its shape must be the broadcasted shape of `shape` and the shape of `mean`.
        The dtype is int32.

    Raises:
        TypeError: If neither `seed` nor `seed2` is an int.
        TypeError: If `shape` is not a tuple.
        TypeError: If `mean` is not a Tensor whose dtype is not float32.

    Supported Platforms:
        deprecated

    Examples:
        >>> shape = (4, 1)
        >>> mean = Tensor(np.array([5.0, 10.0]), mstype.float32)
        >>> poisson = ops.Poisson(seed=5)
        >>> output = poisson(shape, mean)
        >>> result = output.shape
        >>> print(result)
        (4, 2)
    """

    @deprecated("2.0", "mindspore.ops.operations.Poisson", False)
    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize Poisson"""
        self.init_prim_io_names(inputs=['shape', 'mean'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)

    def __infer__(self, shape, mean):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For '{self.name}', the 'shape' cannot be None.")
        Validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            Validator.check_positive_int(shape_i, f'shape[{i}]', self.name)
        Validator.check_tensor_dtype_valid("mean", mean["dtype"], [mstype.float32], self.name)
        broadcast_shape = get_broadcast_shape(mean['shape'], shape_v, self.name, arg_name1="mean", arg_name2="shape")
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.int32,
            'value': None}
        return out


class RandomPoisson(Primitive):
    r"""
    Produces random non-negative  values i, distributed according to discrete probability function:

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!},

    Args:
         seed (int): An optional int. Defaults to 0. If either `seed` or `seed2` are set to be non-zero,
            the seed is set by the given seed. Otherwise, it is seeded by a random seed.
         seed2 (int): An optional int. Defaults to 0. A second seed to avoid seed collision.
         dtype (mindspore.dtype): The type of output. Default: mindspore.int64.

    Inputs:
        - **shape** (Tensor) - The shape of random tensor to be generated, 1-D Tensor, whose dtype must be in
                               [int32, int64]
        - **rate** (Tensor) - μ parameter the distribution was constructed with. The parameter defines mean number
          of occurrences of the event. Its type must be in [float16, float32, float64, int32, int64]

    Outputs:
        Tensor. Its shape is (*shape, *rate.shape). Its type is spcified by `dtype`.

    Raises:
        TypeError: If `shape` is not a Tensor or its dtype is not int32 or int64.
        TypeError: If `dtype` is not int32 or int64.
        ValueError: If `shape` is not a 1-D tensor.
        ValueError: If `shape` elements are negative.

    Supported Platforms:
        ``Ascend````GPU````CPU``

    Examples:
        >>> shape = Tensor(np.array([2, 3]), mstype.int32)
        >>> rate = Tensor(np.array([2, 2]), mstype.int32)
        >>> seed = 0
        >>> seed2 = 0
        >>> random_poisson = ops.RandomPoisson(seed=seed, seed2=seed2)
        >>> output = random_poisson(shape,rate)
        >>> print(output.shape)
        (2, 3, 2)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0, dtype=mstype.int64):
        """Initialize Poisson"""
        self.init_prim_io_names(inputs=['shape', 'rate'], outputs=['output'])
        Validator.check_value_type('seed', seed, [int], self.name)
        Validator.check_value_type('seed2', seed2, [int], self.name)
        valid_values = (mstype.int64, mstype.int32, mstype.float16, mstype.float32, mstype.float64)
        Validator.check_type_name("dtype", dtype, valid_values, self.name)


class UniformInt(Primitive):
    r"""
    Produces random integer values i, uniformly distributed on the closed interval [minval, maxval), that is,
    distributed according to the discrete probability function:

    .. math::
        \text{P}(i|a,b) = \frac{1}{b-a+1},

    where the :math:`a` indicates the min distribution parameter,
    the :math:`b` indicates the max distribution parameter.

    Note:
        The number in tensor minval must be strictly less than maxval at any position after broadcasting.

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. A second seed to avoid seed collision. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **minval** (Tensor) - The distribution parameter, a.
          It defines the minimum possibly generated value, with int32 data type. Only one number is supported.
        - **maxval** (Tensor) - The distribution parameter, b.
          It defines the maximum possibly generated value, with int32 data type. Only one number is supported.

    Outputs:
        Tensor. The shape is the same as the input 'shape', and the data type is int32.

    Raises:
        TypeError: If neither `seed` nor `seed2` is an int.
        TypeError: If `shape` is not a tuple.
        TypeError: If neither `minval` nor `maxval` is a Tensor.
        ValueError: If `shape` is not a constant value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = (2, 4)
        >>> minval = Tensor(1, mstype.int32)
        >>> maxval = Tensor(5, mstype.int32)
        >>> uniform_int = ops.UniformInt(seed=10)
        >>> output = uniform_int(shape, minval, maxval)
        >>> result = output.shape
        >>> print(result)
        (2, 4)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize UniformInt"""
        self.init_prim_io_names(inputs=['shape', 'minval', 'maxval'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)


class UniformReal(Primitive):
    r"""
    Produces random floating-point values, uniformly distributed to the interval [0, 1).

    Args:
        seed (int): The operator-level random seed, used to generate random numbers, must be non-negative. Default: 0.
        seed2 (int): The global random seed and it will combile with the operator-level random seed to determine the
            final generated random number, must be non-negative. Default: 0.

    .. note::
        - Global random seed and operator-level random seed are not set: Use the default value as the random seed.
        - Global random seed is set, but operator-level random seed is not set: A global random seed will splice
          with a randomly generated seed.
        - Global random seed is not set, operator-level random seed is set: The default global random seed is used,
          and splices with the operator-level random seed.
        - Both Global random and operator-level random seed are set: The global random seed will splice with the
          operator-level random seed.

    Inputs:
        - **shape** (tuple) - The shape of tensor to be generated. Only constant value is allowed.

    Outputs:
        Tensor. The shape that the input 'shape' denotes. The dtype is float32.

    Raises:
        TypeError: If `seed` or `seed2` is not an int.
        TypeError: If `shape` is not a tuple.
        ValueError: If `shape` is not a constant value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = (2, 2)
        >>> uniformreal = ops.UniformReal(seed=2)
        >>> output = uniformreal(shape)
        >>> result = output.shape
        >>> print(result)
        (2, 2)
    """
    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize UniformReal"""
        self.init_prim_io_names(inputs=['shape'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)


class RandomChoiceWithMask(Primitive):
    """
    Generates a random sample as index tensor with a mask tensor from a given tensor.

    Refer to :func:'mindspore.ops.choice_with_mask' for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> rnd_choice_mask = ops.RandomChoiceWithMask()
        >>> input_x = Tensor(np.ones(shape=[240000, 4]).astype(np.bool))
        >>> output_y, output_mask = rnd_choice_mask(input_x)
        >>> result = output_y.shape
        >>> print(result)
        (256, 2)
        >>> result = output_mask.shape
        >>> print(result)
        (256,)
    """

    @prim_attr_register
    def __init__(self, count=256, seed=0, seed2=0):
        """Initialize RandomChoiceWithMask"""
        Validator.check_value_type("count", count, [int], self.name)
        Validator.check_positive_int(count, "count", self.name)
        Validator.check_value_type('seed', seed, [int], self.name)
        Validator.check_value_type('seed2', seed2, [int], self.name)
        self.add_prim_attr("side_effect_hidden", True)


class RandomCategorical(PrimitiveWithInfer):
    """
    Generates random samples from a given categorical distribution tensor.

    Args:
        dtype (mindspore.dtype): The type of output. Its value must be one of mindspore.int16,
            mindspore.int32 and mindspore.int64. Default: mindspore.int64.

    Inputs:
        - **logits** (Tensor) - The input tensor. 2-D Tensor with shape [batch_size, num_classes].
        - **num_sample** (int) - Number of sample to be drawn. Only constant values is allowed.
        - **seed** (int) - Random seed. Default: 0. Only constant values is allowed.

    Outputs:
        - **output** (Tensor) - The output Tensor with shape [batch_size, num_samples].

    Raises:
        TypeError: If `dtype` is not one of the following: mindspore.int16, mindspore.int32, mindspore.int64.
        TypeError: If `logits` is not a Tensor.
        TypeError: If neither `num_sample` nor `seed` is an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...   def __init__(self, num_sample):
        ...     super(Net, self).__init__()
        ...     self.random_categorical = ops.RandomCategorical(mindspore.int64)
        ...     self.num_sample = num_sample
        ...   def construct(self, logits, seed=0):
        ...     return self.random_categorical(logits, self.num_sample, seed)
        ...
        >>> x = np.random.random((10, 5)).astype(np.float32)
        >>> net = Net(8)
        >>> output = net(Tensor(x))
        >>> result = output.shape
        >>> print(result)
        (10, 8)
    """

    @prim_attr_register
    def __init__(self, dtype=mstype.int64):
        """Initialize RandomCategorical"""
        self.dtype = dtype

        valid_values = (mstype.int32, mstype.int16, mstype.int64)
        Validator.check_type_name("dtype", dtype, valid_values, self.name)
        self.init_prim_io_names(inputs=['logits', 'num_samples', 'seed'],
                                outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)


class Multinomial(Primitive):
    r"""
    Returns a tensor sampled from the multinomial probability distribution located in the corresponding
    row of tensor input.

    Note:
        The rows of input do not need to sum to one (in which case we use the values as weights),
        but must be non-negative, finite and have a non-zero sum.

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. Default: 0.
        dtype(dtype): The type of output, must be int32 or int64. Default: int32.

    Inputs:
        - **x** (Tensor) - the input tensor containing the cumsum of probabilities, must be 1 or 2
          dimensions. Must be one of the following types: float16, float32, float64. CPU and GPU
          supports x 1 or 2 dimensions and Ascend only supports 2 dimensions.
        - **num_samples** (int) - number of samples to draw, must be a nonnegative number.

    Outputs:
        Tensor with the same rows as `x`, each row has num_samples sampled indices.

    Raises:
        TypeError: If neither `seed` nor `seed2` is an int.
        TypeError: If `x` is not a Tensor whose dtype is float16, float32, float64.
        TypeError: If dtype of `num_samples` is not int.
        TypeError: If dtype is not int32 or int64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0., 9., 4., 0.]], mstype.float32)
        >>> multinomial = ops.Multinomial(seed=10)
        >>> output = multinomial(x, 2)
        >>> print(output) # run in CPU
        [[1 1]]
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0, dtype=mstype.int32):
        """Initialize Multinomial."""
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)
        self.init_prim_io_names(inputs=['x', 'num_samples'], outputs=['output'])
        Validator.check_value_type("dtype", dtype, [mstype.Type], self.name)
        valid_values = (mstype.int64, mstype.int32)
        Validator.check_type_name("dtype", dtype, valid_values, self.name)


class UniformCandidateSampler(PrimitiveWithInfer):
    r"""
    Uniform candidate sampler.

    This function samples a set of classes(sampled_candidates) from [0, range_max-1] based on uniform distribution.

    Refer to :func:`mindspore.ops.uniform_candidate_sampler` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sampler = ops.UniformCandidateSampler(1, 3, False, 4, 1)
        >>> output1, output2, output3 = sampler(Tensor(np.array([[1], [3], [4], [6], [3]], dtype=np.int32)))
        >>> print(output1.shape)
        (3,)
        >>> print(output2.shape)
        (5, 1)
        >>> print(output3.shape)
        (3,)
    """

    @prim_attr_register
    def __init__(self, num_true, num_sampled, unique, range_max, seed=0, remove_accidental_hits=False):
        """Initialize UniformCandidateSampler"""
        Validator.check_value_type("num_true", num_true, [int], self.name)
        Validator.check_value_type("num_sampled", num_sampled, [int], self.name)
        Validator.check_value_type("unique", unique, [bool], self.name)
        Validator.check_value_type("range_max", range_max, [int], self.name)
        Validator.check_value_type("seed", seed, [int], self.name)
        Validator.check_value_type("remove_accidental_hits", remove_accidental_hits, [bool], self.name)
        Validator.check("value of num_true", num_true, '', 0, Rel.GT, self.name)
        Validator.check("value of num_sampled", num_sampled, '', 0, Rel.GT, self.name)
        Validator.check("value of range_max", range_max, '', 0, Rel.GT, self.name)
        self.num_true = num_true
        if unique:
            Validator.check('value of num_sampled', num_sampled, "value of range_max", range_max, Rel.LE, self.name)
        Validator.check("value of seed", seed, '', 0, Rel.GE, self.name)
        self.num_sampled = num_sampled

    def infer_dtype(self, true_classes_type):
        Validator.check_subclass("true_classes_type", true_classes_type, mstype.tensor, self.name)
        Validator.check_tensor_dtype_valid("true_classes_type", true_classes_type,
                                           (mstype.int32, mstype.int64), self.name)
        return true_classes_type, mstype.float32, mstype.float32

    def infer_shape(self, true_classes_shape):
        Validator.check("true_class.shape[1]", true_classes_shape[1], "num_true", self.num_true, Rel.EQ, self.name)
        return [self.num_sampled], true_classes_shape, [self.num_sampled]


class LogUniformCandidateSampler(PrimitiveWithInfer):
    r"""
    Generates random labels with a log-uniform distribution for sampled_candidates.

    Randomly samples a tensor of sampled classes from the range of integers [0, range_max).

    Args:
        num_true (int): The number of target classes per training example. Default: 1.
        num_sampled (int): The number of classes to randomly sample. Default: 5.
        unique (bool): Determines whether sample with rejection. If `unique` is True,
          all sampled classes in a batch are unique. Default: True.
        range_max (int): The number of possible classes. When `unique` is True,
          `range_max` must be greater than or equal to `num_sampled`. Default: 5.
        seed (int): Random seed, must be non-negative. Default: 0.

    Inputs:
        - **true_classes** (Tensor) - The target classes. With data type of int64 and
          shape :math:`(batch\_size, num\_true)` .

    Outputs:
        Tuple of 3 Tensors.

        - **sampled_candidates** (Tensor) - A Tensor with shape :math:`(num\_sampled,)`
          and the same type as `true_classes`.
        - **true_expected_count** (Tensor) - A Tensor with the same shape as `true_classes and` type float32.
        - **sampled_expected_count** (Tensor) - A Tensor with the same shape as `sampled_candidates` and type float32.

    Raises:
        TypeError: If neither `num_true` nor `num_sampled` is an int.
        TypeError: If `unique` is not a bool.
        TypeError: If neither `range_max` nor `seed` is an int.
        TypeError: If `true_classes` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> sampler = ops.LogUniformCandidateSampler(2, 5, True, 5)
        >>> output1, output2, output3 = sampler(Tensor(np.array([[1, 7], [0, 4], [3, 3]])))
        >>> print(output1, output2, output3)
        [3 2 0 4 1]
        [[0.92312991 0.49336370]
         [0.99248987 0.65806371]
         [0.73553443 0.73553443]]
        [0.73553443 0.82625800 0.99248987 0.65806371 0.92312991]

    """

    @prim_attr_register
    def __init__(self, num_true=1, num_sampled=5, unique=True, range_max=5, seed=0):
        """Initialize LogUniformCandidateSampler"""
        self.init_prim_io_names(inputs=['true_classes'],
                                outputs=['sampled_candidates', 'true_expected_count', 'sampled_expected_count'])
        Validator.check_value_type("num_true", num_true, [int], self.name)
        Validator.check_value_type("num_sampled", num_sampled, [int], self.name)
        Validator.check_value_type("unique", unique, [bool], self.name)
        Validator.check_value_type("range_max", range_max, [int], self.name)
        Validator.check_value_type("seed", seed, [int], self.name)
        self.num_true = Validator.check_number("num_true", num_true, 1, Rel.GE, self.name)
        self.num_sampled = Validator.check_number("num_sampled", num_sampled, 1, Rel.GE, self.name)
        Validator.check_number("range_max", range_max, 1, Rel.GE, self.name)
        if unique:
            Validator.check("range_max", range_max, "num_sampled", num_sampled, Rel.GE, self.name)
        self.range_max = range_max
        self.unique = unique
        self.seed = Validator.check_number("seed", seed, 0, Rel.GE, self.name)

    def infer_shape(self, true_classes_shape):
        Validator.check_int(len(true_classes_shape), 2, Rel.EQ, "dim of true_classes", self.name)
        Validator.check("true_classes_shape[1]", true_classes_shape[1], "num_true", self.num_true, Rel.EQ, self.name)
        return (self.num_sampled,), true_classes_shape, (self.num_sampled,)

    def infer_dtype(self, true_classes_type):
        Validator.check_subclass("true_classes_type", true_classes_type, mstype.tensor, self.name)
        valid_types = (mstype.int64,)
        Validator.check_tensor_dtype_valid("true_classes_type", true_classes_type, valid_types, self.name)
        expected_type = mstype.float32
        return true_classes_type, expected_type, expected_type


class RandomShuffle(Primitive):
    r"""
    Randomly shuffles a Tensor along its first dimension.

    Args:
        seed (int): Random seed. If `seed` or `seed2` is set to non-zero, the random number generator will be seeded
            by the given seed. Otherwise, it will be seeded randomly. The seed must be non-negative. Default: 0.
        seed2 (int): Random seed2, a second seed to avoid seed collision. If `seed` is 0, the `seed2` will be used as
            the seed of the random generator. It must be non-negative. Default: 0.

    Inputs:
        - **x** (Tensor) - The Tensor need be shuffled.

    Outputs:
        Tensor. The shape and type are the same as the input `x`.

    Raises:
        TypeError: If data type of `seed` or `seed2` is not int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]), mstype.float32)
        >>> shuffle = ops.RandomShuffle(seed=1, seed2=1)
        >>> output = shuffle(x)
        >>> print(output.shape)
        (4,)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize RandomShuffle"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])
        Validator.check_non_negative_int(seed, "seed", self.name)
        Validator.check_non_negative_int(seed2, "seed2", self.name)
