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
"""Operations for random number generators."""
from mindspore.ops.primitive import constexpr
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common import dtype as mstype
from mindspore.common.seed import _get_graph_seed
from mindspore.common.api import _function_forbid_reuse


@constexpr(reuse_result=False)
def _get_seed(op_seed, kernel_name):
    "Get the graph-level seed."
    return _get_graph_seed(op_seed, kernel_name)


@_function_forbid_reuse
def normal(shape, mean, stddev, seed=None):
    """
    Generates random numbers according to the Normal (or Gaussian) random number distribution.

    Args:
        shape (tuple): The shape of random tensor to be generated.
          The format is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        mean (Tensor): The mean μ distribution parameter, which specifies the location of the peak,
          with data type in [int8, int16, int32, int64, float16, float32].
        stddev (Tensor): The deviation σ distribution parameter. It should be greater than 0,
          with data type in [int8, int16, int32, int64, float16, float32].
        seed (int): Seed is used as entropy source for the Random number engines to generate pseudo-random numbers.
          The value must be non-negative. Default: None, which will be treated as 0.

    Returns:
        Tensor. The shape should be equal to the broadcasted shape between the input `shape` and shapes
        of `mean` and `stddev`.
        The dtype is float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> shape = (3, 1, 2)
        >>> mean = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
        >>> stddev = Tensor(1.0, mindspore.float32)
        >>> output = ops.normal(shape, mean, stddev, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
        >>> shape = (3, 1, 3)
        >>> mean = Tensor(np.array([[3, 4, 3], [3, 5, 6]]), mindspore.float32)
        >>> stddev = Tensor(1.0, mindspore.float32)
        >>> output = ops.normal(shape, mean, stddev, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 3)
        >>> shape = (3, 1, 3)
        >>> mean = Tensor(np.array([[1, 2, 3], [3, 4, 3], [3, 5, 6]]), mindspore.float32)
        >>> stddev = Tensor(1.0, mindspore.float32)
        >>> output = ops.normal(shape, mean, stddev, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 3, 3)
    """
    mean_dtype = F.dtype(mean)
    stddev_dtype = F.dtype(stddev)
    const_utils.check_type_valid(mean_dtype, mstype.int_type + (mstype.float16, mstype.float32), 'normal')
    const_utils.check_type_valid(stddev_dtype, mstype.int_type + (mstype.float16, mstype.float32), 'normal')
    seed1, seed2 = _get_seed(seed, "normal")
    stdnormal = P.StandardNormal(seed1, seed2)
    _check_shape(shape)
    random_normal = stdnormal(shape)
    value = random_normal * stddev + mean
    return value


@_function_forbid_reuse
def laplace(shape, mean, lambda_param, seed=None):
    r"""
    Generates random numbers according to the Laplace random number distribution.
    It is defined as:

    .. math::
        \text{f}(x;μ,λ) = \frac{1}{2λ}\exp(-\frac{|x-μ|}{λ}),

    Args:
        shape (tuple): The shape of random tensor to be generated.
          The format is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        mean (Tensor): The mean μ distribution parameter, which specifies the location of the peak.
          With float32 data type.
        lambda_param (Tensor): The parameter used for controlling the variance of this random distribution. The
          variance of Laplace distribution is equal to twice the square of lambda_param. With float32 data type.
        seed (int): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: None, which will be treated as 0.

    Returns:
        Tensor. The shape should be the broadcasted shape of input `shape` and shapes of `mean` and `lambda_param`.
        The dtype is float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import ops as ops
        >>> shape = (2, 3)
        >>> mean = Tensor(1.0, mindspore.float32)
        >>> lambda_param = Tensor(1.0, mindspore.float32)
        >>> output = ops.laplace(shape, mean, lambda_param, seed=5)
        >>> print(output.shape)
        (2, 3)
    """
    mean_dtype = F.dtype(mean)
    lambda_param_dtype = F.dtype(lambda_param)
    const_utils.check_tensors_dtype_same(mean_dtype, mstype.float32, "laplace")
    const_utils.check_tensors_dtype_same(lambda_param_dtype, mstype.float32, "laplace")
    seed1, seed2 = _get_seed(seed, "laplace")
    stdlaplace = P.StandardLaplace(seed1, seed2)
    _check_shape(shape)
    rnd = stdlaplace(shape)
    value = rnd * lambda_param + mean
    return value


@_function_forbid_reuse
def uniform(shape, minval, maxval, seed=None, dtype=mstype.float32):
    """
    Generates random numbers according to the Uniform random number distribution.

    Note:
        The number in tensor minval should be strictly less than maxval at any position after broadcasting.

    Args:
        shape (tuple): The shape of random tensor to be generated.
          The format is :math:`(N,*)` where :math:`*` means, any number of additional dimensions
          and the length of :math:`(N,*)` should be less than 8 in broadcast operation.
        minval (Tensor): The distribution parameter `a`.
          It defines the minimum possible generated value, with int32 or float32 data type.
          If dtype is int32, only one number is allowed.
        maxval (Tensor): The distribution parameter `b`.
          It defines the maximum possible generated value, with int32 or float32 data type.
          If dtype is int32, only one number is allowed.
        seed (int): Seed is used as entropy source for the random number engines to generate pseudo-random numbers,
          must be non-negative. Default: None, which will be treated as 0.
        dtype (mindspore.dtype): Type of the Uniform distribution. If it is int32, it generates numbers from discrete
          uniform distribution; if it is float32, it generates numbers from continuous uniform distribution. It only
          supports these two data types. Default: mindspore.float32.

    Returns:
        Tensor. The shape should be equal to the broadcasted shape between the input `shape` and shapes
        of `minval` and `maxval`.
        The dtype is designated as the input `dtype`.

    Raises:
        TypeError: If `shape` is not tuple.
        TypeError: If 'minval' or 'maxval' is neither int32 nor float32
            and dtype of 'minval' is not the same as 'maxval'.
        TypeError: If `seed` is not an int.
        TypeError: If 'dtype' is neither int32 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> # For discrete uniform distribution, only one number is allowed for both minval and maxval:
        >>> shape = (4, 2)
        >>> minval = Tensor(1, mindspore.int32)
        >>> maxval = Tensor(2, mindspore.int32)
        >>> output = ops.uniform(shape, minval, maxval, seed=5, dtype=mindspore.int32)
        >>>
        >>> # For continuous uniform distribution, minval and maxval can be multi-dimentional:
        >>> shape = (3, 1, 2)
        >>> minval = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
        >>> maxval = Tensor([8.0, 10.0], mindspore.float32)
        >>> output = ops.uniform(shape, minval, maxval, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
    """
    value = F.uniform(shape, minval, maxval, seed, dtype)
    return value


@_function_forbid_reuse
def gamma(shape, alpha, beta, seed=None):
    """
    Generates random numbers according to the Gamma random number distribution.

    Args:
        shape (tuple): The shape of random tensor to be generated.
          The format is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        alpha (Tensor): The alpha α distribution parameter. It should be greater than 0 with float32 data type.
        beta (Tensor): The beta β distribution parameter. It should be greater than 0 with float32 data type.
        seed (int): Seed is used as entropy source for the random number engines to generate
          pseudo-random numbers, must be non-negative. Default: None, which will be treated as 0.

    Returns:
        Tensor. The shape should be equal to the broadcasted shape between the input `shape` and shapes
        of `alpha` and `beta`.
        The dtype is float32.

    Raises:
        TypeError: If `shape` is not a tuple.
        TypeError: If neither `alpha` nor `beta` is a Tensor.
        TypeError: If `seed` is not an int.
        TypeError: If dtype of `alpha` and `beta` is not float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> # case 1: alpha_shape is (2, 2)
        >>> shape = (3, 1, 2)
        >>> alpha = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
        >>> beta = Tensor(np.array([1.0]), mindspore.float32)
        >>> output = ops.gamma(shape, alpha, beta, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
        >>> # case 2: alpha_shape is (2, 3), so shape is (3, 1, 3)
        >>> shape = (3, 1, 3)
        >>> alpha = Tensor(np.array([[1, 3, 4], [2, 5, 6]]), mindspore.float32)
        >>> beta = Tensor(np.array([1.0]), mindspore.float32)
        >>> output = ops.gamma(shape, alpha, beta, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 3)
        >>> # case 3: beta_shape is (1, 2), the output is different.
        >>> shape = (3, 1, 2)
        >>> alpha = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
        >>> beta = Tensor(np.array([1.0, 2]), mindspore.float32)
        >>> output = ops.gamma(shape, alpha, beta, seed=5)
        >>> result = output.shape
        >>> print(output)
        [[[ 2.2132034  5.8855834]]
         [ 3.3981476  7.5805717]
        [[ 3.3981476  7.5805717]]
         [ 3.7190282 19.941492]
        [[ 2.9512358  2.5969937]]
         [ 3.786061   5.160872 ]]]
        >>> # case 4: beta_shape is (2, 1), the output is different.
        >>> shape = (3, 1, 2)
        >>> alpha = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
        >>> beta = Tensor(np.array([[1.0], [2.0]]), mindspore.float32)
        >>> output = ops.gamma(shape, alpha, beta, seed=5)
        >>> result = output.shape
        >>> print(output)
        [[[ 5.6085486  7.8280783]]
         [ 15.97684  16.116285]
        [[ 1.8347423  1.713663]]
         [ 3.2434065 15.667398]
        [[ 4.2922077  7.3365674]]
         [ 5.3876944  13.159832 ]]]
    """
    seed1, seed2 = _get_seed(seed, "gamma")
    gamma_v = P.Gamma(seed1, seed2)
    value = gamma_v(shape, alpha, beta)
    return value


@_function_forbid_reuse
def poisson(shape, mean, seed=None):
    r"""
    The ops.poisson is deprecated, please use :class:`mindspore.ops.random_poisson`
    Generates random numbers according to the Poisson random number distribution.

    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    Args:
        shape (tuple): The shape of random tensor to be generated.
          The format is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        mean (Tensor): The mean μ distribution parameter. It should be greater than 0 with float32 data type.
        seed (int): Seed is used as entropy source for the random number engines to generate pseudo-random numbers
          and must be non-negative. Default: None, which will be treated as 0.

    Returns:
        Tensor. The shape should be equal to the broadcasted shape between the input "shape" and shapes of `mean`.
        The dtype is float32.

    Raises:
        TypeError: If `shape` is not a tuple.
        TypeError: If `mean` is not a Tensor whose dtype is not float32.
        TypeError: If `seed` is not an int.

    Supported Platforms:
        deprecated

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> # case 1: It can be broadcast.
        >>> shape = (4, 1)
        >>> mean = Tensor(np.array([5.0, 10.0]), mindspore.float32)
        >>> output = ops.poisson(shape, mean, seed=5)
        >>> result = output.shape
        >>> print(result)
        (4, 2)
        >>> # case 2: It can not be broadcast. It is recommended to use the same shape.
        >>> shape = (2, 2)
        >>> mean = Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), mindspore.float32)
        >>> output = ops.poisson(shape, mean, seed=5)
        >>> result = output.shape
        >>> print(result)
        (2, 2)
    """
    seed1, seed2 = _get_seed(seed, "poisson")
    random_poisson = P.Poisson(seed1, seed2)
    value = random_poisson(shape, mean)
    return value


@_function_forbid_reuse
def multinomial(inputs, num_sample, replacement=True, seed=None):
    r"""
    Returns a tensor sampled from the multinomial probability distribution located in the corresponding
    row of the input tensor.

    Note:
        The rows of input do not need to sum to one (in which case we use the values as weights),
        but must be non-negative, finite and have a non-zero sum.

    Args:
        inputs (Tensor): The input tensor containing probabilities, must be 1 or 2 dimensions, with
          float32 data type.
        num_sample (int): Number of samples to draw.
        replacement (bool, optional): Whether to draw with replacement or not, default True.
        seed (int, optional): Seed is used as entropy source for the random number engines to generate
          pseudo-random numbers, must be non-negative. Default: None.

    Returns:
        Tensor, has the same rows with input. The number of sampled indices of each row is `num_samples`.
        The dtype is float32.

    Raises:
        TypeError: If `x` is not a Tensor whose dtype is not float32.
        TypeError: If `num_sample` is not an int.
        TypeError: If `seed` is neither an int nor an optional.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> # case 1: The output is random, and the length of the output is the same as num_sample.
        >>> x = Tensor([0, 9, 4, 0], mindspore.float32)
        >>> output = ops.multinomial(x, 2)
        >>> # print(output)
        >>> # [1 2] or [2 1]
        >>> # the case where the result is [2 1] in multiple times.
        >>> # This is because the value corresponding to the index 1 is larger than the value of the index 2.
        >>> print(len(output))
        2
        >>> # case 2: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is False(Default).
        >>> # If the extracted value is 0, the index value of 1 will be returned.
        >>> x = Tensor([0, 9, 4, 0], mstype.float32)
        >>> output = ops.multinomial(x, 4)
        >>> print(output)
        [1 1 2 1]
        >>> # case 3: The output is random, num_sample == x_length = 4, and replacement is True,
        >>> # Can extract the same elements。
        >>> x = Tensor([0, 9, 4, 0], mstype.float32)
        >>> output = ops.multinomial(x, 4, True)
        >>> print(output)
        [1 1 2 2]
    """
    shape = P.Shape()
    reshape = P.Reshape()
    const_utils.check_valid_dim(len(shape(inputs)), "multinomial")
    seed1, seed2 = _get_seed(seed, "multinomial")
    if not replacement:
        if shape(inputs)[-1] < num_sample:
            const_utils.raise_value_error("For 'multinomial', the 'num_sample' must be less than "
                                          "the last dimension of input without 'replacement', "
                                          "but got 'num_sample': {} and "
                                          "'replacement': {}".format(num_sample, replacement))
        n_dist = 1
        if len(shape(inputs)) > 1:
            n_dist = shape(inputs)[-2]
        random_uniform = P.UniformReal(seed1, seed2)((n_dist * shape(inputs)[-1],))
        if n_dist != 1:
            random_uniform = reshape(random_uniform, (n_dist, shape(inputs)[-1]))
        vals = P.RealDiv()(P.Log()(random_uniform), inputs + 1e-6)
        _, indices = P.TopK()(vals, num_sample)
        return indices
    return P.Multinomial(seed1, seed2)(inputs, num_sample)


def _check_shape(input_shape):
    """
    Check 'shape' value.
    """
    if not isinstance(input_shape, tuple):
        const_utils.raise_type_error("Type of 'shape' must be tuple, but got: {}".format(type(input_shape)))
    for item in input_shape:
        if not isinstance(item, int):
            const_utils.raise_type_error("Elements of 'shape' must be int, but got: {}".format(type(item)))
        if item < 1:
            const_utils.raise_value_error("Elements of 'shape' must be positive int, but got: {}".format(item))
    return True
