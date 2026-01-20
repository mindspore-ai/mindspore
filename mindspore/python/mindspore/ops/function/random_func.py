# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Defines parameter operators with functional form."""

from __future__ import absolute_import
import numpy as np

from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common import dtype as mstype
from mindspore.common.seed import _get_graph_seed
from mindspore.common.tensor import Tensor
from mindspore.ops.operations.random_ops import RandomShuffle, RandomChoiceWithMask
from mindspore.common.api import _function_forbid_reuse
from mindspore.ops.auto_generate import randperm
from mindspore.common.generator import default_generator
from mindspore.ops.auto_generate import UniformExt, NormalTensorTensor, \
    NormalTensorFloat, NormalFloatTensor, NormalFloatFloat, RandExt, RandLikeExt, MultinomialExt, \
    Randn, RandnLike, RandInt, RandIntLike, RandpermExt, InplaceRandom, InplaceNormal
from mindspore.ops.auto_generate.gen_ops_prim import inplace_uniform_op, inplace_exponential_op

inplace_normal_ = InplaceNormal()
normal_tensor_tensor_op = NormalTensorTensor()
normal_tensor_float_op = NormalTensorFloat()
normal_float_tensor_op = NormalFloatTensor()
normal_float_float_op = NormalFloatFloat()
cast_ = P.Cast()
log_ = P.Log()
real_div_ = P.RealDiv()
reshape_ = P.Reshape()
shape_ = P.Shape()
top_k_ = P.TopK()
randperm_ext_ = RandpermExt()
uniform_ext_ = UniformExt()
rand_ext_ = RandExt()
rand_like_ext_ = RandLikeExt()
multinomial_ext_ = MultinomialExt()
randn_ = Randn()
randn_like_ = RandnLike()
randint_ = RandInt()
randint_like_ = RandIntLike()
inplace_random_ = InplaceRandom()
generator_step_ = Tensor(12, mstype.int64)


@constexpr
def _set_prim_op_user_data(prim, key, value):
    prim.add_prim_attr(key, value)
    return prim


@_function_forbid_reuse
def random_gamma(shape, alpha, seed=None):
    r"""
    Generate random numbers from the Gamma distribution(s).


    Args:
        shape (Tensor): The shape of random tensor to be generated.
        alpha (Tensor): The :math:`\alpha` distribution parameter.
        seed (int, optional): Random seed, must be non-negative. Default ``None`` .

    Returns:
        Tensor, the shape is `mindspore.ops.concat([shape, rate.shape], axis=0)`.
        The data type is the same as `alpha`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> shape = mindspore.tensor([7, 5], mindspore.int32)
        >>> alpha = mindspore.tensor([0.5, 1.5], mindspore.float32)
        >>> output = mindspore.ops.random_gamma(shape, alpha, seed=5)
        >>> print(output.shape, output.dtype)
        (7, 5, 2) Float32
    """
    seed1, seed2 = _get_seed(seed, "random_gamma")
    random_gamma_op = P.RandomGamma(seed1, seed2)
    random_gamma_op = _set_prim_op_user_data(
        random_gamma_op, "random_cache", False)
    output = random_gamma_op(shape, alpha)
    return output


@constexpr(reuse_result=False)
def _get_seed(op_seed, kernel_name):
    """Get the graph-level seed."""
    return _get_graph_seed(op_seed, kernel_name)


@_function_forbid_reuse
def standard_laplace(shape, seed=None):
    r"""
    Generates random numbers according to the Laplace random number distribution (mean=0, lambda=1).

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        shape (Union[tuple, Tensor]): The shape of returned tensor.
        seed (int, optional): Random number seed. Default ``None`` .

    Returns:
        Tensor

    Raises:
        ValueError: If shape is a tuple containing non-positive items.
        ValueError: If shape is a Tensor, and the rank of the Tensor is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> shape = (4, 4)
        >>> output = mindspore.ops.standard_laplace(shape, seed=5)
        >>> print(f'output shape is {output.shape}')
        output shape is (4, 4)
    """
    seed1, seed2 = _get_seed(seed, "standard_laplace")
    standard_laplace_op = P.StandardLaplace(seed=seed1, seed2=seed2)
    standard_laplace_op = _set_prim_op_user_data(
        standard_laplace_op, "random_cache", False)
    return standard_laplace_op(shape)


@_function_forbid_reuse
def random_categorical(logits, num_sample, seed=0, dtype=mstype.int64):
    r"""
    Generates random samples from a given categorical distribution tensor.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        logits (Tensor): The input tensor. 2-D Tensor with shape :math:`(batch\_size, num\_classes)`.
        num_sample (int):  Number of sample to be drawn. Only constant values is allowed.
        seed (int):  Random seed. Only constant values is allowed. Default: ``0`` .
        dtype (mindspore.dtype): The type of output. Its value must be one of mindspore.int16,
            mindspore.int32 and mindspore.int64. Default: ``mstype.int64`` .

    Returns:
        Tensor, The output Tensor with shape :math:`(batch\_size, num\_samples)`.

    Raises:
        TypeError: If `dtype` is not one of the following: mindspore.int16, mindspore.int32, mindspore.int64.
        TypeError: If `logits` is not a Tensor.
        TypeError: If neither `num_sample` nor `seed` is an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import numpy as np
        >>> logits = Tensor(np.random.random((10, 5)).astype(np.float32), mstype.float32)
        >>> net = ops.random_categorical(logits, 8)
        >>> result = net.shape
        >>> print(result)
        (10, 8)
    """
    random_categorical_ = P.RandomCategorical(dtype)
    random_categorical_ = _set_prim_op_user_data(
        random_categorical_, "random_cache", False)
    return random_categorical_(logits, num_sample, seed)


@_function_forbid_reuse
def multinomial_with_replacement(x, seed, offset, numsamples, replacement=False):
    r"""
    Generate a tensor from a multinomial distribution.

    Note:
        - The rows of input do not need to sum to one (in which case we use the values as weights),
          but must be non-negative, finite and have a non-zero sum.
        - If `seed` is set to be ``-1`` , and `offset` is set to be ``0``, the random number
          generator is seeded by a random seed.

    Args:
        x (Tensor): The 1-D or 2-D input tensor containing probabilities.
        seed (int): Random seed.
        offset (int): Offset.
        numsamples (int): The number of samples to draw.
        replacement (bool, optional): Whether to draw with replacement or not. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[0., 9., 4., 0.]], mindspore.float32)
        >>> mindspore.ops.multinomial_with_replacement(x, 2, 5, 2, True)
        Tensor(shape=[1, 2], dtype=Int64, value=
        [[1, 1]])
    """
    if not isinstance(seed, Tensor):
        if not isinstance(seed, int):
            # PSJit currently only supports the concatenation of f-strings,
            # and does not support the concatenation of normal strings with f-strings.
            raise TypeError(f"For multinomial_with_replacement, "
                            f"the input[seed] must be int, but got {type(seed)}.")
        seed = Tensor(seed, dtype=mstype.int64)
    if not isinstance(offset, Tensor):
        if not isinstance(offset, int):
            raise TypeError(f"For multinomial_with_replacement, "
                            f"the input[offset] must be int, but got {type(offset)}.")
        offset = Tensor(offset, dtype=mstype.int64)
    multinomial_with_replacement_ = P.MultinomialWithReplacement(numsamples=numsamples,
                                                                 replacement=replacement)
    multinomial_with_replacement_ = _set_prim_op_user_data(
        multinomial_with_replacement_, "random_cache", False)
    return multinomial_with_replacement_(x, seed, offset)


@_function_forbid_reuse
def uniform_ext(tensor, a, b, generator=None):
    """
    Generates random numbers in the half-open interval [a, b).

    Args:
        tensor (Tensor): The origin input tensor.
        a (number): The lower bound of the interval.
        b (number): The upper bound of the interval.
        generator (Generator, optional): The random seed. Default: None.

    Raises:
        TypeError: If `a` is larger than `b`.

    Returns:
        Tensor, with the same shape as tensor.

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> x = ops.ones((4, 2))
        >>> generator = mindspore.Generator()
        >>> generator.manual_seed(100)
        >>> result = ops.function.random_func.uniform_ext(x, 1., 2., generator)
        >>> print(result.shape)
        (4, 2)
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    return uniform_ext_(tensor, a, b, seed, offset)


@_function_forbid_reuse
def uniform_(input, from_=0, to=1, *, generator=None):
    r"""
    Update the `input` tensor in place by generating random numbers sampled from uniform distribution in the half-open
    interval :math:`[from\_, to)`.

    .. math::
        P(x)= \frac{1}{to - from\_}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The origin input tensor.
        from_ (Union[number.Number, Tensor], optional): The lower bound of the uniform distribution, it can be a scalar
            value or a tensor of any dimension with a single element. Default: ``0``.
        to (Union[number.Number, Tensor], optional): The upper bound of the uniform distribution, it can be a scalar
            value or a tensor of any dimension with a single element. Default: ``1``.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.

    Returns:
        Tensor, with the same shape and dtype as `input` tensor.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not one of: bool, int8, int16, int32, int64, uint8, float16, float32, float64,
            bfloat16.
        TypeError: If `from_` or `to` is neither a number nor a Tensor.
        TypeError: If dtype of `from` or `to` is not one of: bool, int8, int16, int32, int64, uint8, float32, float64.
        ValueError: If `from_` or `to` is Tensor but contains multiple elements.
        RuntimeError: If `from_` is larger than `to`.

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> x = ops.ones((4, 2))
        >>> generator = mindspore.Generator()
        >>> generator.manual_seed(100)
        >>> result = ops.function.random_func.uniform_(x, 1., 2., generator=generator)
        >>> print(result.shape)
        (4, 2)
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    return inplace_uniform_op(input, from_, to, seed, offset)


@_function_forbid_reuse
def uniform(shape, minval, maxval, seed=None, dtype=mstype.float32):
    """
    Generates random numbers according to the Uniform random number distribution.

    Note:
        The number in tensor minval should be strictly less than maxval at any position after broadcasting.

    Args:
        shape (Union[tuple, Tensor]): The shape of returned tensor.
        minval (Tensor): Defines the minimum possible generated value.
        maxval (Tensor): Defines the maximum possible generated value.
        seed (int): Random number seed. Default ``None`` .
        dtype (mindspore.dtype): Type of the returned tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # For discrete uniform distribution, only one number is allowed for both minval and maxval:
        >>> shape = (4, 2)
        >>> minval = mindspore.tensor(1, mindspore.int32)
        >>> maxval = mindspore.tensor(2, mindspore.int32)
        >>> output = mindspore.ops.uniform(shape, minval, maxval, seed=5, dtype=mindspore.int32)
        >>>
        >>> # For continuous uniform distribution, minval and maxval can be multi-dimentional:
        >>> shape = (3, 1, 2)
        >>> minval = mindspore.tensor([[3, 4], [5, 6]], mindspore.float32)
        >>> maxval = mindspore.tensor([8.0, 10.0], mindspore.float32)
        >>> output = mindspore.ops.uniform(shape, minval, maxval, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
    """
    if not isinstance(minval, Tensor) or not isinstance(maxval, Tensor):
        raise TypeError(
            "For functional operator[uniform], the input[minval] and input[maxval] must be a Tensor.")

    minval_dtype = F.dtype(minval)
    maxval_dtype = F.dtype(maxval)
    const_utils.check_type_valid(
        dtype, [mstype.int32, mstype.float32], 'uniform')
    const_utils.check_tensors_dtype_same(minval_dtype, dtype, "uniform")
    const_utils.check_tensors_dtype_same(maxval_dtype, dtype, "uniform")
    seed1, seed2 = _get_seed(seed, "uniform")
    if const_utils.is_same_type(dtype, mstype.int32):
        random_uniform = P.UniformInt(seed1, seed2)
        random_uniform = _set_prim_op_user_data(
            random_uniform, "random_cache", False)
        value = random_uniform(shape, minval, maxval)
    else:
        uniform_real = P.UniformReal(seed1, seed2)
        uniform_real = _set_prim_op_user_data(
            uniform_real, "random_cache", False)
        uniform_real = uniform_real(shape)
        value = uniform_real * (maxval - minval) + minval
    return value



@_function_forbid_reuse
def exponential_(input, lambd=1, *, generator=None):
    r"""
    exponential
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    return inplace_exponential_op(input, lambd, seed, offset)


@_function_forbid_reuse
def standard_normal(shape, seed=None):
    r"""
    Generates random numbers according to the standard Normal (or Gaussian) random number distribution.

    .. math::
        f(x)=\frac{1}{\sqrt{2 \pi}} e^{\left(-\frac{x^{2}}{2}\right)}

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        shape (Union[tuple, Tensor]): The shape of returned tensor.
        seed (int, optional): Random number Seed. Default ``None`` .

    Returns:
        Tensor

    Raises:
        ValueError: If `shape` is a tuple containing non-positive items.
        ValueError: If shape is a Tensor, and the rank of the Tensor is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> shape = (4, 4)
        >>> output = mindspore.ops.standard_normal(shape, seed=5)
        >>> print(f'output shape is {output.shape}')
        output shape is (4, 4)
    """
    seed1, seed2 = _get_seed(seed, "standard_normal")
    standard_normal_op = P.StandardNormal(seed=seed1, seed2=seed2)
    standard_normal_op = _set_prim_op_user_data(
        standard_normal_op, "random_cache", False)
    return standard_normal_op(shape)


@_function_forbid_reuse
def uniform_candidate_sampler(true_classes,
                              num_true,
                              num_sampled,
                              unique,
                              range_max,
                              seed=0,
                              remove_accidental_hits=False):
    r"""
    Uniform candidate sampler.

    This function samples a set of classes(sampled_candidates) from [0, range_max-1] based on uniform distribution.
    If unique=True, candidates are drawn without replacement, else unique=False with replacement.

    .. warning::
        - The Ascend backend does not support the reproducibility of random numbers, so
          the `seed` parameter has no effect.
        - The Ascend backend does not support dynamic shape scenarios currently.

    Args:
        true_classes (Tensor): A Tensor. The target classes with a Tensor shape of :math:`(batch\_size, num\_true)` .
            The value range of the elements must be :math:`[0, range\_max)`.
        num_true (int): The number of target classes in each training example.
        num_sampled (int): The number of classes to randomly sample. The sampled_candidates will have a shape
            of num_sampled. If unique=True, num_sampled must be less than or equal to range_max.
        unique (bool): Whether all sampled classes in a batch are unique.
        range_max (int): The number of possible classes, must be positive.
        seed (int): Used for random number generation, must be non-negative. If seed has a value of 0,
            the seed will be replaced with a randomly generated value. Default: ``0`` .
        remove_accidental_hits (bool): Whether accidental hit is removed.
            Accidental hit is when one of the true classes matches one of the sample classes.
            Set ``True`` to remove which accidentally sampling the true class as sample class. Default: ``False`` .

    Returns:
        - **sampled_candidates** (Tensor) - The sampled_candidates is independent of the true classes.
          shape: :math:`(num\_sampled, )` .
        - **true_expected_count** (Tensor) - The expected counts under the sampling distribution of each
          of true_classes. shape: :math:`(batch\_size, num\_true)` .
        - **sampled_expected_count** (Tensor) - The expected counts under the sampling distribution of
          each of sampled_candidates. shape: :math:`(num\_sampled, )` .

    Raises:
        TypeError: If neither `num_true` nor `num_sampled` is an int.
        TypeError: If neither `unique` nor `remove_accidental_hits` is a bool.
        TypeError: If neither `range_max` nor `seed` is an int.
        TypeError: If `true_classes` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> data = Tensor(np.array([[1], [3], [4], [6], [3]], dtype=np.int64))
        >>> output1, output2, output3 = ops.uniform_candidate_sampler(data, 1, 3, False, 4, 1)
        >>> print(output1.shape)
        (3,)
        >>> print(output2.shape)
        (5, 1)
        >>> print(output3.shape)
        (3,)
    """
    sampler_op = P.UniformCandidateSampler(num_true,
                                           num_sampled,
                                           unique,
                                           range_max,
                                           seed=seed,
                                           remove_accidental_hits=remove_accidental_hits)
    sampler_op = _set_prim_op_user_data(sampler_op, "random_cache", False)
    sampled_candidates, true_expected_count, sampled_expected_count = sampler_op(
        true_classes)
    return sampled_candidates, true_expected_count, sampled_expected_count


@_function_forbid_reuse
def random_poisson(shape, rate, seed=None, dtype=mstype.float32):
    r"""
    Generate random number Tensor with `shape` according to a Poisson distribution with mean `rate`.


    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        shape (Tensor): The shape of random tensor to be sampled from each poisson distribution, 1-D integer tensor.
        rate (Tensor): The :math:`μ` parameter the distribution is constructed with.
            It represents the mean of poisson distribution
            and also the variance of the distribution.
        seed (int, optional): Random seed, must be non-negative. Default ``None`` .
        dtype (mindspore.dtype): The data type returned. Default ``mstype.float32``.

    Returns:
        Tensor, the shape is `mindspore.ops.concat([shape, rate.shape], axis=0)`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: 1-D shape, 2-D rate, float64 output
        >>> shape = mindspore.tensor([2, 2], mindspore.int64)
        >>> rate = mindspore.tensor([[5.0, 10.0], [5.0, 1.0]], mindspore.float32)
        >>> output = mindspore.ops.random_poisson(shape, rate, seed=5, dtype=mindspore.float64)
        >>> print(output.shape, output.dtype)
        (2, 2, 2, 2) Float64
        >>> # case 2: 1-D shape, scalar rate, int64 output
        >>> shape = mindspore.tensor([2, 2], mindspore.int64)
        >>> rate = mindspore.tensor(5.0, mindspore.float64)
        >>> output = mindspore.ops.random_poisson(shape, rate, seed=5, dtype=mindspore.int64)
        >>> print(output.shape, output.dtype)
        (2, 2) Int64
    """
    seed1, seed2 = _get_seed(seed, "random_poisson")
    prim_random_poisson = P.RandomPoisson(seed1, seed2, dtype)
    prim_random_poisson = _set_prim_op_user_data(
        prim_random_poisson, "random_cache", False)
    value = prim_random_poisson(shape, rate)
    return value


@_function_forbid_reuse
def shuffle(x, seed=None):
    r"""
    Randomly shuffle a tensor along its first dimension.

    Args:
        x (Tensor): The input tensor.
        seed (int, optional): Random seed. Default ``None`` , which is equivalent to 0.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 2, 3, 4], mindspore.float32)
        >>> output = mindspore.ops.shuffle(x, seed=1)
        >>> print(output)
        [3. 4. 2. 1.]
    """
    seed, seed2 = _get_seed(seed, "shuffle")
    random_shuffle_ = RandomShuffle(seed=seed, seed2=seed2)
    random_shuffle_ = _set_prim_op_user_data(
        random_shuffle_, "random_cache", False)
    output = random_shuffle_(x)
    return output


@_function_forbid_reuse
def log_uniform_candidate_sampler(true_classes, num_true=1, num_sampled=5, unique=True, range_max=5, seed=0):
    r"""
    Generates random labels with a log-uniform distribution for sampled_candidates.

    Randomly samples a tensor of sampled classes from the range of integers [0, range_max).

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        true_classes (Tensor): The target classes. With data type of int64 and
          shape :math:`(batch\_size, num\_true)` .
        num_true (int, optional): The number of target classes per training example. Default: ``1`` .
        num_sampled (int, optional): The number of classes to randomly sample. Default: ``5`` .
        unique (bool, optional): Determines whether sample with rejection. If `unique` is ``True`` ,
          all sampled classes in a batch are unique. Default: ``True`` .
        range_max (int, optional): The number of possible classes. When `unique` is ``True`` ,
          `range_max` must be greater than or equal to `num_sampled`. Default: ``5`` .
        seed (int, optional): Random seed, must be non-negative. Default: ``0`` .

    Returns:
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
        ``Ascend`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> output1, output2, output3 = ops.log_uniform_candidate_sampler(
        ... Tensor(np.array([[1, 7], [0, 4], [3, 3]])), 2, 5, True, 5)
        >>> print(output1, output2, output3)
        [3 2 0 4 1]
        [[0.92312991 0.49336370]
         [0.99248987 0.65806371]
         [0.73553443 0.73553443]]
        [0.73553443 0.82625800 0.99248987 0.65806371 0.92312991]

    """

    sampler = P.LogUniformCandidateSampler(
        num_true, num_sampled, unique, range_max, seed)
    sampler = _set_prim_op_user_data(sampler, "random_cache", False)
    return sampler(true_classes)


@_function_forbid_reuse
def choice_with_mask(input_x, count=256, seed=None):
    """
    Generates a random sample as index tensor with a mask tensor from a given tensor.

    The `input_x` must be a tensor whose dimension is not less than 1. If its dimension is greater than or equal to 2,
    the first dimension specifies the number of samples.
    The returned index tensor denotes the index of the nonzero
    sample, the mask tensor denotes which elements in the index tensor are valid.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        input_x (Tensor[bool]): The input tensor.
            The input tensor rank must be greater than or equal to 1 and less than or equal to 5.
        count (int, optional): Number of items expected to get and the number must be greater than 0. Default: ``256`` .
        seed (int, optional): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
            Default: ``None`` .

    Returns:
        Two tensors, the first one is the index tensor and the other one is the mask tensor.

        - **index** (Tensor) - The output shape is 2-D.
        - **mask** (Tensor) - The output shape is 1-D.

    Raises:
        TypeError: If `count` is not an int.
        TypeError: If `seed` is not an int.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones(shape=[240000, 4]).astype(np.bool_))
        >>> output_y, output_mask = ops.choice_with_mask(input_x)
        >>> result = output_y.shape
        >>> print(result)
        (256, 2)
        >>> result = output_mask.shape
        >>> print(result)
        (256,)
    """
    seed1, seed2 = _get_seed(seed, "choice_with_mask")
    choice_with_mask_ = RandomChoiceWithMask(
        count=count, seed=seed1, seed2=seed2)
    choice_with_mask_ = _set_prim_op_user_data(
        choice_with_mask_, "random_cache", False)
    output = choice_with_mask_(input_x)
    return output


@constexpr
def is_cpu_backend():
    """Check if the CPU is used"""
    return context.get_context('device_target') == 'CPU'


@_function_forbid_reuse
def normal_(input, mean=0, std=1, *, generator=None):
    r"""
    Update the `input` tensor in place by generating random numbers sampled from the normal
    distribution which constructed by the parameters `mean` and `std`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The origin input tensor.
        mean (number, optional): the mean of normal distribution. With float data type.
            Default: ``0``.
        std (number, optional): the std of normal distribution. With float data type.
            Default: ``1``.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.

    Returns:
        A tensor that is filled with random numbers that follow a normal distribution and
        that has the same type and shape as the `self` tensor.

    Raises:
        TypeError: If the dtype of `mean` or `std` is not one of: bool, int, float, complex.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> x = mindspore.Tensor(np.array([[1, 2], [3, 4]]), dtype=mindspore.float32)
        >>> output = x.normal_()
        >>> print(output)
        [[0.2788825 1.3305743]
         [1.244194 1.16303174]]
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    return inplace_normal_(input, mean, std, seed, offset)


def normal_ext(mean=0.0, std=1.0, size=None, generator=None):
    r"""
    normal(mean, std, *, generator=None) -> Tensor

    Generates random numbers according to the standard Normal (or Gaussian) random number distribution.

    Args:
        mean (Union[Tensor]): Mean value of each element, the shape of the `mean` tensor
            should be the same as that of the `std` tensor.
        std (Union[Tensor]): Standard deviation for each element, the shape of the `std` tensor
            should be the same as that of the `mean` tensor. The value of `std` should be greater than or equal to 0.

    Keyword Args:
        generator (generator, optional): MindSpore generator. Default: ``None``.

    Returns:
        Outputs a tensor with the same shape as `mean`.

    Raises:
        TypeError: If `mean` or `std` is not Union[float, Tensor].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> mean = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> std = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> output = ops.function.random_func.normal_ext(mean, std)
        >>> print(output.shape)
        (3,)

    .. function:: normal(mean, std) -> Tensor
        :noindex:

    Similar to the function above, but the means are shared among all drawn elements.

    Args:
        mean (float): Mean value of each element.
        std (Tensor): Standard deviation for each element. The value of `std` should be greater
            than or equal to 0.

    Returns:
        Outputs a tensor with the same shape as `std`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> mean = 1.
        >>> std = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> output = ops.function.random_func.normal_ext(mean, std)
        >>> print(output.shape)
        (3,)

    .. function:: normal(mean, std=1.0) -> Tensor
        :noindex:

    Similar to the function above, but the standard deviations are shared among all drawn elements.

    Args:
        mean (Tensor): Mean value of each element.
        std (float, optional): Standard deviation for each element. The value of `std` should be greater
            than or equal to 0. Default: ``1.0``.

    Returns:
        Outputs a tensor with the same shape as `mean`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> mean = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> output = ops.function.random_func.normal_ext(mean, 1.0)
        >>> print(output.shape)
        (3,)

    .. function:: normal(mean, std, size) -> Tensor
        :noindex:

    Similar to the function above, but the means and standard deviations are shared among all drawn elements. The
    result tensor has size given by `size`.

    Args:
        mean (float): Mean value of each element.
        std (float): Standard deviation for each element.
        size (tuple): output shape.

    Returns:
        Outputs a tensor. The shape is specified as `size`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> output = ops.function.random_func.normal_ext(1.0, 2.0, (2, 4))
        >>> print(output.shape)
        (2, 4)
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)

    is_mean_tensor = isinstance(mean, Tensor)
    is_std_tensor = isinstance(std, Tensor)

    if is_mean_tensor and is_std_tensor:
        return normal_tensor_tensor_op(mean, std, seed, offset)
    if is_mean_tensor and not is_std_tensor:
        return normal_tensor_float_op(mean, std, seed, offset)
    if not is_mean_tensor and is_std_tensor:
        return normal_float_tensor_op(mean, std, seed, offset)
    return normal_float_float_op(mean, std, size, seed, offset)


@_function_forbid_reuse
def normal(shape, mean, stddev, seed=None):
    """
    Return a random tensor that conforms to the normal (Gaussian) distribution.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        shape (tuple): The shape of returned tensor.
        mean (Union[Tensor, int, float]): The mean of the normal distribution for the returned tensor.
        stddev (Union[Tensor, int, float]): The standard deviation of the normal distribution for the returned tensor.
        seed (int, optional): Random seed. Default: ``None`` , which is equivalent to 0.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> shape = (3, 1, 2)
        >>> mean = mindspore.tensor([[3, 4], [5, 6]], mindspore.float32)
        >>> stddev = mindspore.tensor(1.0, mindspore.float32)
        >>> output = mindspore.ops.normal(shape, mean, stddev, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
        >>> shape = (3, 1, 3)
        >>> mean = mindspore.tensor([[3, 4, 3], [3, 5, 6]], mindspore.float32)
        >>> stddev = mindspore.tensor(1.0, mindspore.float32)
        >>> output = mindspore.ops.normal(shape, mean, stddev, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 3)
        >>> shape = (3, 1, 3)
        >>> mean = mindspore.tensor([[1, 2, 3], [3, 4, 3], [3, 5, 6]], mindspore.float32)
        >>> stddev = mindspore.tensor(1.0, mindspore.float32)
        >>> output = mindspore.ops.normal(shape, mean, stddev, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 3, 3)
    """
    _check_param("normal", "mean", mean)
    _check_param("normal", "stddev", stddev)
    if not isinstance(mean, Tensor):
        mean = Tensor(mean)
    if not isinstance(stddev, Tensor):
        stddev = Tensor(stddev)
    seed1, seed2 = _get_seed(seed, "normal")
    stdnormal = P.StandardNormal(seed1, seed2)
    stdnormal = _set_prim_op_user_data(stdnormal, "random_cache", False)
    _check_shape(shape)
    random_normal = stdnormal(shape)
    value = random_normal * stddev + mean
    return value


@_function_forbid_reuse
def laplace(shape, mean, lambda_param, seed=None):
    r"""
    Generates random numbers according to the Laplace random number distribution.

    Support broadcasting.

    .. math::
        \text{f}(x;μ,λ) = \frac{1}{2λ}\exp(-\frac{|x-μ|}{λ}),

    where :math:`μ` is the mean, representing `mean`, and :math:`λ` is the scale,
    representing `lambda_param`.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        shape (tuple): The shape specified.
        mean (Tensor): The mean of distribution.
        lambda_param (Tensor): Control the variance of distribution. The
          variance of Laplace distribution is equal to twice the square of `lambda_param` .
        seed (int, optional): Random seed. Default ``None`` represents 0.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> shape = (2, 3)
        >>> mean = mindspore.tensor(1.0, mindspore.float32)
        >>> lambda_param = mindspore.tensor(1.0, mindspore.float32)
        >>> output = mindspore.ops.laplace(shape, mean, lambda_param, seed=5)
        >>> print(output.shape)
        (2, 3)
    """
    mean_dtype = F.dtype(mean)
    lambda_param_dtype = F.dtype(lambda_param)
    const_utils.check_tensors_dtype_same(mean_dtype, mstype.float32, "laplace")
    const_utils.check_tensors_dtype_same(
        lambda_param_dtype, mstype.float32, "laplace")
    seed1, seed2 = _get_seed(seed, "laplace")
    stdlaplace = P.StandardLaplace(seed1, seed2)
    stdlaplace = _set_prim_op_user_data(stdlaplace, "random_cache", False)
    _check_shape(shape)
    rnd = stdlaplace(shape)
    value = rnd * lambda_param + mean
    return value


@_function_forbid_reuse
def gamma(shape, alpha, beta, seed=None):
    r"""
    Generates random numbers according to the Gamma random number distribution.

    Support broadcasting.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        shape (tuple): The shape specified.
        alpha (Tensor): The shape parameter.
        beta (Tensor): The inverse scale parameter.
        seed (int, optional): The random seed, Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> # case 1: alpha_shape is (2, 2)
        >>> shape = (3, 1, 2)
        >>> alpha = mindspore.tensor([[3, 4], [5, 6]], mindspore.float32)
        >>> beta = mindspore.tensor([1.0], mindspore.float32)
        >>> output = mindspore.ops.gamma(shape, alpha, beta, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
        >>> # case 2: alpha_shape is (2, 3), so shape is (3, 1, 3)
        >>> shape = (3, 1, 3)
        >>> alpha = mindspore.tensor([[1, 3, 4], [2, 5, 6]], mindspore.float32)
        >>> beta = mindspore.tensor([1.0], mindspore.float32)
        >>> output = mindspore.ops.gamma(shape, alpha, beta, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 3)
        >>> # case 3: beta_shape is (1, 2), the output is different.
        >>> shape = (3, 1, 2)
        >>> alpha = mindspore.tensor([[3, 4], [5, 6]], mindspore.float32)
        >>> beta = mindspore.tensor([1.0, 2], mindspore.float32)
        >>> output = mindspore.ops.gamma(shape, alpha, beta, seed=5)
        >>> print(output)
        [[[ 2.2132034  5.8855834]
          [ 3.8825176  8.6066265]]
         [[ 3.3981476  7.5805717]
          [ 3.7190282 19.941492 ]]
         [[ 2.9512358  2.5969937]
          [ 3.786061   5.160872 ]]]
        >>> # case 4: beta_shape is (2, 1), the output is different.
        >>> shape = (3, 1, 2)
        >>> alpha = mindspore.tensor([[3, 4], [5, 6]], mindspore.float32)
        >>> beta = mindspore.tensor([[1.0], [2.0]], mindspore.float32)
        >>> output = mindspore.ops.gamma(shape, alpha, beta, seed=5)
        >>> print(output)
        [[[ 5.6085486  7.8280783]
         [ 15.97684  16.116285]]
        [[ 1.8347423  1.713663]
         [ 3.2434065 15.667398]]
        [[ 4.2922077  7.3365674]
         [ 5.3876944  13.159832 ]]]
    """
    seed1, seed2 = _get_seed(seed, "gamma")
    gamma_v = P.Gamma(seed1, seed2)
    gamma_v = _set_prim_op_user_data(gamma_v, "random_cache", False)
    value = gamma_v(shape, alpha, beta)
    return value


@_primexpr
def _generate_shapes(shape):
    """Generate shapes for randn and rand."""
    if not shape:
        size = (1,)
    elif len(shape) == 1:
        if isinstance(shape[0], int):
            size = shape
        elif isinstance(shape[0], list):
            size = tuple(shape[0])
        elif isinstance(shape[0], tuple):
            size = shape[0]
        else:
            raise TypeError(f"If the length of the argument 'shape' is 1, the type of the argument 'shape' must be "
                            f"one of ['int', 'list', 'tuple'], but got {shape[0]}.")
    else:
        for value in shape:
            if not isinstance(value, int):
                raise TypeError(f"If the length of the argument 'shape' is > 1, the type of the argument 'shape' must "
                                f"all be int, but got {value}.")
        size = shape
    return size


@_function_forbid_reuse
def rand(*size, dtype=None, seed=None):
    r"""
    Return a new tensor that fills numbers from the uniform distribution over an interval :math:`[0, 1)`
    based on the given `size` and `dtype`.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        size (Union[int, tuple(int), list(int)]): The shape of the output tensor.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned. Default ``None`` .
        seed (int, optional): Random seed, must be greater or equal to 0. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> print(mindspore.ops.rand((2,3)))
        [[4.1702199e-01 9.9718481e-01 7.2032452e-01]
         [9.3255734e-01 1.1438108e-04 1.2812445e-01]]
    """
    if dtype is None:
        dtype = mstype.float32
    elif dtype not in mstype.float_type:
        raise ValueError(
            f"For 'rand', the 'dtype' must be a float type, but got {dtype}.")
    shape = _generate_shapes(size)
    seed1, seed2 = _get_seed(seed, 'rand')
    rand_op = P.UniformReal(seed1, seed2)
    rand_op = _set_prim_op_user_data(rand_op, "random_cache", False)
    output = rand_op(shape)
    return cast_(output, dtype)


@_function_forbid_reuse
def rand_like(input, seed=None, *, dtype=None):
    r"""
    Return a tensor with the same shape as `input` that is filled with random numbers from a uniform distribution
    on the interval :math:`[0, 1)`.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        input (Tensor): The input tensor.
        seed (int, optional): Random seed, must be greater or equal to 0. Default ``None`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned.
            Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor([[2, 3, 4], [1, 2, 3]])
        >>> print(mindspore.ops.rand_like(a, dtype=mindspore.float32))
        [[4.1702199e-01 9.9718481e-01 7.2032452e-01]
         [9.3255734e-01 1.1438108e-04 1.2812445e-01]]
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"For 'rand_like', the 'input' must be a Tensor, but got {type(input)}")
    if dtype is None:
        dtype = input.dtype
    if dtype not in mstype.float_type:
        raise ValueError(
            f"For 'rand_like', the 'dtype' must be a float type, but got {dtype}.")
    shape = input.shape
    seed1, seed2 = _get_seed(seed, 'rand_like')
    rand_op = P.UniformReal(seed1, seed2)
    rand_op = _set_prim_op_user_data(rand_op, "random_cache", False)
    output = rand_op(shape)
    return cast_(output, dtype)


@_function_forbid_reuse
def rand_ext(*size, generator=None, dtype=None, device=None):
    r"""
    Returns a new tensor that fills numbers from the uniform distribution over an interval :math:`[0, 1)`
    based on the given shape and dtype.

    Args:
        size (Union[int, tuple(int), list(int)]): Shape of the new tensor, e.g. :math:`(2, 3)` or :math:`2`.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.
        dtype (:class:`mindspore.dtype`, optional): Designated tensor dtype. If ``None``,
            `mindspore.float32` will be applied. Default: ``None`` .
        device (str, optional): The specified device of the output tensor. Only ``"Ascend"`` and ``"npu"`` are
            supported. If `device = None`, the value set by :func:`mindspore.set_device` will be used. 
            Default: ``None`` .

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the uniform distribution on
        the interval :math:`[0, 1)`.

    Raises:
        ValueError: If `size` contains negative numbers.
        ValueError: If `device` is ``"GPU"`` .
        RuntimeError: If `device` is ``"CPU"`` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> print(ops.function.random_func.rand_ext(2, 3).shape)
        (2, 3)
    """
    if not generator:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    if size and isinstance(size[0], (tuple, list)):
        size = size[0]
    return rand_ext_(size, seed, offset, dtype, device)


@_function_forbid_reuse
def rand_like_ext(input, *, dtype=None, device=None):
    r"""
    Returns a new tensor that fills numbers from the uniform distribution over an interval :math:`[0, 1)`
    based on the given dtype and shape of the input tensor.

    Args:
        input (Tensor): Input Tensor to specify the output shape and its default dtype.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): Designated tensor dtype, it must be float type. If None,
            the same dtype of `input` will be applied. Default: ``None`` .
        device (str, optional): The specified device of the output tensor. Only ``"Ascend"`` and ``"npu"``
            are supported. If `device = None`, the device of `input` will be used. Default: ``None`` .

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the uniform distribution on
        the interval :math:`[0, 1)`.

    Raises:
        RuntimeError: If `Input` device is ``"CPU"``, and `device` is ``None`` .
        RuntimeError: If `device` is ``"CPU"`` .
        ValueError: If `device` is ``"GPU"`` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> a = Tensor([[2, 3, 4], [1, 2, 3]]).to('Ascend')
        >>> print(ops.function.random_func.rand_like_ext(a, dtype=ms.float32).shape)
        (2, 3)
    """
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_)
    return rand_like_ext_(input, seed, offset, dtype, device)


@_function_forbid_reuse
def randn_ext(*size, generator=None, dtype=None, device=None):
    r"""
    Returns a new tensor filled with numbers from the normal distribution over an interval :math:`[0, 1)`
    based on the given shape and dtype.

    Args:
        size (Union[int, tuple(int), list(int)]): Shape of the new tensor, e.g. :math:`(2, 3)` or :math:`2`.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.
        dtype (:class:`mindspore.dtype`, optional): Designated tensor dtype. If None,
            `mindspore.float32` will be applied. Default: ``None`` .
        device (str, optional): The specified device of the output tensor. ``"Ascend"`` and ``"npu"`` are supported.
            If `device = None`, the value set by :func:`mindspore.set_device` will be used. Default ``None``.

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the normal distribution on
        the interval :math:`[0, 1)`.

    Raises:
        ValueError: If `size` contains negative numbers.
        ValueError: If `device` is ``"GPU"`` .
        RuntimeError: If `device` is ``"CPU"`` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> print(ops.function.random_func.randn_ext(2, 3).shape)
        (2, 3)
    """
    if not generator:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    if size and isinstance(size[0], (tuple, list)):
        size = size[0]
    return randn_(size, seed, offset, dtype, device)


@_function_forbid_reuse
def randn_like_ext(input, *, dtype=None, device=None):
    r"""
    Returns a new tensor filled with numbers from the normal distribution over an interval :math:`[0, 1)`
    based on the given dtype and shape of the input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Input Tensor to specify the output shape and its default dtype.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): Designated Tensor dtype, it must be float type. If ``None``,
            the same dtype of `input` will be applied. Default: ``None`` .
        device (str, optional): The specified device of the output tensor. ``"Ascend"`` and ``"npu"`` are supported.
            If `device = None`, the device of `input` will be used. Default ``None``.

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the normal distribution on
        the interval :math:`[0, 1)`.

    Raises:
        RuntimeError: If `Input` device is ``"CPU"``, and `device` is ``None`` .
        RuntimeError: If `device` is ``"CPU"``.
        ValueError: If `device` is ``"GPU"``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> a = Tensor([[2, 3, 4], [1, 2, 3]])
        >>> print(ops.function.random_func.randn_like_ext(a, dtype=ms.float32).shape)
        (2, 3)
    """
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_)
    return randn_like_(input, seed, offset, dtype, device)


@_function_forbid_reuse
def randint_ext(*args, generator=None, dtype=None, device=None):
    r"""
    randint(low=0, high, size, *, generator=None, dtype=None, device=None) -> Tensor

    Returns a new tensor filled with integer numbers from the uniform distribution over an interval :math:`[low, high)`
    based on the given shape and dtype.

    Args:
        low (int, optional): the lower bound of the generated random number. Default: ``0``.
        high (int): the upper bound of the generated random number
        size (Union[tuple(int), list(int)]): Shape of the new tensor, e.g. :math:`(2, 3)`.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.
        dtype (:class:`mindspore.dtype`, optional): Designated tensor dtype. If None,
            `mindspore.int64` will be applied. Default: ``None`` .
        device (str, optional): The specified device of the output tensor. Only ``"Ascend"`` and ``"npu"`` are
            supported. If `device = None`, the value set by :func:`mindspore.set_device` will be used. 
            Default: ``None`` .

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the uniform distribution on
        the interval :math:`[low, high)`.

    Raises:
        TypeError: If `size` is not a tuple.
        TypeError: If `low` or `high` is not integer.
        RuntimeError: If `device` is ``"CPU"`` .
        ValueError: If `device` is ``"GPU"`` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> print(ops.function.random_func.randint_ext(0, 5, (2, 3)).shape)
        (2, 3)
    """
    if not generator:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    args = list(args)
    if len(args) == 2:
        args = [0] + args
    args += [seed, offset]
    return randint_(*args, dtype=dtype, device=device)


@_function_forbid_reuse
def randint_like_ext(*args, dtype=None, device=None):
    r"""
    randint_like(input, low=0, high, *, dtype=None, device=None) -> Tensor

    Returns a new tensor filled with integer numbers from the uniform distribution over an interval :math:`[low, high)`
    based on the given dtype and shape of the input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Input Tensor to specify the output shape and its default dtype.
        low (int, optional): the lower bound of the generated random number. Default: ``0``.
        high (int): the upper bound of the generated random number

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): Designated tensor dtype. If None,
            the same dtype of `input` will be applied. Default: ``None`` .
        device (str, optional): The specified device of the output tensor. Only ``"Ascend"`` and ``"npu"`` are
            supported. If `device = None`, the device of `input` will be used. Default: ``None`` .

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the uniform distribution on
        the interval :math:`[low, high)`.

    Raises:
        TypeError: If `low` or `high` is not integer.
        RuntimeError: If `Input` device is ``"CPU"``, and `device` is ``None`` .
        RuntimeError: If `device` is ``"CPU"`` .
        ValueError: If `device` is ``"GPU"`` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> a = Tensor([[2, 3, 4], [1, 2, 3]]).to('Ascend')
        >>> low = 0
        >>> high = 5
        >>> print(ops.function.random_func.randint_like_ext(a, low, high, dtype=ms.int32).shape)
        (2, 3)
    """
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_)
    args = list(args)
    if len(args) == 2:
        args = [args[0], 0, args[1]]
    args += [seed, offset]
    return randint_like_(*args, dtype=dtype, device=device)


@_function_forbid_reuse
def random_(input, from_=0, to=None, *, generator=None):
    r"""
    Fill the input tensor with numbers sampled from a discrete uniform distribution
    over an interval :math:`[low, high)`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): input tensor.
        from_ (int, optional): the lower bound of the generated random number. Default: 0.
        to (int, optional): the upper bound of the generated random number. By default it's the upper limit of
            the input data type. Default: ``None``.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.

    Returns:
        The input tensor.

    Raises:
        TypeError: If `from_` or `to` is not integer.
        ValueError: If `from_` >= `to`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> a = Tensor([[2, 3, 4], [1, 2, 3]])
        >>> from_ = 0
        >>> to = 5
        >>> print(ops.function.random_func.random_(a, from_, to).shape)
        (2, 3)
    """
    if not generator:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    return inplace_random_(input, from_, to, seed, offset)


@_function_forbid_reuse
def randn(*size, dtype=None, seed=None):
    r"""
    Return a new tensor with given shape and dtype, filled with random numbers
    from the standard normal distribution.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        size (Union[int, tuple(int), list(int)]): Shape of the output tensor.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned.
            Default ``None`` .
        seed (int, optional): Random seed, must be non-negative. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> print(mindspore.ops.randn((2, 2)))
        [[ 0.30639967 -0.42438635]
         [-0.4287376   1.3054721 ]]
    """
    if dtype is None:
        dtype = mstype.float32
    elif dtype not in mstype.float_type:
        raise ValueError(
            f"For 'randn', the 'dtype' must be a float type, but got {dtype}.")
    shape = _generate_shapes(size)
    seed1, seed2 = _get_seed(seed, 'randn')
    rand_op = P.StandardNormal(seed1, seed2)
    rand_op = _set_prim_op_user_data(rand_op, "random_cache", False)
    output = rand_op(shape)
    return cast_(output, dtype)


@_function_forbid_reuse
def randn_like(input, seed=None, *, dtype=None):
    r"""
    Return a tensor with the same shape as `input`, filled with random numbers from the standard normal
    distribution.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        input (Tensor): The input tensor.
        seed (int, optional): Random seed, must be non-negative. Default ``None`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
        >>> print(mindspore.ops.randn_like(a, dtype=mindspore.float32))
        [[ 0.30639967 -0.42438635 -0.20454668]
         [-0.4287376   1.3054721   0.64747655]]
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"For 'randn_like', the 'input' must be a Tensor, but got {type(input)}")
    if dtype is None:
        dtype = mstype.float32
    if dtype not in mstype.float_type:
        raise ValueError(
            f"For 'randn_like', the 'dtype' must be a float type, but got {dtype}.")
    shape = input.shape
    seed1, seed2 = _get_seed(seed, 'randn_like')
    rand_op = P.StandardNormal(seed1, seed2)
    rand_op = _set_prim_op_user_data(rand_op, "random_cache", False)
    output = rand_op(shape)
    return cast_(output, dtype)


@_function_forbid_reuse
def randint(low, high, size, seed=None, *, dtype=None):
    r"""
    Return a tensor whose elements are random integers in the range of [ `low` , `high` ) .

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        low (int): Start value of interval.
        high (int): End value of interval.
        size (tuple): Shape of the output tensor.
        seed (int, optional): Random seed, must be non-negative. Default ``None`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned.
            Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> print(mindspore.ops.randint(1, 10, (2,3)))
        [[4 9 7]
         [9 1 2]]
    """
    if dtype is None:
        dtype = mstype.int64
    elif dtype not in mstype.int_type:
        raise ValueError(
            f"For 'randint', the 'dtype' must be an int type, but got {dtype}.")
    if not isinstance(size, tuple):
        raise ValueError(
            f"For 'randint', the input 'size' must be a tuple, but got {size}.")
    if not isinstance(low, int) or isinstance(low, bool):
        raise TypeError(
            f"For 'randint_like', 'low' must be an int, but got {type(low)}.")
    if not isinstance(high, int) or isinstance(high, bool):
        raise TypeError(
            f"For 'randint_like', 'high' must be an int, but got {type(high)}.")
    seed1, seed2 = _get_seed(seed, 'randint')
    rand_op = P.UniformInt(seed1, seed2)
    rand_op = _set_prim_op_user_data(rand_op, "random_cache", False)
    low_ = Tensor(low, mstype.int32)
    high_ = Tensor(high, mstype.int32)
    output = rand_op(size, low_, high_)
    return cast_(output, dtype)


@_function_forbid_reuse
def randint_like(input, low, high, seed=None, *, dtype=None):
    r"""
    Returns a tensor with the same shape as `input` whose elements are random integers in the range
    of [ `low` , `high` ) .

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        input (Tensor): The input tensor.
        low(int): Start value of interval.
        high(int): End value of interval.
        seed (int, optional): Random seed, must be non-negative. Default ``None`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned.
            Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
       >>> import mindspore
       >>> a = mindspore.tensor([[1, 2, 3], [3, 2, 1]])
       >>> print(mindspore.ops.randint_like(a, 1, 10))
       [[4 9 7]
        [9 1 2]]
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"For 'randint_like', the 'input' must be a Tensor, but got {type(input)}")
    if dtype is None:
        dtype = input.dtype
    if dtype not in mstype.int_type:
        raise ValueError(
            f"For 'randint_like', the 'dtype' must be an int type, but got {dtype}.")
    if not isinstance(low, int) or isinstance(low, bool):
        raise TypeError(
            f"For 'randint_like', 'low' must be an int, but got {type(low)}.")
    if not isinstance(high, int) or isinstance(high, bool):
        raise TypeError(
            f"For 'randint_like', 'high' must be an int, but got {type(high)}.")
    size = input.shape
    seed1, seed2 = _get_seed(seed, 'randint_like')
    rand_op = P.UniformInt(seed1, seed2)
    rand_op = _set_prim_op_user_data(rand_op, "random_cache", False)
    low_ = Tensor(low, mstype.int32)
    high_ = Tensor(high, mstype.int32)
    size_ = Tensor(size, mstype.int32)
    output = rand_op(size_, low_, high_)
    return cast_(output, dtype)


def randperm_ext(n, *, generator=None, dtype=mstype.int64):
    r"""
    Generates random permutation of integers from 0 to n-1.

    Args:
        n (Union[Tensor, int]): size of the permutation. int or Tensor with shape: () or (1,) and
            data type int64. The value of `n` must be greater than zero.

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.
        dtype (mindspore.dtype, optional): The type of output. Default: mstype.int64.

    Returns:
        Tensor with shape (n,) and type `dtype`.

    Raises:
        TypeError: If `dtype` is not supported.
        ValueError: If `n` is a negative or 0 element.
        ValueError: If `n` is larger than the maximal data of the set dtype.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> from mindspore import dtype as mstype
        >>> n = 4
        >>> output = ops.randperm_ext(n, dtype=mstype.int64)
        >>> print(output.shape)
        (4,)
    """
    if not generator:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    return randperm_ext_(n, seed, offset, dtype)


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
          and must be non-negative. Default: ``None`` , which will be treated as 0.

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
    random_poisson_op = P.Poisson(seed1, seed2)
    random_poisson_op = _set_prim_op_user_data(
        random_poisson_op, "random_cache", False)
    value = random_poisson_op(shape, mean)
    return value


@_function_forbid_reuse
def multinomial(input, num_samples, replacement=True, seed=None):
    r"""
    Generate a tensor from a multinomial distribution.

    The polynomial distribution is a probability distribution that generalizes the binomial distribution formula to
    multiple states. In the polynomial distribution, each event has a fixed probability, and the sum of these
    probabilities is 1.

    The purpose of this interface is to perform `num_samples` sampling
    on the input `input`, and the output tensor is the index of the input tensor for each sampling.
    The values in `input` represent the probability of selecting the corresponding index for each sampling.

    Here is an extreme example for better understanding. Suppose we have an input probability tensor with
    values `[90 / 100, 10 / 100, 0]`, which means we can sample three indices,
    namely index 0, index 1, and index 2, with probabilities of 90%, 10%, and 0%, respectively. We perform n samplings,
    and the resulting sequence is the calculation result of the polynomial distribution, with a length equal to the
    number of samplings.

    In case 1 of the sample code, we perform two non-replacement samplings (`replacement` is `False`).
    Since the probability of selecting index 0 is 90% for each sampling, the first result is most likely to be index 0.
    Since the probability of selecting index 2 is 0, index 2 cannot appear in the sampling result. Therefore, the
    second result must be index 1, and the resulting sequence is `[0, 1]`.

    In case 2 of the sample code, we perform 10 replacement samplings (`replacement` is `True`).
    As expected, about 90% of the sampling results are index 0.

    In case 3 of the sample code, we extend the input to 2 dimensions, and the sampling results
    in each dimension also match our sampling expectations.

    Note:
        The rows of input do not need to sum to one (in which case we use the values as weights),
        but must be non-negative, finite and have a non-zero sum. When using values as weights, it can be understood as
        normalizing the input along the last dimension.

    .. warning::
        The Ascend backend does not support the reproducibility of random numbers, so
        the `seed` parameter has no effect.

    Args:
        input (Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool, optional): Whether to draw with replacement or not. Default ``True`` .
        seed (int, optional): Random seed. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is False.
        >>> input1 = mindspore.tensor([90 / 100, 10 / 100, 0])
        >>> input2 = mindspore.tensor([90, 10, 0])
        >>> # input1 and input2 have the same meaning.
        >>> mindspore.ops.multinomial(input1, 2, replacement=False)
        Tensor(shape=[2], dtype=Int32, value= [0, 1])
        >>> mindspore.ops.multinomial(input2, 2, replacement=False)
        Tensor(shape=[2], dtype=Int32, value= [1, 0])
        >>>
        >>> # case 2: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is True.
        >>> mindspore.ops.multinomial(input1, 10)
        Tensor(shape=[10], dtype=Int32, value= [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        >>>
        >>> # case 3: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is True.
        >>> # rank is 2
        >>> input3 = mindspore.tensor([[90, 10, 0], [10, 90, 0]], mindspore.float32)
        >>> output = mindspore.ops.multinomial(input3, 10)
        >>> print(output)
        [[0 0 0 0 0 0 0 0 0 0]
         [1 0 1 1 1 1 1 1 1 1]]
    """
    def _check_valid_dim(dim, name):
        if dim not in (1, 2):
            raise ValueError(
                f"For '{name}', the dimension of inputs must be 1d or 2d, but got {dim}.")

    _check_valid_dim(len(shape_(input)), "multinomial")
    seed1, seed2 = _get_seed(seed, "multinomial")
    if not replacement:
        if shape_(input)[-1] < num_samples:
            const_utils.raise_value_error(f"For 'multinomial', the 'num_samples' must be less than "
                                          f"the last dimension of input without 'replacement', "
                                          f"but got 'num_samples': {num_samples} and "
                                          f"'replacement': {replacement}")
        n_dist = 1
        if len(shape_(input)) > 1:
            n_dist = shape_(input)[-2]
        random_uniform_real = P.UniformReal(seed1, seed2)
        random_cache_op = _set_prim_op_user_data(
            random_uniform_real, "random_cache", False)
        random_uniform = random_cache_op((n_dist * shape_(input)[-1],))
        if n_dist != 1:
            random_uniform = reshape_(
                random_uniform, (n_dist, shape_(input)[-1]))

        vals = real_div_(log_(random_uniform), input + 1e-6)
        _, indices = top_k_(vals, num_samples)
        return indices
    random_nomial = P.Multinomial(seed1, seed2)
    random_nomial = _set_prim_op_user_data(
        random_nomial, "random_cache", False)
    return random_nomial(input, num_samples)


@_function_forbid_reuse
def multinomial_ext(input, num_samples, replacement=False, *, generator=None):
    r"""
    Returns a tensor sampled from the multinomial probability distribution located in the corresponding
    row of the input tensor.

    The polynomial distribution is a probability distribution that generalizes the binomial distribution formula to
    multiple states. In the polynomial distribution, each event has a fixed probability, and the sum of these
    probabilities is 1. The purpose of the :func:`mindspore.mint.multinomial` interface
    is to perform `num_samples` sampling
    on the input `input`, and the output tensor is the index of the input tensor for each sampling.
    The values in `input` represent the probability of selecting the corresponding index for each sampling.

    Here is an extreme example for better understanding. Suppose we have an input probability tensor with
    values `Tensor([90 / 100, 10 / 100, 0], mindspore.float32)`, which means we can sample three indices,
    namely index 0, index 1, and index 2, with probabilities of 90%, 10%, and 0%, respectively. We perform n samplings,
    and the resulting sequence is the calculation result of the polynomial distribution, with a length equal to the
    number of samplings.

    In case 1 of the sample code, we perform two non-replacement samplings (`replacement` is `False`).
    The calculation result is most likely `[0, 1]`, and less likely `[1, 0]`. Since the probability of selecting
    index 0 is 90% for each sampling, the first result is most likely to be index 0. Since the probability of selecting
    index 2 is 0, index 2 cannot appear in the sampling result. Therefore, the second result must be index 1,
    and the resulting sequence is `[0, 1]`.

    In case 2 of the sample code, we perform 10 replacement samplings (`replacement` is `True`).
    As expected, about 90% of the sampling results are index 0.

    In case 3 of the sample code, we extend the input to 2 dimensions, and the sampling results
    in each dimension also match our sampling expectations.

    Note:
        The rows of input do not need to sum to one (in which case we use the values as weights),
        but must be non-negative, finite and have a non-zero sum.
        When using values as weights, it can be understood as normalizing the input along the last dimension.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor containing probabilities, must be 1 or 2 dimensions, with float32 data type.
        num_samples (int): Number of samples to draw.
        replacement (bool, optional): Whether to draw with replacement or not. Default: ``False`` .

    Keyword Args:
        generator (generator, optional): MindSpore generator. Default: ``None``.

    Returns:
        Tensor, dtype is Int64.
        If `input` is a vector, out is a vector of size `num_samples`.
        If `input` is a matrix with m rows, out is an matrix of shape(m * num_samples).

    Raises:
        TypeError: If `input` is not a Tensor whose dtype is not in float16, float32, float64 or bfloat16.
        TypeError: If `num_samples` is not an int, a Scalar of int
            or a Tensor with shape[1,] and only one int element.
        RuntimeError: If :math:`\text{num_samples} <= 0`.
        RuntimeError: If `replacement` is False, :math:`\text{num_samples} > shape` of the last dimension of `input`.
        RuntimeError: If shape of the last dimension of `input` exceeds ``2^24``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> # case 1: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is False.
        >>> input1 = Tensor([90 / 100, 10 / 100, 0], mindspore.float32)
        >>> input2 = Tensor([90, 10, 0], mindspore.float32)
        >>> # input1 and input2 have the same meaning.
        >>> output1 = ops.multinomial_ext(input1, 2)
        >>> output2 = ops.multinomial_ext(input2, 2)
        >>> print(output1)
        [0 1]
        >>> print(output2)
        [0 1]
        >>> print(len(output1))
        2
        >>> print(len(output2))
        2
        >>> # case 2: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is True.
        >>> output3 = ops.multinomial_ext(input1, 10, replacement=True)
        >>> print(output3)
        [0 0 1 0 0 0 0 0 0 0]
        >>> print(len(output3))
        10
        >>> # case 3: The output is random, and the length of the output is the same as num_sample.
        >>> # replacement is True.
        >>> # rank is 2
        >>> input4 = Tensor([[90, 10, 0], [10, 90, 0]], mstype.float32)
        >>> output4 = ops.multinomial_ext(input4, 10, replacement=True)
        >>> print(output4)
        [[0 0 0 0 0 0 0 0 1 0]
         [1 1 1 1 1 0 1 1 1 1]]
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    return multinomial_ext_(input, num_samples, replacement, seed, offset)


def _check_shape(input_shape):
    """Check 'shape' value."""
    if not isinstance(input_shape, tuple):
        const_utils.raise_type_error(
            f"Type of 'shape' must be tuple, but got: {type(input_shape)}")
    for item in input_shape:
        if not isinstance(item, int):
            const_utils.raise_type_error(
                f"Elements of 'shape' must be int, but got: {type(item)}")
        if item < 1:
            const_utils.raise_value_error(
                f"Elements of 'shape' must be positive int, but got: {item}")
    return True


def _check_param(op_name, param_name, param_value):
    """Check type of param_value is Tensor, int, or float."""
    if not isinstance(param_value, (Tensor, int, float, np.ndarray)):
        const_utils.raise_type_error("For '{}', the type of '{}' must be Tensor, int, or float, "
                                     "but got: {}".format(op_name, param_name, type(param_value)))
    return True


__all__ = [
    'standard_laplace', 'random_categorical', 'uniform', 'standard_normal', 'random_gamma',
    'uniform_candidate_sampler', 'random_poisson', 'log_uniform_candidate_sampler', 'shuffle', 'choice_with_mask',
    'normal', 'laplace', 'gamma', 'poisson', 'multinomial', 'rand', 'rand_like',
    'randn', 'randn_like',
    'randint', 'randint_like', 'multinomial_with_replacement', 'randperm'
]
__all__.sort()
