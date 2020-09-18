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

"""Operators for random."""

from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register
from .._utils import get_broadcast_shape


class StandardNormal(PrimitiveWithInfer):
    r"""
    Generates random numbers according to the standard Normal (or Gaussian) random number distribution.

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.

    Outputs:
        Tensor. The shape is the same as the input `shape`. The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> stdnormal = P.StandardNormal(seed=2)
        >>> output = stdnormal(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize StandardNormal"""
        self.init_prim_io_names(inputs=['shape'], outputs=['output'])
        validator.check_integer("seed", seed, 0, Rel.GE, self.name)
        validator.check_integer("seed2", seed2, 0, Rel.GE, self.name)

    def __infer__(self, shape):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        out = {
            'shape': shape_v,
            'dtype': mstype.float32,
            'value': None}
        return out


class StandardLaplace(PrimitiveWithInfer):
    r"""
    Generates random numbers according to the Laplace random number distribution (mean=0, lambda=1).
    It is defined as:

    .. math::
        \text{f}(x;0,1) = \frac{1}{2}\exp(-|x|),

    Args:
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.

    Outputs:
        Tensor. The shape that the input 'shape' denotes. The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> stdlaplace = P.StandardLaplace(seed=2)
        >>> output = stdlaplace(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize StandardLaplace"""
        self.init_prim_io_names(inputs=['shape'], outputs=['output'])
        validator.check_value_type('seed', seed, [int], self.name)
        validator.check_value_type('seed2', seed2, [int], self.name)

    def __infer__(self, shape):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        out = {
            'shape': shape_v,
            'dtype': mstype.float32,
            'value': None}
        return out



class Gamma(PrimitiveWithInfer):
    r"""
    Produces random positive floating-point values x, distributed according to probability density function:

    .. math::
        \text{P}(x|α,β) = \frac{\exp(-x/β)}{{β^α}\cdot{\Gamma(α)}}\cdot{x^{α-1}},

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **alpha** (Tensor) - The α distribution parameter. It must be greater than 0.
          It is also known as the shape parameter with float32 data type.
        - **beta** (Tensor) - The β distribution parameter. It must be greater than 0.
          It is also known as the scale parameter with float32 data type.

    Outputs:
        Tensor. The shape must be the broadcasted shape of Input "shape" and shapes of alpha and beta.
        The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> alpha = Tensor(1.0, mstype.float32)
        >>> beta = Tensor(1.0, mstype.float32)
        >>> gamma = P.Gamma(seed=3)
        >>> output = Gamma(shape, alpha, beta)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize Gamma"""
        self.init_prim_io_names(inputs=['shape', 'alpha', 'beta'], outputs=['output'])
        validator.check_integer("seed", seed, 0, Rel.GE, self.name)
        validator.check_integer("seed2", seed2, 0, Rel.GE, self.name)

    def __infer__(self, shape, alpha, beta):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        validator.check_tensor_type_same({"alpha": alpha["dtype"]}, [mstype.float32], self.name)
        validator.check_tensor_type_same({"beta": beta["dtype"]}, [mstype.float32], self.name)
        broadcast_shape = get_broadcast_shape(alpha['shape'], beta['shape'], self.name)
        broadcast_shape = get_broadcast_shape(broadcast_shape, shape_v, self.name)
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.float32,
            'value': None}
        return out


class Poisson(PrimitiveWithInfer):
    r"""
    Produces random non-negative integer values i, distributed according to discrete probability function:

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!},

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

    Examples:
        >>> shape = (4, 16)
        >>> mean = Tensor(5.0, mstype.float32)
        >>> poisson = P.Poisson(seed=5)
        >>> output = poisson(shape, mean)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize Poisson"""
        self.init_prim_io_names(inputs=['shape', 'mean'], outputs=['output'])
        validator.check_integer("seed", seed, 0, Rel.GE, self.name)
        validator.check_integer("seed2", seed2, 0, Rel.GE, self.name)

    def __infer__(self, shape, mean):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        validator.check_tensor_type_same({"mean": mean["dtype"]}, [mstype.float32], self.name)
        broadcast_shape = get_broadcast_shape(mean['shape'], shape_v, self.name)
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.int32,
            'value': None}
        return out


class UniformInt(PrimitiveWithInfer):
    r"""
    Produces random integer values i, uniformly distributed on the closed interval [minval, maxval), that is,
    distributed according to the discrete probability function:

    .. math::
        \text{P}(i|a,b) = \frac{1}{b-a+1},

    Note:
        The number in tensor minval must be strictly less than maxval at any position after broadcasting.

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **minval** (Tensor) - The distribution parameter, a.
          It defines the minimum possibly generated value, with int32 data type. Only one number is supported.
        - **maxval** (Tensor) - The distribution parameter, b.
          It defines the maximum possibly generated value, with int32 data type. Only one number is supported.

    Outputs:
        Tensor. The shape is the same as the input 'shape', and the data type is int32.

    Examples:
        >>> shape = (4, 16)
        >>> minval = Tensor(1, mstype.int32)
        >>> maxval = Tensor(5, mstype.int32)
        >>> uniform_int = P.UniformInt(seed=10)
        >>> output = uniform_int(shape, minval, maxval)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize UniformInt"""
        self.init_prim_io_names(inputs=['shape', 'minval', 'maxval'], outputs=['output'])
        validator.check_integer("seed", seed, 0, Rel.GE, self.name)
        validator.check_integer("seed2", seed2, 0, Rel.GE, self.name)

    def __infer__(self, shape, minval, maxval):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        validator.check_tensor_type_same({"minval": minval["dtype"]}, [mstype.int32], self.name)
        validator.check_tensor_type_same({"maxval": maxval["dtype"]}, [mstype.int32], self.name)
        minval_shape = minval['shape']
        maxval_shape = maxval['shape']
        validator.check("dim of minval", len(minval_shape), '0(scalar)', 0, Rel.EQ, self.name)
        validator.check("dim of maxval", len(maxval_shape), '0(scalar)', 0, Rel.EQ, self.name)
        out = {
            'shape': shape_v,
            'dtype': mstype.int32,
            'value': None}
        return out


class UniformReal(PrimitiveWithInfer):
    r"""
    Produces random floating-point values i, uniformly distributed to the interval [0, 1).

    Args:
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.

    Outputs:
        Tensor. The shape that the input 'shape' denotes. The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> uniformreal = P.UniformReal(seed=2)
        >>> output = uniformreal(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Initialize UniformReal"""
        self.init_prim_io_names(inputs=['shape'], outputs=['output'])
        validator.check_integer("seed", seed, 0, Rel.GE, self.name)
        validator.check_integer("seed2", seed2, 0, Rel.GE, self.name)

    def __infer__(self, shape):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        out = {
            'shape': shape_v,
            'dtype': mstype.float32,
            'value': None}
        return out


class RandomChoiceWithMask(PrimitiveWithInfer):
    """
    Generates a random sample as index tensor with a mask tensor from a given tensor.

    The input must be a tensor of rank not less than 1. If its rank is greater than or equal to 2,
    the first dimension specifies the number of samples.
    The index tensor and the mask tensor have the fixed shapes. The index tensor denotes the index of the nonzero
    sample, while the mask tensor denotes which elements in the index tensor are valid.

    Args:
        count (int): Number of items expected to get and the number must be greater than 0. Default: 256.
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Inputs:
        - **input_x** (Tensor[bool]) - The input tensor.
            The input tensor rank must be greater than or equal to 1 and less than or equal to 5.

    Outputs:
        Two tensors, the first one is the index tensor and the other one is the mask tensor.

        - **index** (Tensor) - The output shape is 2-D.
        - **mask** (Tensor) - The output shape is 1-D.

    Examples:
        >>> rnd_choice_mask = P.RandomChoiceWithMask()
        >>> input_x = Tensor(np.ones(shape=[240000, 4]).astype(np.bool))
        >>> output_y, output_mask = rnd_choice_mask(input_x)
    """

    @prim_attr_register
    def __init__(self, count=256, seed=0, seed2=0):
        """Initialize RandomChoiceWithMask"""
        validator.check_value_type("count", count, [int], self.name)
        validator.check_integer("count", count, 0, Rel.GT, self.name)
        validator.check_value_type('seed', seed, [int], self.name)
        validator.check_value_type('seed2', seed2, [int], self.name)

    def infer_shape(self, x_shape):
        validator.check_integer("input_x rank", len(x_shape), 1, Rel.GE, self.name)
        validator.check_integer("input_x rank", len(x_shape), 5, Rel.LE, self.name)
        return ([self.count, len(x_shape)], [self.count])

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, [mstype.bool_], self.name)
        return (mstype.int32, mstype.bool_)


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

    Examples:
        >>> class Net(nn.Cell):
        >>>   def __init__(self, num_sample):
        >>>     super(Net, self).__init__()
        >>>     self.random_categorical = P.RandomCategorical(mindspore.int64)
        >>>     self.num_sample = num_sample
        >>>   def construct(self, logits, seed=0):
        >>>     return self.random_categorical(logits, self.num_sample, seed)
        >>>
        >>> x = np.random.random((10, 5)).astype(np.float32)
        >>> net = Net(8)
        >>> output = net(Tensor(x))
    """

    @prim_attr_register
    def __init__(self, dtype=mstype.int64):
        """Initialize RandomCategorical"""
        self.dtype = dtype

        valid_values = (mstype.int32, mstype.int16, mstype.int64)
        validator.check_type_name("dtype", dtype, valid_values, self.name)
        self.init_prim_io_names(inputs=['logits', 'num_samples', 'seed'],
                                outputs=['output'])

    def __infer__(self, logits, num_samples, seed):
        logits_dtype = logits['dtype']
        valid_types = (mstype.float32, mstype.float16, mstype.float64)
        validator.check_tensor_type_same({'logits': logits_dtype}, valid_types, self.name)
        num_samples_v = num_samples['value']
        seed_v = seed['value']
        validator.check_value_type('num_samples', num_samples_v, (int,), self.name)
        validator.check_value_type('seed', seed_v, (int,), self.name)
        validator.check_integer("num_samples", num_samples_v, 0, Rel.GT, self.name)
        x_shape = list(logits['shape'])
        if len(x_shape) != 2:
            raise ValueError("RandomCategorical shape should be 2-dimension.")
        ndim = len(x_shape) - 1
        x_shape[ndim] = num_samples_v
        return {'shape': (x_shape),
                'dtype': (self.dtype),
                'value': None}


class Multinomial(PrimitiveWithInfer):
    r"""
    Returns a tensor sampled from the multinomial probability distribution located in the corresponding
    row of tensor input.

    Note:
        The rows of input do not need to sum to one (in which case we use the values as weights),
        but must be non-negative, finite and have a non-zero sum.
    Args:
        seed (int): Seed data is used as entropy source for Random number engines to generate pseudo-random numbers.
          Must be non-negative. Default: 0.
    Inputs:
        - **input** (Tensor[float32]) - the input tensor containing the cumsum of probabilities, must be 1 or 2
            dimensions.
        - **num_samples** (int32) - number of samples to draw.

    Outputs:
        Tensor with the same rows as input, each row has num_samples sampled indices.

    Examples:
        >>> input = Tensor([0., 9., 4., 0.], mstype.float32)
        >>> multinomial = P.Multinomial(seed=10)
        >>> output = multinomial(input, 2)
    """

    @prim_attr_register
    def __init__(self, seed=0):
        """init"""
        validator.check_value_type("seed", seed, [int], self.name)
        validator.check_integer("seed", seed, 0, Rel.GE, self.name)
        self.init_prim_io_names(inputs=['input', 'num_sample'], outputs=['output'])

    def __infer__(self, inputs, num_samples):
        input_shape = inputs["shape"]
        if len(input_shape) != 1 and len(input_shape) != 2:
            raise ValueError("input dim must be 1 or 2")
        validator.check_tensor_type_same({'inputs': inputs['dtype']}, [mstype.float32], self.name)
        num_samples_value = num_samples["value"]
        if num_samples_value is None:
            raise ValueError(f"For {self.name}, shape nust be const")
        validator.check_value_type("num_samples", num_samples_value, (int,), self.name)
        validator.check_integer("num_samples", num_samples_value, 0, Rel.GT, None)
        y_shape = (num_samples_value,)
        if len(input_shape) == 2:
            y_shape = (input_shape[0], num_samples_value)
        out = {
            "shape": y_shape,
            "dtype": mstype.int32,
            "value": None}
        return out
