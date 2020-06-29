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


class Normal(PrimitiveWithInfer):
    r"""
    Generates random numbers according to the Normal (or Gaussian) random number distribution.
    It is defined as:

    .. math::
        \text{f}(x;μ,σ) = \frac{1}{σ\sqrt{2π}}\exp(-\frac{1}{2}(\frac{x-μ}{σ})^2),

    Args:
        seed (int): Seed data is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **mean** (Tensor) - The mean μ distribution parameter, The mean specifies the location of the peak.
            With float32 data type.
        - **stddev** (Tensor) - the deviation σ distribution parameter. With float32 data type.

    Outputs:
        Tensor, has the shape 'shape' input and dtype as float32.

    Examples:
        >>> shape = (4, 16)
        >>> mean = Tensor(1.0, mstype.float32)
        >>> stddev = Tensor(1.0, mstype.float32)
        >>> normal = P.Normal(seed=2)
        >>> output = normal(shape, mean, stddev)
    """

    @prim_attr_register
    def __init__(self, seed=0):
        """Init Normal"""
        self.init_prim_io_names(inputs=['shape', 'mean', 'stddev'], outputs=['output'])
        validator.check_value_type('seed', seed, [int], self.name)

    def __infer__(self, shape, mean, stddev):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        validator.check_tensor_type_same({"mean": mean["dtype"]}, [mstype.float32], self.name)
        validator.check_tensor_type_same({"stddev": stddev["dtype"]}, [mstype.float32], self.name)
        broadcast_shape = get_broadcast_shape(mean['shape'], stddev['shape'], self.name)
        broadcast_shape = get_broadcast_shape(broadcast_shape, shape_v, self.name)
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.float32,
            'value': None}
        return out


class Gamma(PrimitiveWithInfer):
    r"""
    Produces random positive floating-point values x, distributed according to probability density function:

    .. math::
        \text{P}(x|α,β) = \frac{\exp(-x/β)}{{β^α}\cdot{\Gamma(α)}}\cdot{x^{α-1}},

    Args:
        seed (int): Seed data is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **alpha** (Tensor) - The α distribution parameter.
          It is also known as the shape parameter. With float32 data type.
        - **beta** (Tensor) - The β distribution parameter.
          It is also known as the scale parameter. With float32 data type.

    Outputs:
        Tensor, has the shape 'shape' input and dtype as float32.

    Examples:
        >>> shape = (4, 16)
        >>> alpha = Tensor(1.0, mstype.float32)
        >>> beta = Tensor(1.0, mstype.float32)
        >>> gamma = P.Gamma(seed=3)
        >>> output = normal(shape, alpha, beta)
    """

    @prim_attr_register
    def __init__(self, seed=0):
        """Init Gamma"""
        self.init_prim_io_names(inputs=['shape', 'alpha', 'beta'], outputs=['output'])
        validator.check_value_type('seed', seed, [int], self.name)

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
        seed (int): Seed data is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **mean** (Tensor) - μ parameter the distribution was constructed with.
          The parameter defines mean number of occurrences of the event. With float32 data type.

    Outputs:
        Tensor, has the shape 'shape' input and dtype as int32.

    Examples:
        >>> shape = (4, 16)
        >>> mean = Tensor(5.0, mstype.float32)
        >>> poisson = P.Poisson(seed=5)
        >>> output = poisson(shape, mean)
    """

    @prim_attr_register
    def __init__(self, seed=0):
        """Init Poisson"""
        self.init_prim_io_names(inputs=['shape', 'mean'], outputs=['output'])
        validator.check_value_type('seed', seed, [int], self.name)

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
    Produces random integer values i, uniformly distributed on the closed interval [a, b], that is,
    distributed according to the discrete probability function:

    .. math::
        \text{P}(i|a,b) = \frac{1}{b-a+1},

    Args:
        seed (int): Seed data is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **a** (Tensor) - The a distribution parameter.
          It defines the minimum possibly generated value. With int32 data type.
        - **b** (Tensor) - The b distribution parameter.
          It defines the maximum possibly generated value. With int32 data type.

    Outputs:
        Tensor, has the shape 'shape' input and dtype as int32.

    Examples:
        >>> shape = (4, 16)
        >>> a = Tensor(1, mstype.int32)
        >>> b = Tensor(5, mstype.int32)
        >>> uniform_int = P.UniformInt(seed=10)
        >>> output = uniform_int(shape, a, b)
    """

    @prim_attr_register
    def __init__(self, seed=0):
        """Init UniformInt"""
        self.init_prim_io_names(inputs=['shape', 'a', 'b'], outputs=['output'])
        validator.check_value_type('seed', seed, [int], self.name)

    def __infer__(self, shape, a, b):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        validator.check_tensor_type_same({"a": a["dtype"]}, [mstype.int32], self.name)
        validator.check_tensor_type_same({"b": b["dtype"]}, [mstype.int32], self.name)
        broadcast_shape = get_broadcast_shape(a['shape'], b['shape'], self.name)
        broadcast_shape = get_broadcast_shape(broadcast_shape, shape_v, self.name)
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.int32,
            'value': None}
        return out


class UniformReal(PrimitiveWithInfer):
    r"""
    Produces random floating-point values i, uniformly distributed on the interval [a, b), that is,\
    distributed according to the probability density function:

    .. math::
        \text{P}(i|a,b) = \frac{1}{b-a},

    Args:
        seed (int): Seed data is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.
        - **a** (Tensor) - The a distribution parameter.
          It defines the minimum possibly generated value. With float32 data type.
        - **b** (Tensor) - The b distribution parameter.
          It defines the maximum possibly generated value. With float32 data type.

    Outputs:
        Tensor, has the shape 'shape' input and dtype as int32.

    Examples:
        >>> shape = (4, 16)
        >>> a = Tensor(1.0, mstype.float32)
        >>> b = Tensor(5.0, mstype.float32)
        >>> uniform_real = P.UniformReal(seed=10)
        >>> output = uniform_real(shape, a, b)
    """

    @prim_attr_register
    def __init__(self, seed=0):
        """Init UniformReal"""
        self.init_prim_io_names(inputs=['shape', 'a', 'b'], outputs=['output'])
        validator.check_value_type('seed', seed, [int], self.name)

    def __infer__(self, shape, a, b):
        shape_v = shape["value"]
        if shape_v is None:
            raise ValueError(f"For {self.name}, shape must be const.")
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        for i, shape_i in enumerate(shape_v):
            validator.check_integer("shape[%d]" % i, shape_i, 0, Rel.GT, self.name)
        validator.check_tensor_type_same({"a": a["dtype"]}, [mstype.float32], self.name)
        validator.check_tensor_type_same({"b": b["dtype"]}, [mstype.float32], self.name)
        broadcast_shape = get_broadcast_shape(a['shape'], b['shape'], self.name)
        broadcast_shape = get_broadcast_shape(broadcast_shape, shape_v, self.name)
        out = {
            'shape': broadcast_shape,
            'dtype': mstype.float32,
            'value': None}
        return out


class RandomChoiceWithMask(PrimitiveWithInfer):
    """
    Generates a random samply as index tensor with a mask tensor from a given tensor.

    The input must be a tensor of rank >= 1. If its rank >= 2, the first dimension specify the number of sample.
    The index tensor and the mask tensor have the fixed shapes. The index tensor denotes the index of the nonzero
    sample, while the mask tensor denotes which elements in the index tensor are valid.

    Args:
        count (int): Number of items expected to get and the number should be greater than 0. Default: 256.
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Inputs:
        - **input_x** (Tensor[bool]) - The input tensor.

    Outputs:
        Two tensors, the first one is the index tensor and the other one is the mask tensor.

        - **index** (Tensor) - The output has shape between 2-D and 5-D.
        - **mask** (Tensor) - The output has shape 1-D.

    Examples:
        >>> rnd_choice_mask = P.RandomChoiceWithMask()
        >>> input_x = Tensor(np.ones(shape=[240000, 4]).astype(np.bool))
        >>> output_y, output_mask = rnd_choice_mask(input_x)
    """

    @prim_attr_register
    def __init__(self, count=256, seed=0, seed2=0):
        """Init RandomChoiceWithMask"""
        validator.check_value_type("count", count, [int], self.name)
        validator.check_integer("count", count, 0, Rel.GT, self.name)
        validator.check_value_type('seed', seed, [int], self.name)
        validator.check_value_type('seed2', seed2, [int], self.name)

    def infer_shape(self, x_shape):
        validator.check_integer("input_x rank", len(x_shape), 1, Rel.GE, self.name)
        return ([self.count, len(x_shape)], [self.count])

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, [mstype.bool_], self.name)
        return (mstype.int32, mstype.bool_)


class RandomCategorical(PrimitiveWithInfer):
    """
    Generates random samples from a given categorical distribution tensor.

    Args:
        dtype (mindspore.dtype): The type of output. Its value should be one of [mindspore.int16,
            mindspore.int32, mindspore.int64]. Default: mindspore.int64.

    Inputs:
        - **logits** (Tensor) - The input tensor. 2-D Tensor with shape [batch_size, num_classes].
        - **num_sample** (int) - Number of sample to be drawn. Only constant values is allowed.
        - **seed** (int) - Random seed. Default: 0.

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
        """Init RandomCategorical"""
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
