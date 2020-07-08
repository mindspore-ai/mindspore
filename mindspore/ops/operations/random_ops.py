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


class StandardNormal(PrimitiveWithInfer):
    r"""
    Generates random numbers according to the standard Normal (or Gaussian) random number distribution.

    Args:
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Inputs:
        - **shape** (tuple) - The shape of random tensor to be generated. Only constant value is allowed.

    Outputs:
        Tensor. The shape should be the broadcasted shape of Input "shape" and shapes of mean and stddev.
        The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> stdnormal = P.StandardNormal(seed=2)
        >>> output = stdnormal(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0):
        """Init StandardNormal"""
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
