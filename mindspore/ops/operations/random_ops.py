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

from ..._checkparam import ParamValidator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register


class RandomChoiceWithMask(PrimitiveWithInfer):
    """
    Generates a random samply as index tensor with a mask tensor from a given tensor.

    The input must be a tensor of rank >= 2, the first dimension specify the number of sample.
    The index tensor and the mask tensor have the same and fixed shape. The index tensor denotes the index
    of the nonzero sample, while the mask tensor denotes which element in the index tensor are valid.

    Args:
        count (int): Number of items expected to get. Default: 256.
        seed (int): Random seed.
        seed2 (int): Random seed2.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tuple, two tensors, the first one is the index tensor and the other one is the mask tensor.

    Examples:
        >>> rnd_choice_mask = RandomChoiceWithMask()
        >>> input_x = Tensor(np.ones(shape=[240000, 4]), ms.bool_)
        >>> output_y, output_mask = rnd_choice_mask(input_x)
    """

    @prim_attr_register
    def __init__(self, count=256, seed=0, seed2=0):
        """Init RandomChoiceWithMask"""
        validator.check_type("count", count, [int])
        validator.check_integer("count", count, 0, Rel.GT)
        validator.check_type('seed', seed, [int])
        validator.check_type('seed2', seed2, [int])

    def infer_shape(self, x_shape):
        validator.check_shape_length("input_x shape", len(x_shape), 1, Rel.GE)
        return ([self.count, len(x_shape)], [self.count])

    def infer_dtype(self, x_dtype):
        validator.check_subclass('x_dtype', x_dtype, mstype.tensor)
        validator.check_typename('x_dtype', x_dtype, [mstype.bool_])
        return (mstype.int32, mstype.bool_)
