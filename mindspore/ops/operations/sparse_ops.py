# coding: utf-8

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

"""Operators for sparse operators."""

from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register

class SparseToDense(PrimitiveWithInfer):
    """
    Converts a sparse representation into a dense tensor.

    Inputs:
        - **indices** (Tensor) - The indices of sparse representation.
        - **values** (Tensor) - Values corresponding to each row of indices.
        - **dense_shape** (tuple) - An int tuple which specifies the shape of dense tensor.

    Returns:
        Tensor, the shape of tensor is `dense_shape`.

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> dense_shape = (3, 4)
        >>> out = ops.SparseToDense()(indices, values, dense_shape)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize index_select"""
        self.init_prim_io_names(inputs=['indices', 'values', 'dense_shape'], outputs=['output'])

    def __infer__(self, indices, values, dense_shape):
        validator.check_subclass("indices", indices['dtype'], mstype.tensor, self.name)
        validator.check_subclass("values", values['dtype'], mstype.tensor, self.name)
        out = {'shape': dense_shape['value'],
               'dtype': values['dtype'],
               'value': None}
        return out
